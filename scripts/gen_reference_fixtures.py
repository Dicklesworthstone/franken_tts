#!/usr/bin/env python3
"""Freeze native-reference ConformanceExact fixtures for Qwen3-TTS 12Hz Base.

This is deliberately an oracle tool, not an inference frontend.  It runs the full,
checked-out upstream package at the source pin and a locally materialized weights directory at
the HF pin.  It refuses package drift, source drift, network fallback, and an existing output
directory.  The default device is CUDA because only native-device fixtures may become the
correctness oracle; ``--device cpu --cpu-smoke`` exists solely for repeatability diagnostics.

Example (the output directory must not exist):

  python scripts/gen_reference_fixtures.py \
    --source-dir /path/to/Qwen3-TTS \
    --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
    --corpus docs/conformance/oracle_corpus.json \
    --output /secure/fixtures/qwen3-tts-12hz-r1

Corpus schema (paths are relative to the corpus file):

  {"schema_version": 1, "cases": [{
    "id": "short-en", "text": "Hello.", "language": "English",
    "reference_audio": "consented.wav", "reference_text": "Hello."
  }]}

Each case is run as x-vector-only and ICL, in non-streaming and streaming prompt modes.
Fixtures contain stage arrays (``.npy``) plus hash-anchored JSON manifests.  Reference audio is
never copied: its SHA-256 is recorded instead.  Treat the output as sensitive research data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

REPO = Path(__file__).resolve().parent.parent
PINNED_GH_REV = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
PINNED_HF_REV = "5d83992436eae1d760afd27aff78a71d676296fc"
PINNED_RUNTIME = {
    "qwen-tts": "0.1.1",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
    "librosa": "0.11.0",
    "soundfile": "0.13.1",
}
MANIFEST = REPO / "docs" / "truth-pack" / "MANIFEST.sha256"
WEIGHTS = REPO / "docs" / "truth-pack" / "WEIGHTS.lfs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> NoReturn:
    raise RuntimeError(f"REFUSING: {message}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error.msg} at line {error.lineno}")


def package_version(name: str) -> str:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"missing required oracle package {name}") from error
    return version.split("+", 1)[0]


def assert_runtime() -> dict[str, str]:
    actual = {name: package_version(name) for name in PINNED_RUNTIME}
    mismatches = [
        f"{name}={actual[name]} (need {expected})"
        for name, expected in PINNED_RUNTIME.items()
        if actual[name] != expected
    ]
    if mismatches:
        fail("runtime pin mismatch: " + "; ".join(mismatches))
    return actual


def assert_source_pin(source_dir: Path) -> None:
    if not (source_dir / ".git").exists():
        fail(f"--source-dir must be a full git checkout, not a curated snapshot: {source_dir}")
    git = shutil.which("git")
    if git is None:
        fail("git is required to assert the source revision")
    result = subprocess.run(
        [git, "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        fail(f"cannot read source revision: {result.stderr.strip()}")
    actual = result.stdout.strip()
    if actual != PINNED_GH_REV:
        fail(f"source revision {actual} != pinned {PINNED_GH_REV}")


def manifest_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, _class, rel = line.split(maxsplit=2)
        hashes[rel] = digest
    return hashes


def assert_model_pin(model_dir: Path) -> dict[str, str]:
    if not model_dir.is_dir():
        fail(f"model directory does not exist: {model_dir}")
    expected = manifest_hashes()
    for rel in ("hf/config.json", "hf/generation_config.json", "hf/speech_tokenizer/config.json"):
        actual_path = model_dir / rel.removeprefix("hf/")
        if not actual_path.is_file():
            fail(f"model directory is missing {actual_path.name}")
        actual = sha256(actual_path)
        if actual != expected[rel]:
            fail(f"hash mismatch for {actual_path}: {actual} != {expected[rel]}")

    weights = load_json(WEIGHTS)["files"]
    verified: dict[str, str] = {}
    for entry in weights:
        path = model_dir / entry["path"]
        if not path.is_file():
            fail(f"model directory is missing weight {entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            fail(f"weight hash mismatch for {entry['path']}: {actual} != {entry['sha256']}")
        verified[entry["path"]] = actual
    return verified


def sanitize(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def tensor_items(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield tensors from nested module inputs/outputs without serializing Python internals."""
    import torch

    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}.{index}" if prefix else str(index))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_items(item, f"{prefix}.{key}" if prefix else str(key))
    elif hasattr(value, "items") and callable(value.items):
        # Hugging Face ModelOutput is an OrderedDict-like object, not necessarily a dict.
        # This captures codec-encoder `audio_codes` as well as conventional hidden states.
        for key, item in value.items():
            yield from tensor_items(item, f"{prefix}.{key}" if prefix else str(key))
    elif hasattr(value, "last_hidden_state"):
        yield from tensor_items(value.last_hidden_state, prefix or "last_hidden_state")


@dataclass
class SavedTensor:
    path: str
    sha256: str
    shape: list[int]
    dtype: str


class HookRecorder:
    """Forward-hook recorder whose labels retain phase and invocation order."""

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.context = "unscoped"
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.saved: list[SavedTensor] = []
        self.handles: list[Any] = []

    @contextlib.contextmanager
    def in_context(self, name: str) -> Iterator[None]:
        old = self.context
        self.context = sanitize(name)
        try:
            yield
        finally:
            self.context = old

    def save(self, logical_name: str, value: Any) -> None:
        import numpy as np
        import torch

        for suffix, tensor in tensor_items(value):
            counter_key = f"{self.context}/{logical_name}/{suffix}"
            count = self.counts[counter_key]
            self.counts[counter_key] += 1
            rel = Path("stages") / self.context / sanitize(logical_name) / f"{sanitize(suffix)}.{count:03d}.npy"
            path = self.case_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                fail(f"refusing to overwrite fixture array {path}")
            captured = tensor.detach().cpu().contiguous()
            # NumPy has no portable bfloat16 dtype. Preserve the oracle bits rather than silently
            # widening them: consumers reconstruct BF16 from the uint16 payload named below.
            if captured.dtype == torch.bfloat16:
                array = captured.view(torch.uint16).numpy()
                recorded_dtype = "bfloat16_bits_le"
            else:
                array = captured.numpy()
                recorded_dtype = str(array.dtype)
            np.save(path, array, allow_pickle=False)
            self.saved.append(
                SavedTensor(
                    path=str(rel), sha256=sha256(path), shape=list(array.shape), dtype=recorded_dtype
                )
            )

    def hook(self, name: str, module: Any) -> None:
        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            self.save(f"{name}.input", {"args": args, "kwargs": kwargs})

        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            self.save(f"{name}.output", output)

        self.handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self.handles.append(module.register_forward_hook(post_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_hooks(model: Any, recorder: HookRecorder) -> None:
    """Install only named ConformanceExact seams; avoid an unbounded every-op trace."""
    talker = model.talker
    recorder.hook("talker.input", talker)
    recorder.hook("talker.codec_head", talker.codec_head)
    for index, layer in enumerate(talker.model.layers):
        recorder.hook(f"talker.layer_{index:02d}", layer)

    predictor = talker.code_predictor
    recorder.hook("microdecoder.input", predictor.model)
    for index, layer in enumerate(predictor.model.layers):
        recorder.hook(f"microdecoder.layer_{index:02d}", layer)
    for index, head in enumerate(predictor.lm_head):
        recorder.hook(f"microdecoder.head_{index + 1:02d}", head)

    recorder.hook("speaker_encoder", model.speaker_encoder)
    for index, block in enumerate(model.speaker_encoder.blocks):
        recorder.hook(f"speaker_encoder.block_{index}", block)

    tokenizer_model = model.speech_tokenizer.model
    recorder.hook("codec_encoder", tokenizer_model.encoder)
    decoder = tokenizer_model.decoder
    recorder.hook("codec_decoder.input", decoder)
    for index, layer in enumerate(decoder.pre_transformer.layers):
        recorder.hook(f"codec_decoder.transformer_layer_{index:02d}", layer)
    for index, blocks in enumerate(decoder.upsample):
        for block_index, block in enumerate(blocks):
            recorder.hook(f"codec_decoder.upsample_{index}_{block_index}", block)
    for index, block in enumerate(decoder.decoder):
        recorder.hook(f"codec_decoder.block_{index:02d}", block)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        fail("corpus must contain schema_version=1 and a cases array")
    case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for case in data["cases"]:
        required = ("id", "text", "reference_audio", "reference_text")
        if not isinstance(case, dict) or any(not case.get(field) for field in required):
            fail(f"every corpus case requires non-empty {required}")
        case_id = str(case["id"])
        if sanitize(case_id) != case_id or case_id in case_ids:
            fail(f"case id must be unique and filesystem-safe: {case_id!r}")
        case_ids.add(case_id)
        reference = (path.parent / str(case["reference_audio"])).resolve()
        if not reference.is_file():
            fail(f"missing reference audio for {case_id}: {reference}")
        normalized = dict(case)
        normalized["reference_audio"] = reference
        normalized.setdefault("language", "Auto")
        normalized.setdefault("max_new_tokens", 2)
        if int(normalized["max_new_tokens"]) < 2:
            fail(f"{case_id}: max_new_tokens must be at least 2 (upstream minimum)")
        cases.append(normalized)
    return cases


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        fail(f"refusing to overwrite fixture metadata {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(wrapper: Any, case: dict[str, Any], output_dir: Path, recorder: HookRecorder) -> dict[str, Any]:
    import numpy as np
    import torch

    case_dir = output_dir / case["id"]
    case_dir.mkdir()
    modes = (
        ("xvector_non_streaming", True, True),
        ("xvector_streaming", True, False),
        ("icl_non_streaming", False, True),
        ("icl_streaming", False, False),
    )
    entries: list[dict[str, Any]] = []
    for mode_name, xvector_only, non_streaming in modes:
        mode_dir = case_dir / mode_name
        mode_dir.mkdir()
        recorder.case_dir = mode_dir
        recorder.saved = []
        recorder.counts.clear()
        with recorder.in_context("prompt_build"):
            prompt_items = wrapper.create_voice_clone_prompt(
                ref_audio=str(case["reference_audio"]),
                ref_text=str(case["reference_text"]),
                x_vector_only_mode=xvector_only,
            )
            voice_prompt = wrapper._prompt_items_to_voice_clone_prompt(prompt_items)
            input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(str(case["text"]))])
            ref_ids = (
                None
                if xvector_only
                else [wrapper._tokenize_texts([wrapper._build_ref_text(str(case["reference_text"]))])[0]]
            )
            recorder.save("prompt.text_ids", input_ids[0])
            if ref_ids is not None:
                recorder.save("prompt.reference_ids", ref_ids[0])
            recorder.save("prompt.speaker_embedding", voice_prompt["ref_spk_embedding"][0])
            if voice_prompt["ref_code"][0] is not None:
                recorder.save("prompt.reference_codec_codes", voice_prompt["ref_code"][0])

        with recorder.in_context("talker_free_running"):
            codes_list, hidden_list = wrapper.model.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_prompt,
                languages=[str(case["language"])],
                non_streaming_mode=non_streaming,
                max_new_tokens=int(case["max_new_tokens"]),
                do_sample=False,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                repetition_penalty=1.0,
                subtalker_dosample=False,
                subtalker_top_k=1,
                subtalker_top_p=1.0,
                subtalker_temperature=1.0,
            )
            codes = codes_list[0]
            talker_hidden = hidden_list[0]
            recorder.save("talker.codec_codes", codes)
            recorder.save("talker.generated_hidden", talker_hidden)

        for frame_index in range(codes.shape[0]):
            with recorder.in_context(f"teacher_forced_frame_{frame_index:04d}"):
                with torch.inference_mode():
                    logits, _loss = wrapper.model.talker.forward_sub_talker_finetune(
                        codec_ids=codes[frame_index : frame_index + 1],
                        talker_hidden_states=talker_hidden[frame_index : frame_index + 1],
                    )
                recorder.save("microdecoder.teacher_forced_logits", logits)

        codes_for_decode = codes
        reference_codes = voice_prompt["ref_code"][0]
        if reference_codes is not None:
            codes_for_decode = torch.cat([reference_codes.to(codes.device), codes], dim=0)
        with recorder.in_context("codec_decode"):
            wavs, sample_rate = wrapper.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
            waveform = np.asarray(wavs[0], dtype=np.float32)
            if reference_codes is not None:
                cut = round(int(reference_codes.shape[0]) / max(int(codes_for_decode.shape[0]), 1) * waveform.shape[0])
                waveform = waveform[cut:]
            recorder.save("codec.generated_waveform", torch.from_numpy(waveform))

        files = [item.__dict__ for item in recorder.saved]
        mode_manifest = {
            "mode": mode_name,
            "x_vector_only_mode": xvector_only,
            "non_streaming_mode": non_streaming,
            "sample_rate": int(sample_rate),
            "generated_frames": int(codes.shape[0]),
            "generated_codes_sha256": hashlib.sha256(codes.detach().cpu().numpy().tobytes()).hexdigest(),
            "files": files,
        }
        write_json(mode_dir / "manifest.json", mode_manifest)
        entries.append({"mode": mode_name, "manifest_sha256": sha256(mode_dir / "manifest.json")})
    return {
        "id": case["id"],
        "reference_audio_sha256": sha256(case["reference_audio"]),
        "modes": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="full Qwen3-TTS checkout at the pinned Git revision")
    parser.add_argument("--model-dir", type=Path, required=True, help="local HF model directory at the pinned weights revision")
    parser.add_argument("--corpus", type=Path, required=True, help="versioned JSON corpus definition")
    parser.add_argument("--output", type=Path, required=True, help="new fixture directory; it must not already exist")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-smoke", action="store_true", help="acknowledge that a CPU run is not a native-device golden")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cpu" and not args.cpu_smoke:
        fail("CPU fixtures are smoke-only; pass --cpu-smoke to label them as such")
    if args.output.exists():
        fail(f"output already exists; choose a new directory rather than overwriting {args.output}")
    if args.output.parent.exists() and not args.output.parent.is_dir():
        fail(f"output parent is not a directory: {args.output.parent}")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    source_dir = args.source_dir.resolve()
    model_dir = args.model_dir.resolve()
    corpus_path = args.corpus.resolve()
    assert_source_pin(source_dir)
    runtime = assert_runtime()
    weight_hashes = assert_model_pin(model_dir)
    cases = load_corpus(corpus_path)

    sys.path.insert(0, str(source_dir))
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    imported_from = Path(sys.modules["qwen_tts"].__file__).resolve()
    if source_dir not in imported_from.parents:
        fail(f"qwen_tts imported from {imported_from}, not --source-dir {source_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        fail("native oracle requires CUDA, but torch.cuda.is_available() is false")

    device_map = "cuda:0" if args.device == "cuda" else "cpu"
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    wrapper.model.train(False)
    args.output.mkdir(parents=True)
    provenance = {
        "schema_version": 1,
        "oracle_class": "native_cuda" if args.device == "cuda" else "cpu_smoke_only",
        "source_pin": PINNED_GH_REV,
        "weights_pin": PINNED_HF_REV,
        "runtime": runtime,
        "device": str(wrapper.device),
        "attn_implementation": "eager",
        "dtype": "bfloat16",
        "generation": {"do_sample": False, "top_k": 1, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.0},
        "weight_hashes": weight_hashes,
        "corpus_sha256": sha256(corpus_path),
        "generation_config": load_json(model_dir / "generation_config.json"),
        "command": sys.argv,
    }
    write_json(args.output / "provenance.json", provenance)

    recorder = HookRecorder(args.output)
    install_hooks(wrapper.model, recorder)
    try:
        corpus_entries = [run_case(wrapper, case, args.output, recorder) for case in cases]
    finally:
        recorder.close()
    root_manifest = {
        "schema_version": 1,
        "provenance_sha256": sha256(args.output / "provenance.json"),
        "cases": corpus_entries,
    }
    write_json(args.output / "fixture_manifest.json", root_manifest)
    print(json.dumps({"fixture_manifest": str(args.output / "fixture_manifest.json"), "sha256": sha256(args.output / "fixture_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
