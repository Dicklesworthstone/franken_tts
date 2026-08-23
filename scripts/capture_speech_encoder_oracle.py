#!/usr/bin/env python3
"""Capture the speech-tokenizer ENCODER oracle (bead frankentts-p1-codec-encoder-snt).

Runs the pinned reference stack (docs/truth-pack/snapshots/gh qwen_tts +
transformers==4.57.3 in .oracle-venv, CPU float32) over deterministic synthetic
waveforms and records the encoded 16-group codec codes as the EXACT comparison
target for `ftts_model_qwen::speech_encoder`.

The waveforms are defined by integer-seeded arithmetic (xorshift64 noise + an
integer-phase sawtooth, all element-wise in float64 then cast to float32), so the
Rust side regenerates them bit-identically and the fixture only needs to carry
codes — no audio bytes in the repo.

Run from the repo root:
    .oracle-venv/bin/python scripts/capture_speech_encoder_oracle.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "docs/truth-pack/snapshots/gh"))

import numpy as np
import torch
import transformers

EXPECTED_TRANSFORMERS = "4.57.3"
MODEL_DIR = Path.home() / ".cache/franken_tts/model/speech_tokenizer"
OUT = REPO / "crates/ftts-conformance/tests/fixtures/speech_encoder_oracle.json"

# (name, samples, seed): 60000 hits a whole-frame boundary neighborhood, 50000
# forces the ceil-to-frame extra padding on several conv stages.
CASES = [("tone-noise-60000", 60_000, 0x1234_5678_9ABC_DEF0),
         ("tone-noise-50000", 50_000, 0x0F1E_2D3C_4B5A_6978)]


def synthetic_wave(samples: int, seed: int) -> np.ndarray:
    state = np.uint64(seed)
    noise = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        # xorshift64, mirrored exactly on the Rust side.
        state ^= np.uint64((int(state) << 13) & 0xFFFF_FFFF_FFFF_FFFF)
        state ^= state >> np.uint64(7)
        state ^= np.uint64((int(state) << 17) & 0xFFFF_FFFF_FFFF_FFFF)
        noise[i] = (int(state) >> 40) / float(1 << 24) - 0.5
    index = np.arange(samples, dtype=np.float64)
    saw = (np.mod(index, 240.0) / 240.0 - 0.5)
    wave64 = noise * 0.3 + saw * 0.4
    return wave64.astype(np.float32)


def main() -> None:
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise SystemExit(
            f"oracle venv drift: transformers {transformers.__version__} != {EXPECTED_TRANSFORMERS}; "
            "re-pin before capturing (PIN_RECORD.md)"
        )
    torch.set_num_threads(1)
    # The snapshot's package __init__ chain pulls a 25 Hz module missing from the
    # snapshot, so the 12 Hz pair is loaded as a synthetic package directly — its
    # only relative import is the sibling configuration module.
    import importlib.util
    import types

    pkg_dir = REPO / "docs/truth-pack/snapshots/gh/qwen_tts/core/tokenizer_12hz"
    package = types.ModuleType("tok12hz")
    package.__path__ = [str(pkg_dir)]
    sys.modules["tok12hz"] = package
    for stem in ("configuration_qwen3_tts_tokenizer_v2", "modeling_qwen3_tts_tokenizer_v2"):
        spec = importlib.util.spec_from_file_location(f"tok12hz.{stem}", pkg_dir / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"tok12hz.{stem}"] = module
        spec.loader.exec_module(module)
    Qwen3TTSTokenizerV2Model = sys.modules["tok12hz.modeling_qwen3_tts_tokenizer_v2"].Qwen3TTSTokenizerV2Model

    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32
    ).eval()

    fixture = {
        "oracle_class": "cpu_fp32",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "note": "codes are frames-major [T][16], semantic group first; waveform is "
                "regenerated from the recorded (samples, seed) by the consuming test",
        "cases": [],
    }
    with torch.no_grad():
        for name, samples, seed in CASES:
            wave = torch.from_numpy(synthetic_wave(samples, seed))[None]
            mask = torch.ones(1, samples, dtype=torch.long)
            out = model.encode(wave, padding_mask=mask, return_dict=True)
            codes = out.audio_codes[0]  # [frames, 16] after the wrapper transpose
            assert codes.shape[1] == 16, codes.shape
            fixture["cases"].append({
                "name": name,
                "samples": samples,
                "seed": seed,
                "frames": int(codes.shape[0]),
                "codes": codes.to(torch.int64).tolist(),
            })
            print(f"{name}: {codes.shape[0]} frames captured")

    OUT.write_text(json.dumps(fixture) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
