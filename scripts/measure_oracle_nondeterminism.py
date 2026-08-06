"""Derive a repeatability envelope from independently frozen oracle fixtures.

The output is intentionally a machine-readable tolerance source.  It refuses to overwrite an
existing artifact, compares every array in every supplied fixture root, and records both the
per-seam maximum and the Contract-A rung summaries that a ladder runner can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error


def fixture_arrays(root: Path) -> dict[str, Path]:
    arrays = {str(path.relative_to(root)): path for path in root.rglob("*.npy")}
    if not arrays:
        raise RuntimeError(f"{root}: no fixture arrays found")
    return arrays


def seam_for(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) < 4 or parts[2] != "stages":
        raise RuntimeError(f"unrecognized fixture-array path {relative}")
    return "/".join(parts[3:-1])


def record_difference(
    stage: dict[str, Any], left: np.ndarray, right: np.ndarray, pair: str, path: str
) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError(
            f"{pair}: {path} mismatch: {left.dtype}/{left.shape} vs {right.dtype}/{right.shape}"
        )
    if np.issubdtype(left.dtype, np.floating):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        kind = "floating"
    else:
        difference = np.abs(left.astype(np.int64) - right.astype(np.int64))
        kind = "discrete"
    maximum = float(difference.max(initial=0.0))
    differing = int(np.count_nonzero(difference))
    stage["array_pairs"] += 1
    stage["elements_compared"] += int(left.size)
    stage["differing_elements"] += differing
    stage["max_abs"] = max(stage["max_abs"], maximum)
    if stage["kind"] != kind:
        stage["kind"] = "mixed"
    if differing and stage["first_divergence"] is None:
        index = int(np.flatnonzero(difference)[0])
        stage["first_divergence"] = {"pair": pair, "path": path, "flat_index": index}


def rung_summary(stages: dict[str, dict[str, Any]], selector: Any) -> dict[str, Any]:
    selected = {name: value for name, value in stages.items() if selector(name)}
    if not selected:
        return {"status": "not_observed", "source_seams": [], "max_abs": None}
    return {
        "status": "observed",
        "source_seams": sorted(selected),
        "max_abs": max(value["max_abs"] for value in selected.values()),
        "differing_elements": sum(value["differing_elements"] for value in selected.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [root.resolve() for root in args.fixture_root]
    output = args.output.resolve()
    if len(roots) < 2:
        raise RuntimeError("need at least two independent fixture roots")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise RuntimeError(f"output parent does not exist: {output.parent}")

    runs: list[dict[str, Any]] = []
    arrays_by_root: dict[Path, dict[str, Path]] = {}
    expected_paths: set[str] | None = None
    for root in roots:
        provenance_path = root / "provenance.json"
        manifest_path = root / "fixture_manifest.json"
        if not provenance_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"{root}: missing provenance.json or fixture_manifest.json")
        provenance = load_json(provenance_path)
        if provenance.get("oracle_class") != "cpu_fp32_fallback":
            raise RuntimeError(f"{root}: expected cpu_fp32_fallback provenance")
        arrays = fixture_arrays(root)
        paths = set(arrays)
        if expected_paths is None:
            expected_paths = paths
        elif paths != expected_paths:
            raise RuntimeError(f"{root}: fixture array paths differ from the first run")
        arrays_by_root[root] = arrays
        device = provenance["device_provenance"]
        runs.append(
            {
                "root": str(root),
                "fixture_manifest_sha256": sha256(manifest_path),
                "provenance_sha256": sha256(provenance_path),
                "torch_intraop_threads": device.get("torch_intraop_threads"),
                "torch_interop_threads": device.get("torch_interop_threads"),
            }
        )

    stages: dict[str, dict[str, Any]] = {}
    total_pairs = 0
    for left_root, right_root in itertools.combinations(roots, 2):
        pair = f"{left_root.name}__vs__{right_root.name}"
        for path in sorted(expected_paths or ()):
            stage = stages.setdefault(
                seam_for(path),
                {
                    "kind": "floating" if np.issubdtype(np.load(arrays_by_root[left_root][path], mmap_mode="r").dtype, np.floating) else "discrete",
                    "array_pairs": 0,
                    "elements_compared": 0,
                    "differing_elements": 0,
                    "max_abs": 0.0,
                    "first_divergence": None,
                },
            )
            record_difference(
                stage,
                np.load(arrays_by_root[left_root][path], mmap_mode="r"),
                np.load(arrays_by_root[right_root][path], mmap_mode="r"),
                pair,
                path,
            )
            total_pairs += 1

    prompt_ids = lambda name: name.startswith("prompt_build/") and name.endswith(("prompt.text_ids", "prompt.reference_ids"))
    logits = lambda name: "codec_head" in name or "teacher_forced_logits" in name or "/microdecoder.head_" in name
    codes = lambda name: name.endswith("talker.codec_codes")
    waveforms = lambda name: name.endswith("codec.generated_waveform")
    payload = {
        "schema_version": 1,
        "artifact_kind": "cpu_fp32_oracle_nondeterminism_floor",
        "claim_scope": {
            "applies_to": "the pinned CPU FP32 fallback oracle only",
            "does_not_establish": "native CUDA or cross-device tolerances; those require a separately captured native-device envelope",
        },
        "runs": runs,
        "comparison": {
            "comparison_policy": "all pairwise comparisons across the supplied independent captures, including cross-thread-count pairs",
            "array_pairs_compared": total_pairs,
            "stage_count": len(stages),
        },
        "contract_a": {
            "L0_prompt_token_ids": rung_summary(stages, prompt_ids),
            "L1_operator_seams": {"status": "not_observed", "reason": "the fixture generator records L2+ named module seams, not individual operators"},
            "L2_layer_and_component_activations": rung_summary(stages, lambda _name: True),
            "L3_logits": rung_summary(stages, logits),
            "L4_greedy_codec_tokens": rung_summary(stages, codes),
            "L5_codec_waveform": rung_summary(stages, waveforms),
        },
        "stage_envelopes": dict(sorted(stages.items())),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"REFUSING: {error}")
