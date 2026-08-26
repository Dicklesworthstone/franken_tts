#!/usr/bin/env python3
"""Cross-target tap comparator for frankentts-p16p verification.

Compares prefill sub-stage lines (ftts-tapP g/p/h), text-projection
sub-stage lines (ftts-tapX fc1/silu/fc2), and per-layer eight-stage lines
(ftts-tapL l=) between a native tap log and a wasm transcript.

The wasm transcript mirrors every console line ([error] ... plus an
[engine-worker] copy), so repeated identical hashes are deduped
order-preserving before comparison.

Usage: compare_taps.py <native_log> <wasm_transcript>
"""
import re
import sys

L = re.compile(
    r"ftts-tapL l=(\d+) a=([0-9a-f]{16}) b=([0-9a-f]{16}) q=([0-9a-f]{16}) "
    r"k=([0-9a-f]{16}) d=([0-9a-f]{16}) e=([0-9a-f]{16}) f=([0-9a-f]{16}) "
    r"g=([0-9a-f]{16}) h=([0-9a-f]{16})"
)
STAGES = ["anorm", "qkv", "qrope", "krope", "softmax", "ares", "mnorm", "gateup", "out"]
GATHER = re.compile(r"ftts-tapP rows=(\d+) g=([0-9a-f]{16})")
PROJECTED = re.compile(r"ftts-tapP p=([0-9a-f]{16})")
HIDDEN = re.compile(r"ftts-tapP h=([0-9a-f]{16})")
FC1 = re.compile(r"ftts-tapX fc1=([0-9a-f]{16})")
SILU = re.compile(r"ftts-tapX silu=([0-9a-f]{16})")
FC2 = re.compile(r"ftts-tapX fc2=([0-9a-f]{16})")

SCALAR_STAGES = ("projected", "hidden", "fc1", "silu", "fc2")


def collect(path):
    layers = {}
    stage_hashes = {
        "gather": None,
        "projected": [],
        "hidden": [],
        "fc1": [],
        "silu": [],
        "fc2": [],
    }
    patterns = (
        (GATHER, "gather"),
        (PROJECTED, "projected"),
        (HIDDEN, "hidden"),
        (FC1, "fc1"),
        (SILU, "silu"),
        (FC2, "fc2"),
    )
    with open(path, errors="replace") as source:
        for line in source:
            layer_match = L.search(line)
            if layer_match and len(layer_match.group(10)) == 16:
                key = int(layer_match.group(1))
                if key not in layers:
                    layers[key] = layer_match.groups()[1:]
                continue
            for pattern, name in patterns:
                mm = pattern.search(line)
                if not mm:
                    continue
                if name == "gather":
                    stage_hashes[name] = (int(mm.group(1)), mm.group(2))
                else:
                    stage_hashes[name].append(mm.group(1))

    def dedupe(values):
        seen = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return seen

    for name in SCALAR_STAGES:
        stage_hashes[name] = dedupe(stage_hashes[name])
    return layers, stage_hashes


def main(native_path, wasm_path):
    native_layers, native_stages = collect(native_path)
    wasm_layers, wasm_stages = collect(wasm_path)
    for name in ("gather", "projected", "hidden", "fc1", "silu", "fc2"):
        n_value = native_stages[name]
        w_value = wasm_stages[name]
        equal = n_value == w_value
        print(f"tap[{name}] native={n_value} wasm={w_value} equal={equal}")
    common = sorted(set(native_layers) & set(wasm_layers))
    print(f"layers native {len(native_layers)} wasm {len(wasm_layers)} common {len(common)}")
    first = None
    matches = 0
    for k in common:
        diffs = [
            STAGES[i]
            for i, (x, y) in enumerate(zip(native_layers[k], wasm_layers[k]))
            if x != y
        ]
        if not diffs:
            matches += 1
        elif first is None:
            first = (k, diffs)
    print(f"fully-matching layers: {matches}/{len(common)}")
    if len(common) == len(native_layers) == len(wasm_layers) and matches == len(common):
        print("VERDICT: ALL LAYERS x STAGES MATCH")
    elif first:
        print(f"FIRST LAYER DIVERGENCE layer {first[0]} stages: {','.join(first[1])}")
    else:
        print("VERDICT: INCOMPLETE PAIR")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
