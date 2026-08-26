#!/usr/bin/env python3
"""Cross-target tap comparator for frankentts-p16p verification.

Compares prefill sub-stage lines (ftts-tapP rows/g/p/h) and per-layer
eight-stage lines (ftts-tapL l=) between a native tap log and a wasm
transcript. Prints the first divergence point and an overall verdict so the
DISC-006 seam-B chase has one mechanical receipt command.
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
G = re.compile(r"ftts-tapP rows=(\d+) g=([0-9a-f]{16})")
P = re.compile(r"ftts-tapP p=([0-9a-f]{16})")
H = re.compile(r"ftts-tapP h=([0-9a-f]{16})")


def collect(path):
    layers = {}
    gather = None
    projected = []
    hidden = []
    with open(path, errors="replace") as source:
        for line in source:
            m = L.search(line)
            if m and len(m.group(10)) == 16:
                key = int(m.group(1))
                if key not in layers:
                    layers[key] = m.groups()[1:]
                continue
            m = G.search(line)
            if m:
                gather = (int(m.group(1)), m.group(2))
            for pat, bucket in ((P, projected), (H, hidden)):
                mm = pat.search(line)
                if mm:
                    bucket.append(mm.group(1))
    return layers, (gather, projected, hidden)


def main(native_path, wasm_path):
    nl, ngather, nproj, nhid = _triplet(collect(native_path))
    wl, wgather, wproj, whid = _triplet(collect(wasm_path))
    print(f"tapP gather: native={ngather} wasm={wgather} equal={ngather == wgather}")
    print(f"tapP projected: native={nproj} wasm={wproj} equal={nproj == wproj}")
    print(f"tapP hidden: native={nhid} wasm={whid} equal={nhid == whid}")
    common = sorted(set(nl) & set(wl))
    print(f"layers native {len(nl)} wasm {len(wl)} common {len(common)}")
    first = None
    matches = 0
    for k in common:
        diffs = [STAGES[i] for i, (x, y) in enumerate(zip(nl[k], wl[k])) if x != y]
        if not diffs:
            matches += 1
        elif first is None:
            first = (k, diffs)
    print(f"fully-matching layers: {matches}/{len(common)}")
    if len(common) == len(nl) == len(wl) and matches == len(common):
        print("VERDICT: ALL LAYERS x STAGES MATCH")
    elif first:
        print(f"FIRST DIVERGENCE layer {first[0]} stages: {','.join(first[1])}")
    else:
        print("VERDICT: INCOMPLETE PAIR")


def _triplet(parsed):
    return parsed[0], parsed[1][0], parsed[1][1], parsed[1][2]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
