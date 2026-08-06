#!/usr/bin/env python3
"""OQ-13: dump a community GGUF's per-tensor quantization types WITHOUT downloading the file.

The GGUF header (magic, metadata KVs, then one record per tensor: name, dims, ggml type,
offset) lives at the front of the file, so an HTTP Range request over the first few MB is
enough to enumerate every tensor's type. A Q8_0 talker GGUF is ~1 GB; this reads ~1-16 MB.

Why: OQ-13 asks what the upstream GGML conversion keeps at HIGH precision. That set is the
closest thing to prior-art validation our quantization recipe (plan §6.3) has. Any tensor THEY
keep high that WE planned to quantize is our risk to retire by measurement, never to assume safe.
Hand-transcribing their list would be a counterfeit gate -- this script reads the actual bytes.

Usage:
    python3 scripts/oq13_gguf_tensor_types.py <repo_id> <filename> [revision]
    python3 scripts/oq13_gguf_tensor_types.py --json ...     # machine-readable

Stdlib only. Exit 0 on success, 1 on parse/network failure.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request

# ggml_type enum -> name. Only the values that actually appear in TTS conversions are
# load-bearing; the rest are here so an unexpected type is named rather than numeric.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}
# Types we consider "kept high precision" for the reconciliation in OQ-13.
HIGH_PRECISION = {"F32", "F16", "BF16", "F64"}

_VAL = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    """Incremental reader over a byte buffer that can ask for more from the network."""

    def __init__(self, url: str, initial: int = 1 << 20, cap: int = 64 << 20):
        self.url, self.cap = url, cap
        self.buf = b""
        self.pos = 0
        self._fetch(initial)

    def _fetch(self, upto: int) -> None:
        if upto > self.cap:
            raise RuntimeError(f"header exceeded {self.cap} bytes; refusing to keep downloading")
        req = urllib.request.Request(self.url, headers={"Range": f"bytes=0-{upto - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.buf = r.read()

    def need(self, n: int) -> None:
        while self.pos + n > len(self.buf):
            self._fetch(max(len(self.buf) * 4, self.pos + n + (1 << 20)))

    def raw(self, n: int) -> bytes:
        self.need(n)
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def num(self, fmt: str) -> int | float | bool:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.raw(size))[0]

    def string(self) -> str:
        n = self.num("Q")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        """Read one metadata value; arrays are consumed but summarised (vocabs are huge)."""
        if vtype == 8:
            return self.string()
        if vtype == 9:
            elem = self.num("I")
            count = self.num("Q")
            if elem == 8:
                for _ in range(count):
                    self.string()
            elif elem == 9:
                raise RuntimeError("nested arrays unsupported")
            else:
                self.raw(struct.calcsize("<" + _VAL[elem]) * count)
            return f"<array {GGML_TYPES.get(elem, elem)}[{count}]>"
        return self.num(_VAL[vtype])


def read_header(url: str) -> tuple[dict, list[dict]]:
    r = Reader(url)
    if r.raw(4) != b"GGUF":
        raise RuntimeError("not a GGUF file (bad magic)")
    version = r.num("I")
    n_tensors = r.num("Q")
    n_kv = r.num("Q")

    meta: dict = {"gguf_version": version, "tensor_count": n_tensors}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.num("I")
        val = r.value(vtype)
        if not key.startswith("tokenizer.ggml."):  # skip giant vocab blobs
            meta[key] = val

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.num("I")
        dims = [r.num("Q") for _ in range(nd)]
        t = r.num("I")
        r.num("Q")  # offset
        tensors.append({"name": name, "dims": dims, "type": GGML_TYPES.get(t, f"UNKNOWN({t})")})
    return meta, tensors


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 1
    repo, fname = args[0], args[1]
    rev = args[2] if len(args) > 2 else "main"
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"

    try:
        meta, tensors = read_header(url)
    except Exception as e:  # network or parse
        print(f"ERROR reading {url}: {e}", file=sys.stderr)
        return 1

    by_type: dict[str, int] = {}
    for t in tensors:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
    high = [t for t in tensors if t["type"] in HIGH_PRECISION]

    if as_json:
        json.dump(
            {"repo": repo, "file": fname, "revision": rev, "metadata": meta,
             "type_histogram": by_type, "tensors": tensors},
            sys.stdout, indent=1,
        )
        print()
        return 0

    print(f"# {repo}/{fname} @ {rev}")
    for k in ("gguf_version", "tensor_count", "general.architecture", "general.file_type",
              "general.name", "general.quantization_version"):
        if k in meta:
            print(f"  {k}: {meta[k]}")
    print(f"\n  type histogram: {dict(sorted(by_type.items(), key=lambda kv: -kv[1]))}")
    print(f"  kept high precision (F32/F16/BF16): {len(high)} / {len(tensors)} tensors\n")
    print("  KEPT-HIGH TENSORS:")
    for t in high:
        print(f"    {t['type']:<5} {'x'.join(map(str, t['dims'])):<18} {t['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
