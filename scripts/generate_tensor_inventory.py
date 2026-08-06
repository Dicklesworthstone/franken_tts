#!/usr/bin/env python3
"""Generate the OQ-2 tensor inventory from pinned safetensors headers.

This intentionally reads only the safetensors JSON headers: no model tensor is
loaded into memory.  The result is the exact manifest consumed by later census
and loader work; checksum validation remains the truth-pack fetcher's job.
"""

import json
import struct
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "docs/truth-pack/snapshots/hf"
OUTPUT = ROOT / "docs/truth-pack/TENSOR_INVENTORY.json"
SOURCES = [
    ("model.safetensors", SNAPSHOTS / "model.safetensors"),
    ("speech_tokenizer/model.safetensors", SNAPSHOTS / "speech_tokenizer/model.safetensors"),
]
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def header(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        return json.loads(file.read(length))


def tensor_record(source: str, name: str, meta: dict[str, object]) -> dict[str, object]:
    shape = meta["shape"]
    elements = 1
    for dimension in shape:
        elements *= dimension
    dtype = meta["dtype"]
    return {
        "source": source,
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "elements": elements,
        "bytes": elements * DTYPE_BYTES[dtype],
    }


def main() -> None:
    tensors = []
    for source, path in SOURCES:
        if not path.is_file():
            raise SystemExit(f"missing pinned weight file: {path}")
        tensors.extend(
            tensor_record(source, name, meta)
            for name, meta in header(path).items()
            if name != "__metadata__"
        )
    tensors.sort(key=lambda tensor: (tensor["source"], tensor["name"]))
    counts = Counter(tensor["source"] for tensor in tensors)
    by_dtype = Counter(tensor["dtype"] for tensor in tensors)
    document = {
        "schema_version": 1,
        "source_pin": "Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc",
        "method": "safetensors header only; tensor payloads are not loaded",
        "summary": {
            "tensor_count": len(tensors),
            "by_source": dict(sorted(counts.items())),
            "by_dtype": dict(sorted(by_dtype.items())),
            "payload_bytes": sum(tensor["bytes"] for tensor in tensors),
        },
        "tensors": tensors,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
