# Oracle fixtures r1 — fetch & verification

Native-CUDA ConformanceExact fixtures for `synthetic-tone-en` (bead `frankentts-rf4`).
Captured 2026-08-23 on an RTX 4090 at the exact pinned source/weights/runtime recorded in
`provenance.json` in this directory.

## Fetch

<https://github.com/Dicklesworthstone/franken_tts/releases/download/oracle-fixtures-r1/qwen3-tts-oracle-fixtures-r1.tar.gz>

```bash
curl -LO <url>
shasum -a 256 qwen3-tts-oracle-fixtures-r1.tar.gz
# expect 983d57a2b1dc8b1a77a5ec8334b2c8c0c4d86ebcfe6a7a1ea5259f0db120fe42
tar xzf qwen3-tts-oracle-fixtures-r1.tar.gz
```

## Anchors

| Artifact | SHA-256 |
|---|---|
| `fixture_manifest.json` (also committed here) | `57f6c273b397dbc13d27d74a636bef0263d48b90b58a7953ffba2e086273d7d1` |
| `provenance.json` (also committed here) | `a146a81081be5145d626c4e5982bbf6d6fa29058a2dab621d210bb205be803f3` |
| tarball | `983d57a2b1dc8b1a77a5ec8334b2c8c0c4d86ebcfe6a7a1ea5259f0db120fe42` |

Per-mode manifests (`xvector/icl × non_streaming/streaming`) are anchored inside
`fixture_manifest.json`; spot-checked byte-exact against the mode directories at capture time.

## Regeneration

`scripts/gen_reference_fixtures.py --help` documents the full contract. The r1 invocation:

```
python scripts/gen_reference_fixtures.py \
  --source-dir <Qwen3-TTS @ 022e286b> \
  --model-dir  <HF rev 5d83992436 materialized> \
  --corpus     docs/conformance/oracle_corpus.json \
  --output     <new dir>
```

The script refuses any runtime/source/weights drift, so a faithful regeneration reproduces the
anchors above on the same device class.
