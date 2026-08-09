#!/usr/bin/env bash
# Builds the wasm package into site/pkg for local serving or Pages deploy.
set -euo pipefail
cd "$(dirname "$0")/.."
# +simd128: SIMD kernels; --max-memory: rustc's wasm default caps linear memory at 1 GB,
# and this engine holds ~3.5 GB of model at peak — 4 GB is wasm32's ceiling.
RUSTFLAGS="-C target-feature=+simd128 -C link-arg=--max-memory=4294967296" \
  wasm-pack build crates/ftts-wasm --target web --release --out-dir ../../site/pkg
# wasm-pack writes a .gitignore that would hide the artifacts from Pages' upload.
rm -f site/pkg/.gitignore
echo "site/ ready — serve locally with: python3 -m http.server -d site 8788"
echo "(the /model proxy only exists on Pages; for local runs, point loader.js at a local copy)"
