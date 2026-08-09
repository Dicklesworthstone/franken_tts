#!/usr/bin/env bash
# Builds the wasm package into site/pkg for local serving or Pages deploy.
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/ftts-wasm --target web --release --out-dir ../../site/pkg
# wasm-pack writes a .gitignore that would hide the artifacts from Pages' upload.
rm -f site/pkg/.gitignore
echo "site/ ready — serve locally with: python3 -m http.server -d site 8788"
echo "(the /model proxy only exists on Pages; for local runs, point loader.js at a local copy)"
