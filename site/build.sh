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

# wasm-pack's own wasm-opt pass is conservative. -O4 with SIMD enabled is worth another pass over
# a module whose hot loop is now hand-written v128: it inlines across the kernel boundary and
# strips the load/store churn that shows up between accumulator streams. Optional by design —
# a missing wasm-opt should cost speed, not the build.
WASM="site/pkg/ftts_wasm_bg.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  before=$(wc -c < "$WASM")
  wasm-opt -O4 --enable-simd --enable-bulk-memory --enable-mutable-globals \
    "$WASM" -o "$WASM.opt" && mv "$WASM.opt" "$WASM"
  after=$(wc -c < "$WASM")
  echo "wasm-opt -O4: $before -> $after bytes"
else
  echo "WARNING: wasm-opt not found (brew install binaryen) — shipping wasm-pack's output as-is"
fi
echo "site/ ready — serve locally with: python3 -m http.server -d site 8788"
echo "(the /model proxy only exists on Pages; for local runs, point loader.js at a local copy)"
