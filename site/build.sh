#!/usr/bin/env bash
# Builds the wasm package into site/pkg for local serving or Pages deploy.
#
# One build serves every browser. Threads are compiled IN unconditionally and armed at runtime
# only where `SharedArrayBuffer` and a growable shared memory actually exist, so Chrome and
# Firefox get the worker team while an iPhone falls back to the serial path from the same bytes.
# Shipping two builds and choosing between them in JS would double the download of a 2 MB module
# for no benefit the runtime check does not already give.
set -euo pipefail
cd "$(dirname "$0")/.."

# +simd128       : the hand-written int8 island (Int8Tier::WasmSimd128).
# +atomics,...   : threads. `atomics` alone is not enough — the shared-memory linker flags below
#                  are what make the resulting memory usable from more than one Worker, and
#                  bulk-memory/mutable-globals are the companion features the threads proposal
#                  assumes.
# --max-memory   : `+atomics` already makes LLD emit a *shared* imported memory, so passing
#                  --shared-memory/--import-memory by hand is redundant — and harmful: doing so
#                  dropped `__heap_base` from the exports, and wasm-bindgen needs it to inject
#                  per-thread ids ("failed to find __heap_base for injecting thread id"). A shared
#                  memory must still declare its ceiling, and 4 GB is wasm32's hard limit.
# --export=...   : belt and braces for the two symbols the threading transform looks up.
# -Z build-std   : std itself must be rebuilt with the atomics feature; the prebuilt std is not
#                  thread-enabled, which is why this needs nightly and a rust-src component.
RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--max-memory=4294967296 \
  -C link-arg=--export=__heap_base \
  -C link-arg=--export=__tls_base" \
  wasm-pack build crates/ftts-wasm --target web --release --out-dir ../../site/pkg \
  -- -Z build-std=std,panic_abort

# wasm-pack writes a .gitignore that would hide the artifacts from Pages' upload.
rm -f site/pkg/.gitignore

# wasm-pack's own wasm-opt pass is conservative. -O4 with SIMD enabled is worth another pass over
# a module whose hot loop is now hand-written v128: it inlines across the kernel boundary and
# strips the load/store churn that shows up between accumulator streams. Optional by design —
# a missing wasm-opt should cost speed, not the build. --enable-threads must be passed or wasm-opt
# rejects the atomics the module now contains.
WASM="site/pkg/ftts_wasm_bg.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  before=$(wc -c < "$WASM")
  wasm-opt -O4 --enable-simd --enable-threads --enable-bulk-memory --enable-mutable-globals \
    "$WASM" -o "$WASM.opt" && mv "$WASM.opt" "$WASM"
  after=$(wc -c < "$WASM")
  echo "wasm-opt -O4: $before -> $after bytes"
else
  echo "WARNING: wasm-opt not found (brew install binaryen) — shipping wasm-pack's output as-is"
fi
echo "site/ ready — serve locally with: python3 -m http.server -d site 8788"
echo "(the /model proxy only exists on Pages; for local runs, point loader.js at a local copy)"
