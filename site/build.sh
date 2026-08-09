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

# Built with `cargo` + the `wasm-bindgen` CLI directly rather than through wasm-pack.
#
# wasm-pack injects its own RUSTFLAGS, which clobbers the target-feature list — and if `+atomics`
# does not reach `std`, std is rebuilt without TLS, LLD emits no `__wasm_init_tls`, and
# wasm-bindgen's threading transform fails looking for it. Driving the two steps ourselves is the
# only way to be certain the flags below apply to every unit, std included.
#
# +simd128       : the hand-written int8 island (Int8Tier::WasmSimd128).
# +atomics,...   : threads. std must be rebuilt with these, which is what -Z build-std is for and
#                  why this needs nightly plus the rust-src component.
# --export=...   : symbols wasm-bindgen's threading transform looks up by name.
#
# THREADS ARE COMPILED IN BUT NOT YET ARMED, and the runtime notices rather than hanging.
# Adding `--shared-memory --import-memory` DOES produce the right memory — objdump confirms
# `memory[0] ... shared <- env.memory` — but wasm-bindgen then fails with "failed to find
# `__wasm_init_tls`": LLD emitted no TLS segment for it to initialize. That is the one remaining
# step, and it is a toolchain question (force a `#[thread_local]` symbol so LLD emits the segment,
# or adopt wasm-bindgen-rayon, which carries this plumbing), not an engine question — the Rust
# team, the two-phase arm protocol, and the Worker pool are all written and compile.
#
# `startTeam` refuses to arm unless the INSTANTIATED memory is really a SharedArrayBuffer, so this
# ships as a correct serial engine rather than a dispatcher waiting on partitions that never
# existed.
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--max-memory=4294967296 \
  -C link-arg=--export=__heap_base \
  -C link-arg=--export=__tls_base" \
  cargo build -p ftts-wasm --target wasm32-unknown-unknown --release \
    -Z build-std=std,panic_abort

wasm-bindgen "$TARGET_DIR/wasm32-unknown-unknown/release/ftts_wasm.wasm" \
  --out-dir site/pkg --typescript --target web

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
