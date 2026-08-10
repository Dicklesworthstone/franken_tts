#!/usr/bin/env bash
# Builds the wasm package into site/pkg for local serving or Pages deploy.
#
# One build serves every browser, and it is a THREADED build: the memory is shared and imported,
# so the module cannot be instantiated without `SharedArrayBuffer`. That is safe here because
# site/_headers sets COOP/COEP site-wide, which is what makes `crossOriginIsolated` — and with it
# SharedArrayBuffer — true for every visitor, iOS Safari included (supported since 15.2).
#
# The team is still ARMED at runtime rather than assumed: `startTeam` only sizes it after the
# kernel Workers confirm they parked, so a browser that refuses to start Workers runs serially
# from the same bytes instead of waiting on partitions that never report.
#
# LOCAL SERVING CAVEAT: `python3 -m http.server` sends no COOP/COEP, so `crossOriginIsolated` is
# false and instantiation fails outright. Use a server that sets both headers when testing
# locally; the deployed site is unaffected.
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
# THE TLS EXPORTS ARE THE WHOLE TRICK, and they cost a day to find. wasm-bindgen's threading
# transform fails with "failed to find `__wasm_init_tls`", which reads like a missing TLS segment
# and sends you off writing `#[thread_local]` anchors. It is not. `wasm-objdump` showed `.tdata`
# present at 8,008 bytes the entire time — LLD had built the segment and simply had not EXPORTED
# the symbols that describe it, and wasm-bindgen looks them up by name among the exports. Exporting
# the set below fixes it outright; each one was discovered by the transform naming the next missing
# symbol in turn (`__wasm_init_tls`, then `__tls_size`).
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--shared-memory \
  -C link-arg=--import-memory \
  -C link-arg=--max-memory=4294967296 \
  -C link-arg=--export=__heap_base \
  -C link-arg=--export=__tls_base \
  -C link-arg=--export=__tls_size \
  -C link-arg=--export=__tls_align \
  -C link-arg=--export=__wasm_init_tls" \
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
