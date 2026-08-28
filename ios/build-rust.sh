#!/usr/bin/env bash
# Builds the Rust engine for iOS device + simulator + Mac Catalyst and assembles
# FttsCore.xcframework.
# Run before the first Xcode build and after any Rust change:  ios/build-rust.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
# Apple cross targets require the local Xcode SDK and linker. Resolve Cargo
# through rustup so an RCH shim cannot reject the build for lack of a Darwin worker.
APPLE_RUST_TOOLCHAIN="${APPLE_RUST_TOOLCHAIN:-nightly-2026-08-25-aarch64-apple-darwin}"
APPLE_CARGO="${APPLE_CARGO:-$(rustup which --toolchain "$APPLE_RUST_TOOLCHAIN" cargo)}"

for target in \
  aarch64-apple-ios \
  aarch64-apple-ios-sim \
  aarch64-apple-ios-macabi \
  x86_64-apple-ios-macabi
do
  # Exact matching matters: `aarch64-apple-ios` is a prefix of both simulator
  # and Catalyst triples, and a substring match can falsely skip installation.
  rustup target list --toolchain "$APPLE_RUST_TOOLCHAIN" --installed | grep -qx "$target" || \
    rustup target add --toolchain "$APPLE_RUST_TOOLCHAIN" "$target"
  RUSTUP_TOOLCHAIN="$APPLE_RUST_TOOLCHAIN" RCH_CARGO_WRAPPER_BYPASS=1 "$APPLE_CARGO" build \
    --release --locked -p ftts-ffi --target "$target"
done

HEADERS=$(mktemp -d /tmp/ftts-ffi-headers.XXXXXX)
cp crates/ftts-ffi/include/ftts_ffi.h crates/ftts-ffi/include/module.modulemap "$HEADERS/"

CATALYST_LIB=$(mktemp /tmp/libftts_ffi-maccatalyst.XXXXXX)
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-macabi/release/libftts_ffi.a" \
  "$TARGET_DIR/x86_64-apple-ios-macabi/release/libftts_ffi.a" \
  -output "$CATALYST_LIB"

FRAMEWORK=ios/FttsCore.xcframework
OUTPUT_ROOT=$(mktemp -d /tmp/ftts-xcframework.XXXXXX)
STAGED_FRAMEWORK="$OUTPUT_ROOT/FttsCore.xcframework"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/release/libftts_ffi.a" -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/release/libftts_ffi.a" -headers "$HEADERS" \
  -library "$CATALYST_LIB" -headers "$HEADERS" \
  -output "$STAGED_FRAMEWORK"

# Preserve the previous generated framework as a recoverable build artifact.
if [[ -e "$FRAMEWORK" ]]; then
  mv "$FRAMEWORK" "$FRAMEWORK.previous-$(date +%Y%m%d-%H%M%S)"
fi
mv "$STAGED_FRAMEWORK" "$FRAMEWORK"

echo "built $FRAMEWORK"
echo "next: (cd ios && xcodegen generate) if project.yml changed, then build in Xcode"
