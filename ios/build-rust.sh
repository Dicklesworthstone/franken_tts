#!/usr/bin/env bash
# Builds the Rust engine for iOS device + simulator and assembles FttsCore.xcframework.
# Run before the first Xcode build and after any Rust change:  ios/build-rust.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"

for target in aarch64-apple-ios aarch64-apple-ios-sim; do
  rustup target list --installed | grep -q "$target" || rustup target add "$target"
  cargo build --release --locked -p ftts-ffi --target "$target"
done

HEADERS=$(mktemp -d /tmp/ftts-ffi-headers.XXXXXX)
cp crates/ftts-ffi/include/ftts_ffi.h "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'EOF'
module FttsCore {
    header "ftts_ffi.h"
    export *
}
EOF

FRAMEWORK=ios/FttsCore.xcframework
rm -rf "$FRAMEWORK"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/release/libftts_ffi.a" -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/release/libftts_ffi.a" -headers "$HEADERS" \
  -output "$FRAMEWORK"

echo "built $FRAMEWORK"
echo "next: (cd ios && xcodegen generate) if project.yml changed, then build in Xcode"
