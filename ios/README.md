# FrankenTTS for iOS

The website playground as a native SwiftUI app: download the model once, pick or clone a
voice, synthesize on device, listen, share. Design and constraints are in
[`docs/IOS_APP_PLAN.md`](../docs/IOS_APP_PLAN.md).

## Building

```bash
# 1. Build the Rust engine for device + simulator and assemble FttsCore.xcframework
ios/build-rust.sh

# 2. Generate the Xcode project (only needed after project.yml changes)
cd ios && xcodegen generate

# 3. Build/run
xcodebuild -project FrankenTTS.xcodeproj -scheme FrankenTTS \
  -destination "generic/platform=iOS Simulator" CODE_SIGNING_ALLOWED=NO build
# or open FrankenTTS.xcodeproj in Xcode and run on a device/simulator.
```

Requirements: Xcode with the iOS platform installed, `xcodegen` (brew), and the
`aarch64-apple-ios` / `aarch64-apple-ios-sim` Rust targets (the script adds them).
Simulator builds are arm64-only (Apple Silicon hosts); an Intel host would need the
`x86_64-apple-ios` Rust target added to `build-rust.sh`.

## Notes

- `FttsCore.xcframework` and `FrankenTTS.xcodeproj` are generated; only `project.yml`,
  `build-rust.sh`, and `Sources/` are source.
- Running with the real model needs ~2 GB free on the device and, in practice, an
  8 GB-RAM iPhone; the app warns below 6 GB. The entitlements file requests the
  increased-memory limit, which requires that capability on your signing profile for
  device runs (delete the entitlement to run without it; synthesis then risks jetsam).
- Real-time factor on A18-class hardware is unmeasured; the app shows the measured
  figure after each run and claims nothing else.
