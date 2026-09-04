#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/ios"

build_root="${FRANKEN_APPLE_BUILD_ROOT:-${DSR_QUALITY_RUN_DIR:-$repo_root/ios/build/dsr-apple-quality}}"
mkdir -p "$build_root"
sbh check --need 20G "$build_root"
command -v xcodegen >/dev/null
xcodegen generate --spec project.yml
git diff --exit-code -- FrankenTTS.xcodeproj Sources/Info.plist
git ls-files -z -- '*.swift' | xargs -0 xcrun swiftc -parse
plutil -lint Sources/Info.plist
plutil -lint Sources/PrivacyInfo.xcprivacy
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenTTS.xcodeproj -scheme FrankenTTS \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO build
xcodebuild -project FrankenTTS.xcodeproj -scheme FrankenTTS \
  -destination 'platform=macOS,variant=Mac Catalyst' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO test -only-testing:FrankenTTSTests

# Keep routine product proof deterministic and inaudible. The dedicated
# FrankenTTS simulator prevents this lane from mutating another app's device,
# while exact selectors keep the opt-in playback/lifecycle tests out.
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
ui_simulator_udid="${FRANKENTTS_UI_SIMULATOR_UDID:-$(
  xcrun simctl list devices available --json | jq -r '
    [
      .devices[][]
      | select(.isAvailable == true)
      | select(.name | test("^FrankenTTS .*iPhone"))
    ][0].udid // empty
  '
)}"
if [[ -z "$ui_simulator_udid" ]]; then
  echo "No available dedicated FrankenTTS iPhone simulator; set FRANKENTTS_UI_SIMULATOR_UDID." >&2
  exit 1
fi

/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenTTS.xcodeproj -scheme FrankenTTS \
  -destination "platform=iOS Simulator,id=$ui_simulator_udid" \
  -derivedDataPath "$build_root/derived-data" \
  -resultBundlePath "$build_root/frankentts-iphone-ui.xcresult" \
  -parallel-testing-enabled NO \
  -maximum-parallel-testing-workers 1 \
  CODE_SIGNING_ALLOWED=NO test \
  -only-testing:FrankenTTSUITests/FrankenTTSAppearanceUITests/testAppearanceTogglePersistsLightModeAcrossLaunches \
  -only-testing:FrankenTTSUITests/EnrollmentUITests/testNewVoiceCanStartWithoutTypingANameFirst \
  -only-testing:FrankenTTSUITests/UtteranceEditorUITests/testRecentVoicesIsDiscoverableFromTheMainHeader \
  -only-testing:FrankenTTSUITests/UtteranceEditorUITests/testSelectAllReplacesTextAndClearKeepsEditorUsable \
  -only-testing:FrankenTTSUITests/UtteranceEditorUITests/testMultilineEmojiEditAndOutsideTapDismissesKeyboard \
  -only-testing:FrankenTTSUITests/VoiceCardUITests/testVoiceCardRendersAndSharesFromTheOwningLibraryCover \
  -only-testing:FrankenTTSUITests/VoiceLabUITests/testComparisonWorkspaceIsDiscoverableWithoutPlayingAudio \
  -only-testing:FrankenTTSUITests/VoiceBrowserUITests/testLongPersonalVoiceNameStaysOnOneLineOnCompactPhone
