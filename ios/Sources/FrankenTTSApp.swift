import SwiftUI
import Foundation

@main
struct FrankenTTSApp: App {
    init() {
        // Five team partitions, not the Mac's six: 2P+4E leaves one E core for UI and
        // audio (docs/IOS_APP_PLAN.md §1). A launch override is preserved so physical-device
        // A/B runs measure distinct team sizes instead of silently testing five versus five.
        if ProcessInfo.processInfo.environment["FTTS_INT8_THREADS"] == nil {
            #if targetEnvironment(macCatalyst)
            let cores = ProcessInfo.processInfo.activeProcessorCount
            setenv("FTTS_INT8_THREADS", String(max(2, min(10, cores - 2))), 1)
            #else
            setenv("FTTS_INT8_THREADS", "5", 1)
            #endif
        }
    }

    var body: some Scene {
        WindowGroup {
            LabView()
                .preferredColorScheme(.dark)
        }
        .commands { VoiceForgeCommands() }
    }
}

struct VoiceForgeCommandActions {
    let importFile: () -> Void
    let synthesize: () -> Void
    let stop: () -> Void
    let togglePlayback: () -> Void
    let canSynthesize: Bool
    let canStop: Bool
    let canTogglePlayback: Bool
}

private struct VoiceForgeCommandKey: FocusedValueKey {
    typealias Value = VoiceForgeCommandActions
}

extension FocusedValues {
    var voiceForgeCommands: VoiceForgeCommandActions? {
        get { self[VoiceForgeCommandKey.self] }
        set { self[VoiceForgeCommandKey.self] = newValue }
    }
}

private struct VoiceForgeCommands: Commands {
    @FocusedValue(\.voiceForgeCommands) private var actions

    var body: some Commands {
        CommandMenu("Voice Forge") {
            Button("Open Text or Voice Card…") { actions?.importFile() }
                .keyboardShortcut("o", modifiers: [.command])

            Divider()

            Button("Synthesize") { actions?.synthesize() }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(actions?.canSynthesize != true)

            Button("Play or Pause") { actions?.togglePlayback() }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(actions?.canTogglePlayback != true)

            Divider()

            Button("Stop Synthesis") { actions?.stop() }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(actions?.canStop != true)
        }
    }
}
