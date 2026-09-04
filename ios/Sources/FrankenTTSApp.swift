import SwiftUI
import Foundation
import UIKit

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
                .background(CatalystWindowFreedom())
#if targetEnvironment(macCatalyst)
                .frame(minWidth: 480, minHeight: 420)
#endif
        }
#if targetEnvironment(macCatalyst)
        .defaultSize(width: 1180, height: 820)
        .windowResizability(.contentMinSize)
#endif
        .commands {
            VoiceForgeCommands()
            LabTextSizeCommands()
        }
    }
}

private struct CatalystWindowFreedom: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> Controller { Controller() }
    func updateUIViewController(_ controller: Controller, context: Context) { controller.configure() }

    final class Controller: UIViewController {
        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)
            configure()
        }

        override func viewDidLayoutSubviews() {
            super.viewDidLayoutSubviews()
            configure()
        }

        func configure() {
#if targetEnvironment(macCatalyst)
            guard let restrictions = view.window?.windowScene?.sizeRestrictions else { return }
            restrictions.minimumSize = CGSize(width: 480, height: 420)
            restrictions.maximumSize = CGSize(width: 10_000, height: 10_000)
#endif
        }
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

private struct LabTextSizeCommands: Commands {
    @AppStorage(Lab.textScaleStorageKey) private var textScale = Lab.defaultTextScale

    var body: some Commands {
        CommandMenu("Text Size") {
            Button("Make Text Larger") {
                textScale = Lab.steppedTextScale(textScale + Lab.textScaleStep)
            }
            .keyboardShortcut("+", modifiers: [.command])
            .disabled(textScale >= Lab.maximumTextScale)

            Button("Make Text Smaller") {
                textScale = Lab.steppedTextScale(textScale - Lab.textScaleStep)
            }
            .keyboardShortcut("-", modifiers: [.command])
            .disabled(textScale <= Lab.minimumTextScale)

            Divider()

            Button("Actual Size") {
                textScale = Lab.defaultTextScale
            }
            .keyboardShortcut("0", modifiers: [.command])
            .disabled(abs(textScale - Lab.defaultTextScale) < 0.001)
        }
    }
}
