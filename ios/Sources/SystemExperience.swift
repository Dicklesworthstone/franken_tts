import AppIntents
import Foundation
import WidgetKit

#if canImport(ActivityKit) && !targetEnvironment(macCatalyst)
import ActivityKit

@MainActor
final class VoiceForgeActivityController {
    static let shared = VoiceForgeActivityController()

    private var activity: Activity<FrankenTTSRunActivityAttributes>?
    private var lastPublishedStage = ""
    private var lastPublishedUnits: UInt64 = 0

    private init() {}

    func begin() {
        finish(status: .cancelled, headline: "Previous run ended", detail: "Ready for a new voice")
        guard ActivityAuthorizationInfo().areActivitiesEnabled else { return }
        let attributes = FrankenTTSRunActivityAttributes(runID: UUID(), startedAt: .now)
        let state = FrankenTTSRunActivityAttributes.ContentState(
            stage: "Charging the Voice Forge",
            detail: "Preparing the private on-device engine",
            completedUnits: 0,
            totalUnits: 0,
            elapsedSeconds: 0,
            status: .preparing
        )
        activity = try? Activity.request(
            attributes: attributes,
            content: ActivityContent(state: state, staleDate: nil),
            pushType: nil
        )
        lastPublishedStage = state.stage
        lastPublishedUnits = 0
        publishWidget(.working, headline: state.stage, detail: state.detail)
    }

    func update(from telemetry: VoiceForgeTelemetry, elapsed: TimeInterval) {
        guard let activity else { return }
        let completed = max(telemetry.generatedFrames, telemetry.decodedFrames)
        // Native callbacks can arrive quickly. A stage change is always visible;
        // within a stage, update at useful unit boundaries rather than per callback.
        guard telemetry.phase.title != lastPublishedStage || completed >= lastPublishedUnits + 4 else {
            return
        }
        lastPublishedStage = telemetry.phase.title
        lastPublishedUnits = completed
        let status: FrankenTTSRunActivityAttributes.ContentState.Status =
            telemetry.phase == .cancelling ? .cancelling : .running
        let state = FrankenTTSRunActivityAttributes.ContentState(
            stage: telemetry.phase.title,
            detail: telemetry.factualDetail,
            completedUnits: completed,
            totalUnits: telemetry.predictedMaximumFrames,
            elapsedSeconds: max(0, Int(elapsed.rounded(.down))),
            status: status
        )
        Task { await activity.update(ActivityContent(state: state, staleDate: nil)) }
        publishWidget(.working, headline: state.stage, detail: state.detail)
    }

    func finish(
        status: FrankenTTSRunActivityAttributes.ContentState.Status,
        headline: String,
        detail: String
    ) {
        guard let current = activity else { return }
        activity = nil
        let state = FrankenTTSRunActivityAttributes.ContentState(
            stage: headline,
            detail: detail,
            completedUnits: lastPublishedUnits,
            totalUnits: lastPublishedUnits,
            elapsedSeconds: max(0, Int(Date().timeIntervalSince(current.attributes.startedAt))),
            status: status
        )
        let dismissal: ActivityUIDismissalPolicy = status == .complete ? .after(.now + 45) : .immediate
        Task { await current.end(ActivityContent(state: state, staleDate: nil), dismissalPolicy: dismissal) }
        let readiness: FrankenTTSWidgetSnapshot.Readiness = status == .complete ? .complete : .ready
        publishWidget(readiness, headline: headline, detail: detail)
    }

    private func publishWidget(
        _ readiness: FrankenTTSWidgetSnapshot.Readiness,
        headline: String,
        detail: String
    ) {
        FrankenTTSSharedStore.save(
            FrankenTTSWidgetSnapshot(
                readiness: readiness,
                headline: headline,
                detail: detail,
                updatedAt: .now
            )
        )
        WidgetCenter.shared.reloadTimelines(ofKind: "FrankenTTSForgeWidget")
    }
}
#else
@MainActor
final class VoiceForgeActivityController {
    static let shared = VoiceForgeActivityController()
    private init() {}
    func begin() {}
    func update(from telemetry: VoiceForgeTelemetry, elapsed: TimeInterval) {}
    func finish(
        status: FrankenTTSRunActivityAttributes.ContentState.Status,
        headline: String,
        detail: String
    ) {}
}
#endif

struct SpeakTextIntent: AppIntent {
    static let title: LocalizedStringResource = "Speak Text"
    static let description = IntentDescription(
        "Open FrankenTTS with text ready in the private on-device Voice Forge."
    )
    static let openAppWhenRun = true

    @Parameter(title: "Text")
    var text: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !value.isEmpty { FrankenTTSSharedStore.stage(text: value) }
        return .result(dialog: "Ready in the Voice Forge.")
    }
}

struct OpenVoiceForgeIntent: AppIntent {
    static let title: LocalizedStringResource = "Open Voice Forge"
    static let description = IntentDescription("Open FrankenTTS, ready to create private speech.")
    static let openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult {
        .result()
    }
}

struct FrankenTTSShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: SpeakTextIntent(),
            phrases: [
                "Speak with \(.applicationName)",
                "Make a voice with \(.applicationName)"
            ],
            shortTitle: "Speak Text",
            systemImageName: "waveform.badge.plus"
        )
        AppShortcut(
            intent: OpenVoiceForgeIntent(),
            phrases: ["Open the Voice Forge in \(.applicationName)"],
            shortTitle: "Voice Forge",
            systemImageName: "bolt.fill"
        )
    }
}
