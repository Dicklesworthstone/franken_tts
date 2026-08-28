import Foundation

#if !targetEnvironment(macCatalyst)
import ActivityKit
#endif

/// Privacy-safe state shared with WidgetKit and Live Activities. Source text,
/// cloned-voice names, and generated audio never enter the shared container.
struct FrankenTTSRunContentState: Codable, Hashable {
    enum Status: String, Codable, Hashable {
        case preparing
        case running
        case cancelling
        case complete
        case cancelled
        case failed
    }

    var stage: String
    var detail: String
    var completedUnits: UInt64
    var totalUnits: UInt64
    var elapsedSeconds: Int
    var status: Status
}

#if !targetEnvironment(macCatalyst)
struct FrankenTTSRunActivityAttributes: ActivityAttributes {
    typealias ContentState = FrankenTTSRunContentState
    var runID: UUID
    var startedAt: Date
}
#else
struct FrankenTTSRunActivityAttributes {
    typealias ContentState = FrankenTTSRunContentState
    var runID: UUID
    var startedAt: Date
}
#endif

struct FrankenTTSWidgetSnapshot: Codable, Hashable {
    enum Readiness: String, Codable {
        case modelRequired
        case warming
        case ready
        case working
        case complete
        case needsAttention
    }

    var readiness: Readiness
    var headline: String
    var detail: String
    var updatedAt: Date

    static let placeholder = FrankenTTSWidgetSnapshot(
        readiness: .ready,
        headline: "Voice Forge ready",
        detail: "Private, on-device speech",
        updatedAt: .now
    )
}

enum FrankenTTSSharedStore {
    static let suiteName = "group.com.frankentts.FrankenTTS"
    private static let snapshotKey = "widget.snapshot.v1"
    private static let stagedTextKey = "intent.staged-text.v1"

    static func loadSnapshot() -> FrankenTTSWidgetSnapshot {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = defaults.data(forKey: snapshotKey),
              let snapshot = try? JSONDecoder().decode(FrankenTTSWidgetSnapshot.self, from: data)
        else { return .placeholder }
        return snapshot
    }

    static func save(_ snapshot: FrankenTTSWidgetSnapshot) {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = try? JSONEncoder().encode(snapshot)
        else { return }
        defaults.set(data, forKey: snapshotKey)
    }

    static func stage(text: String) {
        UserDefaults(suiteName: suiteName)?.set(text, forKey: stagedTextKey)
    }

    static func consumeStagedText() -> String? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let text = defaults.string(forKey: stagedTextKey)
        else { return nil }
        defaults.removeObject(forKey: stagedTextKey)
        return text
    }
}
