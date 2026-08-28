import Foundation

#if !targetEnvironment(macCatalyst)
import ActivityKit
#endif

/// Privacy-safe state shared with WidgetKit and Live Activities. Their status
/// payloads never contain source text, cloned-voice names, or generated audio.
/// A separate, length-limited text slot exists only for an explicit Share or
/// App Intent handoff and expires if the app does not consume it promptly.
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
    var totalIsUpperBound: Bool
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
        headline: "Open the Voice Forge",
        detail: "Create private, on-device speech",
        updatedAt: .now
    )
}

enum FrankenTTSSharedStore {
    static let suiteName = "group.com.frankentts.FrankenTTS"
    private static let snapshotKey = "widget.snapshot.v1"
    private static let stagedTextKey = "intent.staged-text.v1"
    private static let stagedTextDateKey = "intent.staged-text-date.v1"
    private static let stagedTextLifetime: TimeInterval = 60 * 60
    private static let maximumTextLength = 600

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
        guard let defaults = UserDefaults(suiteName: suiteName) else { return }
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            defaults.removeObject(forKey: stagedTextKey)
            defaults.removeObject(forKey: stagedTextDateKey)
            return
        }
        defaults.set(String(value.prefix(maximumTextLength)), forKey: stagedTextKey)
        defaults.set(Date().timeIntervalSince1970, forKey: stagedTextDateKey)
    }

    static func consumeStagedText() -> String? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let text = defaults.string(forKey: stagedTextKey)
        else { return nil }
        let stagedAt = defaults.double(forKey: stagedTextDateKey)
        defaults.removeObject(forKey: stagedTextKey)
        defaults.removeObject(forKey: stagedTextDateKey)
        let age = Date().timeIntervalSince1970 - stagedAt
        guard stagedAt > 0, age >= 0, age <= stagedTextLifetime
        else { return nil }
        return text
    }
}
