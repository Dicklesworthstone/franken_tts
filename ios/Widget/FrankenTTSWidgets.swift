import ActivityKit
import SwiftUI
import WidgetKit

private let forgeGreen = Color(red: 0.20, green: 0.83, blue: 0.60)
private let forgeInk = Color(red: 0.005, green: 0.035, blue: 0.022)

struct ForgeTimelineEntry: TimelineEntry {
    let date: Date
    let snapshot: FrankenTTSWidgetSnapshot
}

struct ForgeTimelineProvider: TimelineProvider {
    func placeholder(in context: Context) -> ForgeTimelineEntry {
        ForgeTimelineEntry(date: .now, snapshot: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (ForgeTimelineEntry) -> Void) {
        completion(ForgeTimelineEntry(date: .now, snapshot: FrankenTTSSharedStore.loadSnapshot()))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<ForgeTimelineEntry>) -> Void) {
        let entry = ForgeTimelineEntry(date: .now, snapshot: FrankenTTSSharedStore.loadSnapshot())
        completion(Timeline(entries: [entry], policy: .after(.now.addingTimeInterval(15 * 60))))
    }
}

struct FrankenTTSForgeWidget: Widget {
    let kind = "FrankenTTSForgeWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: ForgeTimelineProvider()) { entry in
            ForgeWidgetView(entry: entry)
                .containerBackground(for: .widget) {
                    LinearGradient(
                        colors: [forgeInk, Color.black, forgeGreen.opacity(0.13)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                }
                .widgetURL(URL(string: "frankentts://forge"))
        }
        .configurationDisplayName("Voice Forge")
        .description("See private model readiness and jump into the Voice Forge.")
        .supportedFamilies([.systemSmall, .systemMedium, .accessoryInline, .accessoryRectangular])
    }
}

private struct ForgeWidgetView: View {
    let entry: ForgeTimelineEntry
    @Environment(\.widgetFamily) private var family

    var body: some View {
        switch family {
        case .accessoryInline:
            Label(entry.snapshot.headline, systemImage: icon)
        case .accessoryRectangular:
            VStack(alignment: .leading, spacing: 2) {
                Label(entry.snapshot.headline, systemImage: icon).font(.headline)
                Text(entry.snapshot.detail).font(.caption).lineLimit(1)
            }
        default:
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: icon)
                        .font(.title3.weight(.bold))
                        .foregroundStyle(forgeGreen)
                    Spacer()
                    Text("VOICE_ALIVE")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .tracking(1.4)
                        .foregroundStyle(forgeGreen.opacity(0.8))
                }
                Spacer(minLength: 0)
                Text(entry.snapshot.headline)
                    .font(.headline)
                    .foregroundStyle(.white)
                    .lineLimit(2)
                Text(entry.snapshot.detail)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.68))
                    .lineLimit(family == .systemMedium ? 2 : 1)
            }
        }
    }

    private var icon: String {
        switch entry.snapshot.readiness {
        case .modelRequired: "externaldrive.badge.plus"
        case .warming: "brain.head.profile"
        case .ready: "bolt.circle"
        case .working: "waveform.path.ecg"
        case .complete: "checkmark.seal.fill"
        case .needsAttention: "exclamationmark.triangle.fill"
        }
    }
}

struct FrankenTTSLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: FrankenTTSRunActivityAttributes.self) { context in
            ForgeLockScreenView(context: context)
                .activityBackgroundTint(forgeInk)
                .activitySystemActionForegroundColor(forgeGreen)
                .widgetURL(URL(string: "frankentts://forge"))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: statusIcon(context.state.status))
                        .font(.title2.weight(.black))
                        .foregroundStyle(forgeGreen)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text(timerInterval: context.attributes.startedAt...Date.distantFuture, countsDown: false)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                DynamicIslandExpandedRegion(.center) {
                    Text(context.state.stage)
                        .font(.headline)
                        .lineLimit(1)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack(alignment: .firstTextBaseline) {
                            Text(context.state.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                            Spacer(minLength: 8)
                            if context.state.status == .running || context.state.status == .preparing {
                                Link("Stop", destination: URL(string: "frankentts://cancel")!)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.red)
                            }
                        }
                        ForgeUnitRail(state: context.state)
                    }
                }
            } compactLeading: {
                Image(systemName: statusIcon(context.state.status))
                    .foregroundStyle(forgeGreen)
            } compactTrailing: {
                if context.state.completedUnits > 0 {
                    Text(context.state.completedUnits, format: .number)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(forgeGreen)
                } else {
                    Image(systemName: "bolt.fill").foregroundStyle(forgeGreen)
                }
            } minimal: {
                Image(systemName: statusIcon(context.state.status))
                    .foregroundStyle(forgeGreen)
            }
            .widgetURL(URL(string: "frankentts://forge"))
            .keylineTint(forgeGreen)
        }
    }
}

private struct ForgeLockScreenView: View {
    let context: ActivityViewContext<FrankenTTSRunActivityAttributes>

    var body: some View {
        HStack(spacing: 13) {
            ZStack {
                Circle().fill(forgeGreen.opacity(0.14)).frame(width: 44, height: 44)
                Image(systemName: statusIcon(context.state.status))
                    .font(.headline.weight(.black))
                    .foregroundStyle(forgeGreen)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(context.state.stage).font(.headline).lineLimit(1)
                Text(context.state.detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                ForgeUnitRail(state: context.state)
            }
            Spacer(minLength: 4)
            Text(timerInterval: context.attributes.startedAt...Date.distantFuture, countsDown: false)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(15)
    }
}

private struct ForgeUnitRail: View {
    let state: FrankenTTSRunActivityAttributes.ContentState

    var body: some View {
        if state.totalUnits > 0 {
            ProgressView(value: Double(state.completedUnits), total: Double(state.totalUnits))
                .tint(forgeGreen)
        } else if state.status == .running || state.status == .preparing {
            HStack(spacing: 4) {
                ForEach(0..<6, id: \.self) { index in
                    Capsule()
                        .fill(index < Int(state.completedUnits % 7) ? forgeGreen : forgeGreen.opacity(0.18))
                        .frame(height: 3)
                }
            }
        }
    }
}

private func statusIcon(_ status: FrankenTTSRunActivityAttributes.ContentState.Status) -> String {
    switch status {
    case .preparing: "bolt.circle"
    case .running: "waveform.path.ecg"
    case .cancelling: "bolt.slash"
    case .complete: "checkmark.seal.fill"
    case .cancelled: "stop.circle"
    case .failed: "exclamationmark.triangle.fill"
    }
}

@main
struct FrankenTTSWidgetBundle: WidgetBundle {
    var body: some Widget {
        FrankenTTSForgeWidget()
        FrankenTTSLiveActivity()
    }
}
