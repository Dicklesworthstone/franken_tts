import AVFoundation
import Observation
import SwiftUI

struct SynthesisHistorySheet: View {
    @Bindable var history: SynthesisHistoryStore
    @Environment(\.dismiss) private var dismiss
    @State private var player: AVAudioPlayer?
    @State private var playingID: UUID?
    @State private var confirmClear = false

    var body: some View {
        NavigationStack {
            ZStack {
                LaboratoryBackground()
                if history.entries.isEmpty {
                    ContentUnavailableView(
                        "No recent voices",
                        systemImage: "waveform.badge.plus",
                        description: Text("Finished audio appears here automatically after synthesis.")
                    )
                    .foregroundStyle(Lab.textSecondary)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            privacyNote
                            ForEach(history.entries) { entry in
                                historyCard(entry)
                            }
                        }
                        .padding(16)
                    }
                    .scrollIndicators(.hidden)
                }
            }
            .navigationTitle("Recent voices")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                if !history.entries.isEmpty {
                    ToolbarItem(placement: .destructiveAction) {
                        Button("Clear All", role: .destructive) { confirmClear = true }
                    }
                }
            }
        }
        .alert("Clear recent voices?", isPresented: $confirmClear) {
            Button("Clear All", role: .destructive) {
                player?.stop()
                player = nil
                playingID = nil
                history.deleteAll()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This permanently removes every locally saved audio result. "
                    + "Your voices and model stay intact."
            )
        }
        .onDisappear { player?.stop() }
        .accessibilityIdentifier("synthesis-history-sheet")
    }

    private var privacyNote: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "lock.shield.fill")
                .foregroundStyle(Lab.emerald)
            VStack(alignment: .leading, spacing: 5) {
                Text(
                    "Only audio, voice name, date, and duration are stored. "
                        + "Utterance text, seed, and voiceprints are never written to history. "
                        + "The newest 12 clips are kept for up to 7 days."
                )
                .font(.system(size: Lab.typeSize(11), weight: .medium))
                .foregroundStyle(Lab.textSecondary)
                Text(storageSummary)
                    .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                    .foregroundStyle(Lab.cyan)
            }
        }
        .padding(12)
        .background(Lab.panelSoft, in: RoundedRectangle(cornerRadius: 14))
        .accessibilityElement(children: .combine)
    }

    private var storageSummary: String {
        let size = ByteCountFormatter.string(
            fromByteCount: Int64(history.storageBytes),
            countStyle: .file
        )
        return "\(history.entries.count) clips · \(size) on this device"
    }

    private func historyCard(_ entry: SynthesisHistoryEntry) -> some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(entry.voiceLabel)
                            .font(.system(size: Lab.typeSize(16), weight: .black, design: .rounded))
                            .foregroundStyle(Lab.textPrimary)
                        Text(entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.system(size: Lab.typeSize(10), design: .monospaced))
                            .foregroundStyle(Lab.textSecondary)
                    }
                    Spacer()
                    Text(Self.duration(entry.durationSeconds))
                        .font(.system(size: Lab.typeSize(10), weight: .bold, design: .monospaced))
                        .foregroundStyle(Lab.cyan)
                }
                HStack(spacing: 10) {
                    Button {
                        play(entry)
                    } label: {
                        Label(playingID == entry.id ? "Replay" : "Play", systemImage: "play.fill")
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                    if let url = history.fileURL(for: entry) {
                        ShareLink(item: url) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(GhostButtonStyle())
                    }
                    Spacer()
                    Button(role: .destructive) {
                        if playingID == entry.id {
                            player?.stop()
                            player = nil
                            playingID = nil
                        }
                        history.delete(entry)
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    .accessibilityLabel("Delete \(entry.voiceLabel) result")
                }
            }
        }
        .accessibilityIdentifier("history-entry-\(entry.id.uuidString)")
    }

    private func play(_ entry: SynthesisHistoryEntry) {
        guard let url = history.fileURL(for: entry) else { return }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            let nextPlayer = try AVAudioPlayer(contentsOf: url)
            nextPlayer.play()
            player?.stop()
            player = nextPlayer
            playingID = entry.id
        } catch {
            player = nil
            playingID = nil
        }
    }

    private static func duration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
