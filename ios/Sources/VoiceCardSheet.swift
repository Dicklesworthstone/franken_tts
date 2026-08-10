// Share sheet for a voice card: preview the card, explain it in plain words, share
// the PNG. The explanation matters — the card looks like art, and people should know
// the picture itself is the voice.

import Photos
import SwiftUI

struct VoiceCardSheet: View {
    let voice: EnrolledVoice
    @Environment(\.dismiss) private var dismiss

    @State private var cardUrl: URL?
    @State private var cardData: Data?
    @State private var preview: UIImage?
    @State private var failed = false
    @State private var savedToPhotos = false
    @State private var saveError: String?

    var body: some View {
        ZStack {
            Lab.background.ignoresSafeArea()
            // Scrolls so the share button stays reachable on small screens, where the
            // preview plus the explanation outgrow the sheet.
            ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    LabLabel(text: "Voice card")
                    Spacer()
                    Button("Done") { dismiss() }
                        .buttonStyle(GhostButtonStyle())
                }
                if let preview {
                    Image(uiImage: preview)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14).stroke(Lab.stroke, lineWidth: 1)
                        )
                        .frame(maxWidth: .infinity)
                        .accessibilityLabel("Voice card for \(voice.name)")
                } else if failed {
                    Text("Something went wrong making the card. Close this and try again.")
                        .font(.system(size: 14))
                        .foregroundStyle(Lab.danger)
                } else {
                    ProgressView()
                        .frame(maxWidth: .infinity, minHeight: 200)
                }
                Text(
                    "The green mosaic is \(voice.name), written as thousands of tiny tiles. Send this picture to a friend; in their FrankenTTS app they tap \"Add a voice from a picture\", pick it, and the voice appears in their library. It survives screenshots and messaging apps, and it holds only the small voiceprint, never a recording of you."
                )
                .font(.system(size: 14))
                .foregroundStyle(Lab.textSecondary)
                Text("Only share a voice that is yours to share.")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                if let cardUrl {
                    ShareLink(item: cardUrl) {
                        HStack(spacing: 8) {
                            Image(systemName: "square.and.arrow.up")
                            Text("Share the card")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    Button {
                        saveToPhotos()
                    } label: {
                        HStack(spacing: 8) {
                            Image(
                                systemName: savedToPhotos
                                    ? "checkmark" : "square.and.arrow.down")
                            Text(savedToPhotos ? "Saved to Photos" : "Save to Photos")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                    .disabled(savedToPhotos)
                    if let saveError {
                        Text(saveError)
                            .font(.system(size: 13))
                            .foregroundStyle(Lab.danger)
                    }
                }
            }
            .padding(18)
            }
        }
        .presentationDetents([.large])
        .task {
            do {
                let png = try VoicePrintCard.pngData(name: voice.name, vector: voice.vector)
                let safeName = voice.name.map { $0.isLetter || $0.isNumber ? $0 : "-" }
                let directory = FileManager.default.temporaryDirectory
                    .appendingPathComponent(
                        "franken_tts-card-\(ProcessInfo.processInfo.globallyUniqueString)",
                        isDirectory: true)
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                let url = directory.appendingPathComponent(
                    "\(String(safeName))-voice-card.png")
                try png.write(to: url)
                preview = UIImage(data: png)
                cardData = png
                cardUrl = url
            } catch {
                failed = true
            }
        }
    }

    /// Save the EXACT PNG bytes as a photo asset — through PHAssetCreationRequest, not
    /// UIImage re-encoding, so the lossless data chunk inside the file survives.
    private func saveToPhotos() {
        guard let cardData else { return }
        saveError = nil
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                Task { @MainActor in
                    saveError =
                        "Photos access is off for FrankenTTS; allow \"Add Photos Only\" in Settings and try again."
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: cardData, options: nil)
            } completionHandler: { success, error in
                Task { @MainActor in
                    if success {
                        savedToPhotos = true
                    } else {
                        saveError = error?.localizedDescription
                            ?? "Saving to Photos did not work; try again."
                    }
                }
            }
        }
    }
}
