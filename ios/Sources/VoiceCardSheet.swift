// Share sheet for a voice card: preview the card, explain it in plain words, share
// the PNG. The explanation matters — the card looks like art, and people should know
// the picture itself is the voice.

import Photos
import SwiftUI
import CryptoKit

private struct VoiceCardArtifact: Sendable {
    let data: Data
    let url: URL
}

private actor VoiceCardArtifactStore {
    static let shared = VoiceCardArtifactStore()

    private var artifacts: [String: VoiceCardArtifact] = [:]

    func artifact(for voice: EnrolledVoice) async throws -> VoiceCardArtifact {
        let key = cacheKey(for: voice)
        if let artifact = artifacts[key] { return artifact }

        let png = try await VoicePrintCard.pngData(name: voice.name, vector: voice.vector)
        try Task.checkCancellation()
        let safeName = voice.name
            .map { $0.isLetter || $0.isNumber ? $0 : "-" }
            .prefix(48)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("franken_tts-voice-cards", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let url = directory.appendingPathComponent(
            "\(safeName.isEmpty ? "Voice" : String(safeName))-\(key.prefix(12))-voice-card.png"
        )
        try png.write(to: url, options: .atomic)
        let artifact = VoiceCardArtifact(data: png, url: url)
        artifacts[key] = artifact
        return artifact
    }

    private func cacheKey(for voice: EnrolledVoice) -> String {
        var hash = SHA256()
        hash.update(data: Data(voice.name.utf8))
        for value in voice.vector {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { hash.update(bufferPointer: $0) }
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

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
                        .accessibilityIdentifier("voice-card-preview")
                } else if failed {
                    Text("Something went wrong making the card. Close this and try again.")
                        .font(.system(size: Lab.typeSize(14)))
                        .foregroundStyle(Lab.danger)
                } else {
                    ProgressView()
                        .frame(maxWidth: .infinity, minHeight: 200)
                        .accessibilityIdentifier("voice-card-progress")
                }
                Text(
                    "The green mosaic is \(voice.name), written as thousands of tiny tiles. Send this picture to a friend; in their FrankenTTS app they tap \"Add a voice from a picture\", pick it, and the voice appears in their library. It survives screenshots and messaging apps, and it holds only the small voiceprint, never a recording of you."
                )
                .font(.system(size: Lab.typeSize(14)))
                .foregroundStyle(Lab.textSecondary)
                Text("Only share a voice that is yours to share.")
                    .font(.system(size: Lab.typeSize(12), design: .monospaced))
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
                            .font(.system(size: Lab.typeSize(13)))
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
                let artifact = try await VoiceCardArtifactStore.shared.artifact(for: voice)
                try Task.checkCancellation()
                guard let decodedPreview = UIImage(data: artifact.data) else {
                    throw EngineError.native("cannot decode the rendered voice-card preview")
                }
                preview = decodedPreview
                cardData = artifact.data
                cardUrl = artifact.url
            } catch {
                if !Task.isCancelled { failed = true }
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
