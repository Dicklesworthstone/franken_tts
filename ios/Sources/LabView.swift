// The laboratory: one scrolling screen mirroring the site's playground.

import AVFoundation
import SwiftUI
import UIKit

@MainActor
@Observable
final class LabModel {
    let store = ModelStore()
    let engine = Engine()
    let recorder = AudioRecorder()
    let presets = Engine.presets()

    var selectedVoice: String = "matt"
    var clonedName: String?
    var clonedVector: [Float]?

    var text =
        "The rainbow is a division of white light into many beautiful colors. Now, spoken entirely on this phone."
    var seed: UInt64 = 0

    var isSynthesizing = false
    var synthesisSeconds = 0.0
    var lastError: String?
    var lastAudio: [Float]?
    var lastRealTimeFactor: Double?
    var player: AVAudioPlayer?
    var shareUrl: URL?

    var lowMemoryDevice: Bool {
        ProcessInfo.processInfo.physicalMemory < 6 * 1024 * 1024 * 1024
    }

    func speakerVector() throws -> [Float] {
        if selectedVoice == "__cloned__", let clonedVector { return clonedVector }
        return try Engine.presetVector(named: selectedVoice)
    }

    // The engine's load and synthesize are long BLOCKING calls made from an actor, which
    // parks one cooperative-pool thread for their duration. Tolerable for this app (the UI
    // runs on the main actor and nothing else contends), but a dedicated-thread executor is
    // the right refinement if background work ever grows.
    func synthesize() {
        guard !isSynthesizing else { return }
        isSynthesizing = true
        lastError = nil
        synthesisSeconds = 0
        let text = self.text
        let seed = self.seed
        let ticker = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.synthesisSeconds += 0.5 }
        }
        Task {
            defer { ticker.invalidate() }
            do {
                let speaker = try speakerVector()
                if await !engine.isLoaded {
                    try await engine.load(modelDirectory: store.modelDirectory)
                }
                let started = Date()
                let pcm = try await engine.synthesize(text: text, speaker: speaker, seed: seed)
                let elapsed = Date().timeIntervalSince(started)
                lastAudio = pcm
                lastRealTimeFactor = (Double(pcm.count) / Double(WavWriter.sampleRate)) / elapsed
                try startPlayback(of: pcm)
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } catch {
                lastError = error.localizedDescription
            }
            isSynthesizing = false
        }
    }

    private func startPlayback(of pcm: [Float]) throws {
        let wav = WavWriter.data(from: pcm)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("franken_tts.wav")
        try wav.write(to: url)
        shareUrl = url
        try AVAudioSession.sharedInstance().setCategory(.playback)
        player = try AVAudioPlayer(contentsOf: url)
        player?.play()
    }

    func finishEnrollment(named name: String) {
        let pcm = recorder.stop()
        guard pcm.count >= 3 * Int(AudioRecorder.targetRate) else {
            lastError = "recording too short; read at least a few seconds of the script"
            return
        }
        Task {
            do {
                if await !engine.isLoaded {
                    try await engine.load(modelDirectory: store.modelDirectory)
                }
                let vector = try await engine.enroll(pcm: pcm)
                clonedVector = vector
                clonedName = name.isEmpty ? "my voice" : name
                selectedVoice = "__cloned__"
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    /// Frees the ~2.3 GB engine heap; the next synthesis reloads it.
    func unloadEngineForMemoryPressure() {
        guard !isSynthesizing else { return }
        Task { await engine.unload() }
    }
}

struct LabView: View {
    @State private var model = LabModel()
    @State private var showConsent = false
    @State private var showEnrollment = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Lab.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 26) {
                    header
                    specimenCard
                    voicesCard
                    utteranceCard
                    footer
                }
                .padding(16)
            }
        }
        .sheet(isPresented: $showEnrollment) {
            EnrollmentSheet(model: model)
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIApplication.didReceiveMemoryWarningNotification)
        ) { _ in
            model.unloadEngineForMemoryPressure()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { model.unloadEngineForMemoryPressure() }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(
                        LinearGradient(
                            colors: [Lab.emeraldDeep, Lab.emerald], startPoint: .bottomLeading,
                            endPoint: .topTrailing))
                    .frame(width: 42, height: 42)
                Text("F")
                    .font(.system(size: 24, weight: .black, design: .monospaced))
                    .foregroundStyle(.black)
            }
            .overlay(alignment: .topLeading) { Bolt().offset(x: -4, y: -4) }
            VStack(alignment: .leading, spacing: 2) {
                Text("FrankenTTS")
                    .font(.system(size: 22, weight: .black))
                    .foregroundStyle(Lab.textPrimary)
                Text("VOICE_ALIVE")
                    .font(.system(size: 8, weight: .black, design: .monospaced))
                    .kerning(2)
                    .foregroundStyle(Lab.emerald)
            }
            Spacer()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("FrankenTTS, the monster voice engine")
    }

    private var specimenCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "01 · The Specimen (model)")
                switch model.store.phase {
                case .idle:
                    Text(
                        "First use downloads the quantized model (≈2.0 GB) into this app's private storage: verified against pinned digests, resumable, kept until you clear it."
                    )
                    .font(.system(size: 14))
                    .foregroundStyle(Lab.textSecondary)
                    if model.lowMemoryDevice {
                        Text(
                            "This device reports under 6 GB of memory; the engine may not fit. A recent Pro-class iPhone is recommended."
                        )
                        .font(.system(size: 13))
                        .foregroundStyle(Lab.danger)
                    }
                    Button("Download the 2.0 GB model") { showConsent = true }
                        .buttonStyle(PrimaryButtonStyle())
                        .confirmationDialog(
                            "Download 2.0 GB now? It stays on this device, resumes if interrupted, and Clear Model removes it.",
                            isPresented: $showConsent, titleVisibility: .visible
                        ) {
                            Button("Start the download") { model.store.startDownload() }
                            Button("Not now", role: .cancel) {}
                        }
                case .downloading(let asset, let done, let total, let eta):
                    ProgressView(value: Double(done), total: Double(total))
                        .tint(Lab.emerald)
                    Text(
                        "\(asset)  ·  \(Self.gigabytes(done)) / \(Self.gigabytes(total)) GB  ·  \(eta)"
                    )
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                case .verifying(let asset):
                    ProgressView().tint(Lab.emerald)
                    Text("verifying \(asset)…")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                case .ready:
                    HStack {
                        Image(systemName: "checkmark.seal.fill").foregroundStyle(Lab.emerald)
                        Text("Model on device · \(Self.gigabytes(model.store.cachedBytes)) GB")
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundStyle(Lab.textPrimary)
                        Spacer()
                        Button("Clear") { model.store.clear() }
                            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    }
                case .failed(let message):
                    Text(message).font(.system(size: 13)).foregroundStyle(Lab.danger)
                    Button("Retry") { model.store.startDownload() }
                        .buttonStyle(PrimaryButtonStyle())
                }
            }
        }
    }

    private var voicesCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "02 · The Voice")
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10
                ) {
                    ForEach(model.presets) { preset in
                        VoiceTile(
                            name: preset.name, character: preset.character,
                            selected: model.selectedVoice == preset.name
                        ) { model.selectedVoice = preset.name }
                    }
                    VoiceTile(
                        name: model.clonedName ?? "your voice",
                        character: model.clonedVector == nil
                            ? "read a short script to clone it" : "locally cloned",
                        selected: model.selectedVoice == "__cloned__",
                        accent: true
                    ) {
                        if model.clonedVector != nil {
                            model.selectedVoice = "__cloned__"
                        } else if model.store.phase == .ready {
                            showEnrollment = true
                        }
                    }
                }
                Text(
                    "Cloning runs the speaker encoder on this phone; the recording is discarded once the 4 KB voice vector exists. Clone only voices you have the right to use."
                )
                .font(.system(size: 12))
                .foregroundStyle(Lab.textSecondary)
            }
        }
    }

    private var utteranceCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "03 · The Utterance")
                TextEditor(text: Binding(
                    get: { model.text },
                    set: { model.text = String($0.prefix(600)) }
                ))
                .scrollContentBackground(.hidden)
                .frame(minHeight: 110)
                .padding(8)
                .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(Lab.textPrimary)
                .font(.system(size: 15))
                HStack {
                    Text("\(model.text.count) / 600")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    Spacer()
                    Text("seed")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    TextField(
                        "0",
                        text: Binding(
                            get: { String(model.seed) },
                            set: { model.seed = UInt64($0.filter(\.isNumber).prefix(10)) ?? 0 }
                        )
                    )
                    .keyboardType(.numberPad)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Lab.textPrimary)
                    .frame(width: 74)
                    .padding(.vertical, 5)
                    .padding(.horizontal, 8)
                    .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel("Seed")
                    Button {
                        model.seed = UInt64.random(in: 0..<100_000)
                    } label: {
                        Image(systemName: "dice")
                    }
                    .buttonStyle(GhostButtonStyle())
                    .accessibilityLabel("Randomize seed")
                }
                Button(model.isSynthesizing ? "Synthesizing…" : "⚡ Synthesize") {
                    model.synthesize()
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(model.isSynthesizing || model.store.phase != .ready)
                if model.isSynthesizing {
                    Text(
                        "\(Int(model.synthesisSeconds))s elapsed · first run also loads the model; no percentage is shown because none would be honest"
                    )
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                }
                if let audio = model.lastAudio {
                    WaveformView(samples: audio)
                    HStack(spacing: 10) {
                        Button("▶ Play again") {
                            model.player?.currentTime = 0
                            model.player?.play()
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        if let url = model.shareUrl {
                            ShareLink(item: url) {
                                Text("Share WAV")
                            }
                            .buttonStyle(GhostButtonStyle())
                        }
                        Spacer()
                        if let factor = model.lastRealTimeFactor {
                            Text(String(format: "%.2f× real time", factor))
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Lab.textSecondary)
                        }
                    }
                }
                if let error = model.lastError {
                    Text(error).font(.system(size: 13)).foregroundStyle(Lab.danger)
                }
            }
        }
    }

    private var footer: some View {
        Text("Runs entirely on this device · frankentts.com")
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(Lab.textSecondary)
            .frame(maxWidth: .infinity)
            .padding(.top, 6)
    }

    private static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f", Double(bytes) / 1_073_741_824.0)
    }
}

struct VoiceTile: View {
    let name: String
    let character: String
    let selected: Bool
    var accent = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 4) {
                Text(name)
                    .font(.system(size: 15, weight: .black))
                    .foregroundStyle(Lab.textPrimary)
                Text(character)
                    .font(.system(size: 11))
                    .foregroundStyle(Lab.textSecondary)
                    .multilineTextAlignment(.leading)
                    .frame(minHeight: 28, alignment: .top)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(
                        selected
                            ? Lab.emerald
                            : (accent ? Lab.emerald.opacity(0.35) : Lab.stroke),
                        lineWidth: selected ? 1.5 : 1))
        }
        .accessibilityLabel("\(name): \(character)\(selected ? ", selected" : "")")
    }
}

struct EnrollmentSheet: View {
    @Bindable var model: LabModel
    @Environment(\.dismiss) private var dismiss
    @State private var cloneName = ""

    private static let script = """
        Please call Stella. Ask her to bring these things with her from the store: six \
        spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for \
        her brother Bob. We also need a small plastic snake and a big toy frog for the \
        kids. She can scoop these things into three red bags, and we will go meet her \
        Wednesday at the train station.

        When the sunlight strikes raindrops in the air, they act as a prism and form a \
        rainbow. The rainbow is a division of white light into many beautiful colors.
        """

    var body: some View {
        ZStack {
            Lab.background.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 16) {
                LabLabel(text: "Clone your voice")
                Text("Read this aloud (about thirty seconds):")
                    .font(.system(size: 14))
                    .foregroundStyle(Lab.textSecondary)
                ScrollView {
                    Text(Self.script)
                        .font(.system(size: 15))
                        .foregroundStyle(Lab.textPrimary)
                        .padding(12)
                }
                .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                TextField("name your voice", text: $cloneName)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                    .foregroundStyle(Lab.textPrimary)
                HStack {
                    if model.recorder.isRecording {
                        Button("⏹ Stop & clone (\(Int(model.recorder.seconds))s)") {
                            model.finishEnrollment(named: cloneName)
                            dismiss()
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    } else {
                        Button("🎙 Start recording") {
                            do {
                                try model.recorder.start()
                            } catch {
                                model.lastError = error.localizedDescription
                            }
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    }
                    Spacer()
                    Button("Cancel") {
                        _ = model.recorder.stop()
                        dismiss()
                    }
                    .buttonStyle(GhostButtonStyle())
                }
            }
            .padding(18)
        }
        .presentationDetents([.large])
    }
}
