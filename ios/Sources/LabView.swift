// The laboratory: one scrolling screen mirroring the site's playground.

import AVFoundation
import PhotosUI
import SwiftUI
import UIKit

@MainActor
@Observable
final class LabModel {
    let store = ModelStore()
    let engine = Engine()
    let recorder = AudioRecorder()
    let presets = Engine.presets()

    let library = VoiceLibrary()

    /// A preset name, or "voice:<uuid>" for an enrolled voice.
    var selectedVoice: String = "matt"
    /// When re-recording an existing voice, its id; nil enrolls a new one.
    var enrollmentTarget: UUID?
    /// Set when an enrollment just saved, so the sheet knows to dismiss.
    var enrollmentSaved = false

    var text =
        "The rainbow is a division of white light into many beautiful colors. Now, spoken entirely on this phone."
    var seed: UInt64 = 0

    var isSynthesizing = false
    var synthesisSeconds = 0.0
    var lastError: String?
    var lastAudio: [Float]?
    var lastRealTimeFactor: Double?
    var player: AVAudioPlayer?
    /// The playback WAV (internal); shares go out as M4A and MP4.
    var wavUrl: URL?
    var m4aUrl: URL?
    var videoUrl: URL?
    var isExportingVideo = false
    var videoProgress = 0.0
    /// Bumped per synthesis so a slow export cannot stamp its output onto a newer clip.
    private var synthesisGeneration = 0

    var lowMemoryDevice: Bool {
        ProcessInfo.processInfo.physicalMemory < 6 * 1024 * 1024 * 1024
    }

    func speakerVector() throws -> [Float] {
        if let id = enrolledSelection() {
            guard let voice = library.voice(id: id) else {
                throw EngineError.native("that enrolled voice no longer exists; pick another")
            }
            return voice.vector
        }
        return try Engine.presetVector(named: selectedVoice)
    }

    func enrolledSelection() -> UUID? {
        guard selectedVoice.hasPrefix("voice:") else { return nil }
        return UUID(uuidString: String(selectedVoice.dropFirst("voice:".count)))
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
        // The screen must not sleep mid-run: suspension kills a minutes-long synthesis.
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            defer {
                ticker.invalidate()
                UIApplication.shared.isIdleTimerDisabled = false
            }
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
            } catch {
                lastError = error.localizedDescription
            }
            isSynthesizing = false
        }
    }

    private func startPlayback(of pcm: [Float]) throws {
        let wav = WavWriter.data(from: pcm)
        // Unique per synthesis: an in-flight video export reads the previous WAV for its
        // audio track, and overwriting it mid-read would mux corrupt audio.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "franken_tts-\(ProcessInfo.processInfo.globallyUniqueString).wav")
        try wav.write(to: url)
        wavUrl = url
        m4aUrl = nil
        videoUrl = nil
        synthesisGeneration += 1
        let generation = synthesisGeneration
        try AVAudioSession.sharedInstance().setCategory(.playback)
        player = try AVAudioPlayer(contentsOf: url)
        player?.play()
        // The share default is the small file; convert as soon as audio exists.
        Task {
            let converted = try? await MediaExporter.exportM4A(fromWav: url)
            if generation == synthesisGeneration { m4aUrl = converted }
        }
    }

    /// The label stamped on the video's voice pill.
    var currentVoiceLabel: String {
        if let id = enrolledSelection(), let voice = library.voice(id: id) {
            return voice.name
        }
        return selectedVoice.capitalized
    }

    func exportVideo() {
        guard let wavUrl, let audio = lastAudio, !isExportingVideo else { return }
        isExportingVideo = true
        videoProgress = 0
        let label = currentVoiceLabel
        let generation = synthesisGeneration
        Task {
            defer { isExportingVideo = false }
            do {
                let rendered = try await MediaExporter.exportVideo(
                    pcm: audio, voiceLabel: label, wavUrl: wavUrl
                ) { [weak self] fraction in
                    Task { @MainActor in self?.videoProgress = fraction }
                }
                if generation == synthesisGeneration { videoUrl = rendered }
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    var isEnrolling = false

    func finishEnrollment(named name: String) {
        let raw = recorder.stop()
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            lastError = "name the voice before saving it"
            return
        }
        isEnrolling = true
        enrollmentSaved = false
        Task {
            defer { isEnrolling = false }
            do {
                let pcm = try Self.conditioned(raw)
                guard pcm.count >= 3 * Int(AudioRecorder.targetRate) else {
                    throw EngineError.native(
                        "recording too short; a few sentences of the script is all it takes")
                }
                if await !engine.isLoaded {
                    try await engine.load(modelDirectory: store.modelDirectory)
                }
                let vector = try await engine.enroll(pcm: pcm)
                let selected: UUID
                if let target = enrollmentTarget {
                    try library.replaceVector(id: target, with: vector)
                    try library.rename(id: target, to: trimmedName)
                    selected = target
                } else {
                    selected = try library.add(name: trimmedName, vector: vector).id
                }
                enrollmentTarget = nil
                selectedVoice = "voice:\(selected.uuidString)"
                enrollmentSaved = true
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    /// Trim edge silence and peak-normalize before enrollment. The encoder embeds
    /// whatever it is given: a quiet recording embeds "a quiet voice" and everything
    /// synthesized from it inherits that, which on a real device meant inaudible output.
    /// Refusing outright silence beats enrolling it.
    private static func conditioned(_ pcm: [Float]) throws -> [Float] {
        var peak: Float = 0
        for value in pcm { peak = max(peak, abs(value)) }
        guard peak > 0.01 else {
            throw EngineError.native(
                "we couldn't hear you (peak level \(String(format: "%.3f", peak))); check the microphone and try again")
        }
        let gate = peak * 0.02
        let first = pcm.firstIndex { abs($0) > gate } ?? 0
        let last = pcm.lastIndex { abs($0) > gate } ?? pcm.count - 1
        let pad = Int(AudioRecorder.targetRate * 0.2)
        let low = max(0, first - pad)
        let high = min(pcm.count, last + 1 + pad)
        let scale = 0.85 / peak
        return pcm[low..<high].map { $0 * scale }
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
    @State private var showGalaxy = false
    @State private var renameTarget: EnrolledVoice?
    @State private var renameText = ""
    @State private var cardVoice: EnrolledVoice?
    @State private var importItem: PhotosPickerItem?
    @State private var importFailed = false
    @State private var importCount = 0
    /// Bumped to refresh the play/pause icon, which tracks external playback state.
    @State private var playbackTick = 0
    @Environment(\.scenePhase) private var scenePhase
    @FocusState private var focusedField: Bool

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
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { focusedField = false }
            }
        }
        .sheet(isPresented: $showEnrollment) {
            EnrollmentSheet(model: model)
        }
        .sheet(isPresented: $showGalaxy) {
            VoiceGalaxyView(presets: model.presets, enrolled: model.library.voices)
        }
        .sheet(item: $cardVoice) { voice in
            VoiceCardSheet(voice: voice)
        }
        .alert(
            "No voice in that picture", isPresented: $importFailed
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(
                "This works with voice cards made in FrankenTTS. Ask for the card picture itself, uncropped, and try again."
            )
        }
        .onChange(of: importItem) { _, item in
            guard let item else { return }
            importItem = nil
            Task {
                let data = try? await item.loadTransferable(type: Data.self)
                let decoded = await Task.detached(priority: .userInitiated) {
                    data.flatMap { VoicePrintCard.decode($0) }
                }.value
                if let (name, vector) = decoded {
                    // Importing the same card twice selects the existing voice
                    // instead of duplicating it.
                    if let existing = model.library.voices.first(
                        where: { $0.vector == vector }) {
                        model.selectedVoice = "voice:\(existing.id.uuidString)"
                        importCount += 1
                    } else if let voice = try? model.library.add(
                        name: name, vector: vector) {
                        model.selectedVoice = "voice:\(voice.id.uuidString)"
                        importCount += 1
                    } else {
                        importFailed = true
                    }
                } else {
                    importFailed = true
                }
            }
        }
        .alert(
            "Rename voice", isPresented: Binding(
                get: { renameTarget != nil },
                set: { if !$0 { renameTarget = nil } })
        ) {
            TextField("name", text: $renameText)
            Button("Save") {
                if let target = renameTarget {
                    let trimmed = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        try? model.library.rename(id: target.id, to: trimmed)
                    }
                }
                renameTarget = nil
            }
            Button("Cancel", role: .cancel) { renameTarget = nil }
        }
        .onChange(of: renameTarget) { _, target in
            if let target { renameText = target.name }
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
        .sensoryFeedback(.selection, trigger: model.selectedVoice)
        .sensoryFeedback(.success, trigger: model.lastAudio?.count)
        .sensoryFeedback(.success, trigger: model.enrollmentSaved) { _, saved in saved }
        .sensoryFeedback(.success, trigger: importCount) { _, count in count > 0 }
        .onAppear(perform: debugCardHook)
        .onAppear(perform: debugVideoHook)
        .scrollDismissesKeyboard(.interactively)
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
                        model.store.cachedBytes > 0
                            ? "Part of the model is already here; completing the download fetches only what is missing and re-verifies the rest."
                            : "First use downloads the quantized model (≈2.0 GB) into this app's private storage: verified against pinned digests, resumable, kept until you clear it."
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
                    Button(
                        model.store.cachedBytes > 0
                            ? "Complete the download" : "Download the 2.0 GB model"
                    ) { showConsent = true }
                        .buttonStyle(PrimaryButtonStyle())
                        .confirmationDialog(
                            model.store.cachedBytes > 0
                                ? "Fetch the missing files now? Existing files are kept and re-verified, not re-downloaded."
                                : "Download 2.0 GB now? It stays on this device, resumes if interrupted, and Clear Model removes it.",
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
                    ForEach(model.library.voices) { voice in
                        EnrolledVoiceTile(
                            voice: voice,
                            selected: model.enrolledSelection() == voice.id,
                            select: { model.selectedVoice = "voice:\(voice.id.uuidString)" },
                            rename: { renameTarget = voice },
                            reRecord: {
                                model.enrollmentTarget = voice.id
                                showEnrollment = true
                            },
                            share: { cardVoice = voice },
                            delete: {
                                model.library.delete(id: voice.id)
                                if model.enrolledSelection() == voice.id {
                                    model.selectedVoice = "matt"
                                }
                            })
                    }
                    VoiceTile(
                        name: "+ new voice",
                        character: "read a few sentences to clone one",
                        selected: false,
                        accent: true
                    ) {
                        if model.store.phase == .ready {
                            model.enrollmentTarget = nil
                            showEnrollment = true
                        }
                    }
                }
                .animation(.snappy, value: model.library.voices)
                PhotosPicker(
                    selection: $importItem, matching: .images, photoLibrary: .shared()
                ) {
                    HStack(spacing: 8) {
                        Image(systemName: "photo.badge.plus")
                        Text("Add a voice from a picture")
                        Spacer()
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Lab.emerald)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                    .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Lab.emerald.opacity(0.35), lineWidth: 1))
                }
                .accessibilityHint(
                    "Pick a voice card someone sent you; the voice joins your library")
                Text(
                    "Cloning runs the speaker encoder on this phone; the recording is discarded once the 4 KB voice vector exists. A voice card someone sends you works the same way: the picture holds their voiceprint, and importing it never touches the internet. Clone or import only voices you have the right to use."
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
                .focused($focusedField)
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
                    .focused($focusedField)
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
                Button {
                    focusedField = false
                    model.synthesize()
                } label: {
                    HStack(spacing: 8) {
                        if model.isSynthesizing {
                            ProgressView().tint(.white).controlSize(.small)
                        }
                        Text(model.isSynthesizing ? "Synthesizing" : "⚡ Synthesize")
                    }
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
                        Button {
                            if model.player?.isPlaying == true {
                                model.player?.pause()
                            } else {
                                if model.player?.currentTime == model.player?.duration {
                                    model.player?.currentTime = 0
                                }
                                model.player?.play()
                            }
                            playbackTick += 1
                        } label: {
                            Image(
                                systemName: model.player?.isPlaying == true
                                    ? "pause.fill" : "play.fill")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        .accessibilityLabel(
                            model.player?.isPlaying == true ? "Pause" : "Play")
                        .id(playbackTick)
                        .onReceive(
                            Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()
                        ) { _ in
                            playbackTick += 1
                        }
                        if let url = model.m4aUrl ?? model.wavUrl {
                            // M4A once the fast transcode lands; the WAV covers the gap.
                            ShareLink(item: url) {
                                Text(url.pathExtension == "m4a" ? "Share M4A" : "Share…")
                            }
                            .buttonStyle(GhostButtonStyle())
                        }
                        if let video = model.videoUrl {
                            ShareLink(item: video) {
                                Text("Share Video")
                            }
                            .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        } else if model.isExportingVideo {
                            HStack(spacing: 6) {
                                ProgressView().tint(Lab.emerald).controlSize(.small)
                                Text("\(Int(model.videoProgress * 100))%")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Lab.textSecondary)
                            }
                        } else {
                            Button("Make video") { model.exportVideo() }
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
        VStack(spacing: 14) {
            Button {
                showGalaxy = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                    Text("Visualize the voices")
                }
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
            .accessibilityHint("Shows every voice as a shape; similar voices look alike and sit together")
            Text("Runs entirely on this device · frankentts.com")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 6)
    }

    /// Screenshot harness: FTTS_DEBUG_CARD=1 opens the card sheet with a synthetic
    /// voice so the layout is checkable on a model-less simulator. Debug builds only.
    private func debugCardHook() {
        #if DEBUG
            guard ProcessInfo.processInfo.environment["FTTS_DEBUG_CARD"] != nil else {
                return
            }
            var state: UInt64 = 0x0DDB_A11
            let vector: [Float] = (0..<Engine.speakerWidth).map { _ in
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                return Float(Double(state >> 40) / Double(1 << 24) - 0.5) * 3
            }
            cardVoice = EnrolledVoice(id: UUID(), name: "Jeff", vector: vector)
            // Round-trip the REAL composed PNG through both import paths: the raw
            // bytes exercise the lossless chunk; a re-encode via UIImage strips the
            // private chunk, forcing the pixel decoder against the exact pixels
            // ImageRenderer produced (which the off-device harness only approximates).
            Task {
                guard let png = try? VoicePrintCard.pngData(name: "Jeff", vector: vector)
                else {
                    NSLog("FTTS_DEBUG_CARD render FAILED")
                    return
                }
                let chunk = VoicePrintCard.decode(png)
                let chunkOk = chunk?.vector == vector && chunk?.name == "Jeff"
                let stripped = UIImage(data: png)?.pngData()
                let pixels = stripped.flatMap { VoicePrintCard.decode($0) }
                let pixelsOk = pixels?.vector == vector && pixels?.name == "Jeff"
                NSLog("FTTS_DEBUG_CARD roundtrip chunk=\(chunkOk) pixels=\(pixelsOk)")
            }
        #endif
    }

    /// Reproduction harness: FTTS_DEBUG_VIDEO=1 runs a full video export over 20 s of
    /// synthetic tone on launch, logging progress and per-frame timing. Debug only.
    private func debugVideoHook() {
        #if DEBUG
            guard ProcessInfo.processInfo.environment["FTTS_DEBUG_VIDEO"] != nil else {
                return
            }
            let pcm: [Float] = (0..<(24_000 * 20)).map { index in
                sinf(Float(index) * 2 * .pi * 220 / 24_000) * 0.4
            }
            let wav = WavWriter.data(from: pcm)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("ftts-debug-video.wav")
            try? wav.write(to: url)
            let started = Date()
            NSLog("FTTS_DEBUG_VIDEO starting: \(pcm.count) samples")
            Task {
                do {
                    let out = try await MediaExporter.exportVideo(
                        pcm: pcm, voiceLabel: "debug", wavUrl: url
                    ) { fraction in
                        let percent = Int(fraction * 100)
                        if percent % 10 == 0 {
                            NSLog("FTTS_DEBUG_VIDEO %d%% at %.1fs", percent,
                                Date().timeIntervalSince(started))
                        }
                    }
                    NSLog("FTTS_DEBUG_VIDEO done in %.1fs: %@",
                        Date().timeIntervalSince(started), out.path)
                } catch {
                    NSLog("FTTS_DEBUG_VIDEO FAILED: \(error.localizedDescription)")
                }
            }
        #endif
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

/// An enrolled voice: selectable like a preset, with its management row.
struct EnrolledVoiceTile: View {
    let voice: EnrolledVoice
    let selected: Bool
    let select: () -> Void
    let rename: () -> Void
    let reRecord: () -> Void
    let share: () -> Void
    let delete: () -> Void
    @State private var confirmDelete = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: select) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(voice.name)
                        .font(.system(size: 15, weight: .black))
                        .foregroundStyle(Lab.textPrimary)
                        .lineLimit(1)
                    Text("locally cloned")
                        .font(.system(size: 11))
                        .foregroundStyle(Lab.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Four controls must fit the narrowest two-column tile (~146 pt of
            // content width on an SE-class phone): 4 × 32 + 3 × 4 = 140.
            HStack(spacing: 4) {
                Button(action: rename) {
                    Image(systemName: "pencil").frame(width: 32, height: 30)
                }
                .accessibilityLabel("Rename \(voice.name)")
                Button(action: reRecord) {
                    Image(systemName: "arrow.clockwise").frame(width: 32, height: 30)
                }
                .accessibilityLabel("Re-record \(voice.name)")
                Button(action: share) {
                    Image(systemName: "square.and.arrow.up").frame(width: 32, height: 30)
                }
                .accessibilityLabel("Share \(voice.name) as a voice card")
                Spacer()
                Button { confirmDelete = true } label: {
                    Image(systemName: "trash").frame(width: 32, height: 30)
                }
                .foregroundStyle(Lab.danger)
                .accessibilityLabel("Delete \(voice.name)")
                .confirmationDialog(
                    "Delete \"\(voice.name)\"? This cannot be undone.",
                    isPresented: $confirmDelete, titleVisibility: .visible
                ) {
                    Button("Delete", role: .destructive, action: delete)
                    Button("Cancel", role: .cancel) {}
                }
            }
            .font(.system(size: 13))
            .foregroundStyle(Lab.textSecondary)
        }
        .padding(10)
        .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(
                    selected ? Lab.emerald : Lab.emerald.opacity(0.35),
                    lineWidth: selected ? 1.5 : 1))
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
                Text(
                    "Read this aloud. The first few sentences are enough for a good clone; the whole script polishes it slightly."
                )
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
                    // Locked once recording starts: the name is required to save, and
                    // clearing it mid-read would cost the whole take at auto-stop.
                    .disabled(model.isEnrolling || model.recorder.isRecording)
                if model.recorder.isRecording {
                    // The live meter is the tell that the microphone is actually hearing
                    // you; a silent bar during the script means stop and fix it now, not
                    // after a minute of reading.
                    HStack(spacing: 10) {
                        Image(systemName: "waveform")
                            .foregroundStyle(Lab.emerald)
                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.white.opacity(0.06))
                                Capsule()
                                    .fill(Lab.emerald)
                                    .frame(
                                        width: proxy.size.width
                                            * CGFloat(max(0.02, model.recorder.level)))
                                    .animation(.linear(duration: 0.2), value: model.recorder.level)
                            }
                        }
                        .frame(height: 8)
                        Text("\(Int(model.recorder.seconds))s")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Lab.textSecondary)
                    }
                    .accessibilityLabel("Recording level meter")
                    Text("Thirty seconds is plenty; recording stops itself at sixty.")
                        .font(.system(size: 11))
                        .foregroundStyle(Lab.textSecondary)
                }
                if model.isEnrolling {
                    HStack(spacing: 10) {
                        ProgressView().tint(Lab.emerald)
                        Text("computing your voice vector…")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Lab.textSecondary)
                    }
                }
                HStack {
                    if model.recorder.isRecording {
                        Button("⏹ Stop & clone") {
                            model.finishEnrollment(named: cloneName)
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    } else if !model.isEnrolling {
                        Button("🎙 Start recording") {
                            do {
                                try model.recorder.start()
                            } catch {
                                model.lastError = error.localizedDescription
                            }
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        // The name is the save key; requiring it up front beats losing a
                        // recording to a validation error after the read.
                        .disabled(
                            cloneName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    Spacer()
                    Button("Cancel") {
                        _ = model.recorder.stop()
                        model.enrollmentTarget = nil
                        dismiss()
                    }
                    .buttonStyle(GhostButtonStyle())
                    .disabled(model.isEnrolling)
                }
                if let error = model.lastError, !model.recorder.isRecording, !model.isEnrolling {
                    Text(error).font(.system(size: 13)).foregroundStyle(Lab.danger)
                }
            }
            .padding(18)
        }
        .presentationDetents([.large])
        // Recording must also block swipe-dismissal: a dismissed sheet takes its
        // observers with it, and a recorder nobody is watching would keep the
        // microphone live indefinitely.
        .interactiveDismissDisabled(model.isEnrolling || model.recorder.isRecording)
        .onChange(of: model.recorder.seconds) { _, seconds in
            // Backstop: the script reads in about half a minute.
            if seconds >= 60, model.recorder.isRecording {
                model.finishEnrollment(named: cloneName)
            }
        }
        .onChange(of: model.isEnrolling) { was, now in
            // Enrollment just finished; leave the sheet only on success.
            if was, !now, model.enrollmentSaved {
                dismiss()
            }
        }
        .onAppear {
            model.lastError = nil
            model.enrollmentSaved = false
            if let target = model.enrollmentTarget,
                let voice = model.library.voice(id: target)
            {
                cloneName = voice.name
            }
        }
        .onDisappear {
            // Failsafe for any dismissal path: the microphone never outlives the sheet.
            if model.recorder.isRecording {
                _ = model.recorder.stop()
                model.enrollmentTarget = nil
            }
        }
    }
}
