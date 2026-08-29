// The laboratory: one scrolling screen mirroring the site's playground.

import AVFoundation
import CryptoKit
import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

private enum VoiceForgeTextEntry: Hashable {
    case utterance
    case seed
}

private struct VoiceForgeTextEntryFrameKey: PreferenceKey {
    static let defaultValue: [VoiceForgeTextEntry: CGRect] = [:]

    static func reduce(
        value: inout [VoiceForgeTextEntry: CGRect],
        nextValue: () -> [VoiceForgeTextEntry: CGRect]
    ) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

private extension View {
    func reportVoiceForgeTextEntry(_ entry: VoiceForgeTextEntry) -> some View {
        background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: VoiceForgeTextEntryFrameKey.self,
                    value: [entry: proxy.frame(in: .named("voice-forge-text-space"))]
                )
            }
        }
    }
}

/// A deliberately conservative, device-local synthesis forecast.  The first run is
/// seeded from broad memory-tier measurements; completed runs replace that prior with
/// an exponentially weighted measurement from this exact device.  A number is not
/// published until the native engine emits a real semantic frame.
private struct TTSAdaptiveETA {
    private static let version = "v1"
    private static let framesPerWordKey = "eta.\(version).framesPerWord"
    private static let denoiseSecondsKey = "eta.\(version).denoiseSeconds"

    private var words = 1
    private var beganWarm = true
    private var expectedFrames = 0.0
    private var predictedFinishElapsed: TimeInterval?
    private var hasMeasuredWork = false
    private var denoiseStartedAt: TimeInterval?

    mutating func reset(text: String, warm: Bool) {
        words = max(1, text.split(whereSeparator: { $0.isWhitespace }).count)
        beganWarm = warm
        expectedFrames = learnedFramesPerWord * Double(words)
        predictedFinishElapsed = nil
        hasMeasuredWork = false
        denoiseStartedAt = nil
    }

    mutating func observe(_ event: EngineProgress, elapsed: TimeInterval) {
        guard elapsed > 0 else { return }

        let priorTotal = learnedSecondsPerWord * Double(words)
        var fraction: Double?

        if event.kind == .unit, event.stage == .frames, event.current > 0 {
            hasMeasuredWork = true
            let current = Double(event.current)
            let ceiling = event.total > 0 ? Double(event.total) : .greatestFiniteMagnitude
            if expectedFrames <= 0 {
                expectedFrames = current * 1.25
            }
            expectedFrames = min(ceiling, max(expectedFrames, current * 1.08))
            let frameFraction = min(0.985, current / max(current, expectedFrames))
            fraction = 0.10 + 0.80 * frameFraction
        } else if event.kind == .unit, event.stage == .codec, event.current > 0 {
            hasMeasuredWork = true
            let current = Double(event.current)
            if expectedFrames <= 0 {
                expectedFrames = max(current * 1.08, learnedFramesPerWord * Double(words))
            }
            let codecFraction = min(0.985, current / max(current, expectedFrames))
            fraction = 0.90 + 0.085 * codecFraction
        } else if event.kind == .stageFinished, event.stage == .synthesis {
            hasMeasuredWork = true
            fraction = 0.985
        }

        guard let fraction else { return }
        let observedTotal = elapsed / max(0.04, fraction)
        // Real progress earns most of the vote quickly, while the learned device prior
        // prevents the first few frames from producing a wildly unstable forecast.
        let observedWeight = min(0.88, 0.38 + fraction * 0.52)
        let candidate = max(elapsed + 0.75, priorTotal * (1 - observedWeight) + observedTotal * observedWeight)
        if let old = predictedFinishElapsed {
            let smoothed = old * 0.68 + candidate * 0.32
            // Permit honest upward revision, but prevent a single noisy callback from
            // making the displayed countdown jump by tens of seconds.
            predictedFinishElapsed = min(smoothed, old + 3.0)
        } else {
            predictedFinishElapsed = candidate
        }
    }

    mutating func beginDenoise(elapsed: TimeInterval) {
        denoiseStartedAt = elapsed
        let learned = UserDefaults.standard.double(forKey: Self.denoiseSecondsKey)
        let tail = learned > 0 ? learned : 1.5
        predictedFinishElapsed = max(predictedFinishElapsed ?? elapsed, elapsed + tail)
    }

    func remainingSeconds(at elapsed: TimeInterval) -> Int? {
        guard hasMeasuredWork, let predictedFinishElapsed else { return nil }
        return max(1, Int(ceil(predictedFinishElapsed - elapsed)))
    }

    mutating func finish(elapsed: TimeInterval, frames: UInt64) {
        let defaults = UserDefaults.standard
        let secondsSample = elapsed / Double(words)
        Self.update(
            key: secondsPerWordKey,
            sample: secondsSample,
            in: defaults
        )
        if frames > 0 {
            Self.update(
                key: Self.framesPerWordKey,
                sample: Double(frames) / Double(words),
                in: defaults
            )
        }
        if let denoiseStartedAt, elapsed > denoiseStartedAt {
            Self.update(
                key: Self.denoiseSecondsKey,
                sample: elapsed - denoiseStartedAt,
                in: defaults
            )
        }
        predictedFinishElapsed = elapsed
    }

    private var secondsPerWordKey: String {
        "eta.\(Self.version).secondsPerWord.\(beganWarm ? "warm" : "cold")"
    }

    private var learnedSecondsPerWord: Double {
        let learned = UserDefaults.standard.double(forKey: secondsPerWordKey)
        if learned > 0 { return learned }

        let gib = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let warmSeed: Double
        switch gib {
        case 8...: warmSeed = 1.05
        case 6..<8: warmSeed = 1.25
        default: warmSeed = 1.65
        }
        return warmSeed + (beganWarm ? 0 : 0.28)
    }

    private var learnedFramesPerWord: Double {
        let learned = UserDefaults.standard.double(forKey: Self.framesPerWordKey)
        return learned > 0 ? learned : 4.6
    }

    private static func update(key: String, sample: Double, in defaults: UserDefaults) {
        guard sample.isFinite, sample > 0 else { return }
        let old = defaults.double(forKey: key)
        defaults.set(old > 0 ? old * 0.72 + sample * 0.28 : sample, forKey: key)
    }
}

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
        "The rainbow is a division of white light into many beautiful colors. Now, spoken entirely on this device."
    var seed: UInt64 = 0

    var isSynthesizing = false
    var isLoadingModel = false
    var isEngineWarm = false
    var synthesisSeconds = 0.0
    var estimatedRemainingSeconds: Int?
    var lastError: String?
    var lastAudio: [Float]?
    var lastRealTimeFactor: Double?
    var lastProfile: SynthesisProfile?
    var forge = VoiceForgeTelemetry()
    var nativeProgressEvents: [EngineProgress] = []
    var player: AVAudioPlayer?
    /// The playback WAV (internal); shares go out as M4A and MP4.
    var wavUrl: URL?
    var m4aUrl: URL?
    var videoUrl: URL?
    var isExportingVideo = false
    var videoProgress = 0.0
    /// Bumped per synthesis so a slow export cannot stamp its output onto a newer clip.
    private var synthesisGeneration = 0
    private var engineWarmTask: Task<Void, Never>?
    private let activity = VoiceForgeActivityController.shared
    private var eta = TTSAdaptiveETA()

    var lowMemoryDevice: Bool {
        ProcessInfo.processInfo.physicalMemory < 6 * 1024 * 1024 * 1024
    }

    var canSynthesizeFromCommand: Bool {
        !isSynthesizing
            && !isLoadingModel
            && store.phase == .ready
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
        guard canSynthesizeFromCommand else { return }
        isSynthesizing = true
        lastError = nil
        lastProfile = nil
        synthesisSeconds = 0
        estimatedRemainingSeconds = nil
        nativeProgressEvents.removeAll(keepingCapacity: true)
        let beganWarm = isEngineWarm
        forge.reset(for: beganWarm ? .checkingMemory : .readingBundle)
        eta.reset(text: text, warm: beganWarm)
        activity.begin()
        let text = self.text
        let seed = self.seed
        let runStartedAt = Date()
        let ticker = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.synthesisSeconds = Date().timeIntervalSince(runStartedAt)
                self.estimatedRemainingSeconds = self.eta.remainingSeconds(
                    at: self.synthesisSeconds
                )
            }
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
                    isLoadingModel = true
                    try await engine.load(modelDirectory: store.modelDirectory) { [weak self] event in
                        Task { @MainActor in self?.receive(event) }
                    }
                }
                isEngineWarm = true
                isLoadingModel = false
                let started = Date()
                let output = try await engine.synthesize(
                    text: text,
                    speaker: speaker,
                    seed: seed
                ) { [weak self] event in
                    Task { @MainActor in self?.receive(event) }
                }
                var pcm = output.pcm
                lastProfile = output.profile
                let elapsed = Date().timeIntervalSince(started)
                let factor = (Double(pcm.count) / Double(WavWriter.sampleRate)) / elapsed
                lastRealTimeFactor = factor
                UserDefaults.standard.set(factor, forKey: "measuredRealTimeFactor")
                // The same neural denoiser enrollment uses, run over the OUTPUT: it
                // strips residual hiss (especially audible with cloned voices) and a
                // failure just keeps the original audio.
                if await engine.denoiseAvailable {
                    forge.phase = .denoising
                    eta.beginDenoise(elapsed: Date().timeIntervalSince(runStartedAt))
                    pcm = (try? await engine.denoise(pcm: pcm)) ?? pcm
                }
                synthesisSeconds = Date().timeIntervalSince(runStartedAt)
                eta.finish(
                    elapsed: synthesisSeconds,
                    frames: output.profile?.frames ?? forge.generatedFrames
                )
                estimatedRemainingSeconds = nil
                lastAudio = pcm
                try startPlayback(of: pcm)
                forge.phase = .complete
                activity.finish(
                    status: .complete,
                    headline: "Voice alive",
                    detail: "Your private on-device audio is ready"
                )
            } catch EngineError.cancelled {
                forge.phase = .cancelled
                activity.finish(
                    status: .cancelled,
                    headline: "Forge stopped",
                    detail: "No partial audio was published"
                )
            } catch {
                isEngineWarm = await engine.isLoaded
                forge.phase = .failed
                lastError = error.localizedDescription
                activity.finish(
                    status: .failed,
                    headline: "Voice Forge needs attention",
                    detail: "Open FrankenTTS to retry"
                )
            }
            isLoadingModel = false
            isSynthesizing = false
            estimatedRemainingSeconds = nil
        }
    }

    /// Physical-device profiling lane. This is environment-triggered and absent
    /// from the product UI: it measures the app's exact FFI route repeatedly in
    /// an optimized app build without adding benchmark controls to the interface.
    func runProfilingBenchmarkIfRequested() async {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FTTS_IOS_PROFILE"] == "1" else { return }

        let requestedRuns = Int(environment["FTTS_IOS_PROFILE_RUNS"] ?? "20") ?? 20
        let runs = max(1, min(100, requestedRuns))
        let benchmarkText = environment["FTTS_IOS_PROFILE_TEXT"]
            ?? "Frankenstein listened carefully while the rain tapped softly against the laboratory window."
        let voice = environment["FTTS_IOS_PROFILE_VOICE"] ?? "matt"
        let benchmarkSeed = UInt64(environment["FTTS_IOS_PROFILE_SEED"] ?? "0") ?? 0

        let documents = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        )[0]
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let receiptURL = documents.appendingPathComponent(
            "ftts-ios-profile-\(stamp).jsonl")
        var receiptLines: [String] = []

        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }

        func appendReceipt(_ object: [String: Any]) throws {
            let data = try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
            guard let line = String(data: data, encoding: .utf8) else {
                throw EngineError.native("profiling receipt was not UTF-8")
            }
            receiptLines.append(line)
            try Data((receiptLines.joined(separator: "\n") + "\n").utf8)
                .write(to: receiptURL, options: .atomic)
            print("FTTS_IOS_PROFILE \(line)")
        }

        do {
            guard store.isComplete else {
                throw EngineError.native("profiling requires the complete downloaded model")
            }
            let speaker = try Engine.presetVector(named: voice)
            try appendReceipt([
                "event": "run_start",
                "schema_version": 1,
                "runs": runs,
                "text": benchmarkText,
                "voice": voice,
                "seed": benchmarkSeed,
                "device_model": UIDevice.current.model,
                "system_name": UIDevice.current.systemName,
                "system_version": UIDevice.current.systemVersion,
                "active_processors": ProcessInfo.processInfo.activeProcessorCount,
                "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "team_threads": environment["FTTS_INT8_THREADS"] ?? "unset",
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                "receipt_path": receiptURL.path,
                "model_manifest": ModelManifest.files.map { file in
                    [
                        "asset": file.asset,
                        "relative_path": file.relativePath,
                        "bytes": file.bytes,
                        "sha256": file.sha256,
                    ] as [String: Any]
                },
            ])

            let loadStarted = Date()
            try await engine.load(modelDirectory: store.modelDirectory)
            try appendReceipt([
                "event": "engine_loaded",
                "load_ms": Date().timeIntervalSince(loadStarted) * 1_000,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])

            var firstDigest: String?
            var validRuns = 0
            var allAudioIdentical = true
            for index in 0..<runs {
                let started = Date()
                let output = try await engine.synthesize(
                    text: benchmarkText,
                    speaker: speaker,
                    seed: benchmarkSeed
                )
                let wallMs = Date().timeIntervalSince(started) * 1_000
                let audioSeconds = Double(output.pcm.count) / Double(WavWriter.sampleRate)
                let realtimeSpeed = audioSeconds / max(wallMs / 1_000, 0.000_001)
                let wav = WavWriter.data(from: output.pcm)
                let digest = SHA256.hash(data: wav)
                    .map { String(format: "%02x", $0) }.joined()
                if firstDigest == nil { firstDigest = digest }
                guard let profile = output.profile else {
                    throw EngineError.native("native synthesis profile was unavailable")
                }
                let matchesFirstAudio = digest == firstDigest
                allAudioIdentical = allAudioIdentical && matchesFirstAudio
                try appendReceipt([
                    "event": "sample",
                    "index": index,
                    "wall_ms": wallMs,
                    "audio_seconds": audioSeconds,
                    "realtime_speed": realtimeSpeed,
                    "wav_sha256": digest,
                    "matches_first_audio": matchesFirstAudio,
                    "total_ms": profile.totalMs,
                    "generation_ms": profile.generationMs,
                    "prefill_ms": profile.prefillMs,
                    "talker_ms": profile.talkerMs,
                    "microdecoder_ms": profile.microdecoderMs,
                    "feedback_ms": profile.feedbackMs,
                    "codec_active_ms": profile.codecActiveMs,
                    "other_generation_ms": profile.otherGenerationMs,
                    "frames": profile.frames,
                    "team_partitions": profile.teamPartitions,
                    "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                ])
                validRuns += 1
            }

            try appendReceipt([
                "event": "run_complete",
                "completed_runs": validRuns,
                "all_audio_identical": validRuns == runs && allAudioIdentical,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        } catch {
            try? appendReceipt([
                "event": "run_error",
                "message": error.localizedDescription,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        }
    }

    func cancelSynthesis() {
        guard isSynthesizing else { return }
        forge.phase = .cancelling
        engine.cancelCurrentWork()
    }

    private func receive(_ event: EngineProgress) {
        forge.apply(event)
        eta.observe(event, elapsed: synthesisSeconds)
        estimatedRemainingSeconds = eta.remainingSeconds(at: synthesisSeconds)
        activity.update(from: forge, elapsed: synthesisSeconds)
        nativeProgressEvents.append(event)
        if nativeProgressEvents.count > 160 {
            nativeProgressEvents.removeFirst(nativeProgressEvents.count - 160)
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

    func togglePlayback() {
        guard let player else { return }
        if player.isPlaying { player.pause() } else { player.play() }
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
                // The denoiser is not optional: a profile built from un-denoised audio
                // carries the recording's noise into every synthesis. Its absence
                // means the model download is incomplete — refuse and say so.
                guard await engine.denoiseAvailable else {
                    throw EngineError.native(
                        "the noise-removal file is missing; it downloads automatically — check the connection, relaunch, and try again")
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
        engine.cancelCurrentWork()
        engineWarmTask?.cancel()
        engineWarmTask = nil
        isEngineWarm = false
        isLoadingModel = false
        Task { await engine.unload() }
    }

    /// Hydrate as soon as verified model files are present. Users should wait
    /// for the model once at app/foreground entry, not discover the same cold
    /// start only after pressing Synthesize.
    func warmEngineIfPossible() {
        guard store.phase == .ready,
              !isEngineWarm,
              engineWarmTask == nil,
              !isSynthesizing
        else { return }

        isLoadingModel = true
        engineWarmTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.isLoadingModel = false
                self.engineWarmTask = nil
            }
            do {
                if await !self.engine.isLoaded {
                    self.forge.reset(for: .readingBundle)
                    try await self.engine.load(modelDirectory: self.store.modelDirectory) { [weak self] event in
                        Task { @MainActor in self?.receive(event) }
                    }
                }
                try Task.checkCancellation()
                self.isEngineWarm = true
                if !self.isSynthesizing { self.forge.phase = .idle }
            } catch EngineError.cancelled {
                self.isEngineWarm = false
                self.forge.phase = .cancelled
            } catch is CancellationError {
                self.isEngineWarm = false
                self.forge.phase = .cancelled
            } catch {
                self.isEngineWarm = false
                self.forge.phase = .failed
                self.lastError = "Could not warm the model: \(error.localizedDescription)"
            }
        }
    }

    func clearModel() {
        unloadEngineForMemoryPressure()
        store.clear()
    }

    func importDesktopFile(_ url: URL) {
        let supportedText = ["txt", "md", "markdown"]
        let ext = url.pathExtension.lowercased()
        let scoped = url.startAccessingSecurityScopedResource()
        Task {
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            do {
                let data = try await Task.detached(priority: .userInitiated) {
                    try Data(contentsOf: url, options: .mappedIfSafe)
                }.value
                if supportedText.contains(ext) {
                    guard let imported = String(data: data, encoding: .utf8) else {
                        throw EngineError.native("that text file is not UTF-8")
                    }
                    text = String(imported.prefix(600))
                    return
                }
                guard let (name, vector) = await Task.detached(priority: .userInitiated, operation: {
                    VoicePrintCard.decode(data)
                }).value else {
                    throw EngineError.native("that image does not contain a FrankenTTS voice card")
                }
                if let existing = library.voices.first(where: { $0.vector == vector }) {
                    selectedVoice = "voice:\(existing.id.uuidString)"
                } else {
                    let voice = try library.add(name: name, vector: vector)
                    selectedVoice = "voice:\(voice.id.uuidString)"
                }
            } catch {
                lastError = error.localizedDescription
            }
        }
    }
}

struct LabView: View {
    private enum EditorFocus: Hashable {
        case utterance
        case seed
    }

    @State private var model = LabModel()
    @State private var showEnrollment = false
    @State private var showGalaxy = false
    @State private var showSpecimen = false
    @State private var showVoiceLab = false
    @State private var renameTarget: EnrolledVoice?
    @State private var renameText = ""
    @State private var cardVoice: EnrolledVoice?
    @State private var importItem: PhotosPickerItem?
    @State private var importFailed = false
    @State private var importCount = 0
    @State private var showDesktopImporter = false
    @State private var textEntryFrames: [VoiceForgeTextEntry: CGRect] = [:]
    /// Bumped to refresh the play/pause icon, which tracks external playback state.
    @State private var playbackTick = 0
    @State private var showMachineProfile = false
    @Environment(\.scenePhase) private var scenePhase
    @FocusState private var focusedField: EditorFocus?

    /// The profiling lane owns engine hydration while it is active. Suppressing
    /// the ordinary foreground/store-ready warm hooks avoids racing two opens of
    /// the same multi-gigabyte engine and invalidating a physical-device sample.
    private var profilingRequested: Bool {
        ProcessInfo.processInfo.environment["FTTS_IOS_PROFILE"] == "1"
    }

    var body: some View {
        systemIntegrationView
    }

    private var workspaceView: some View {
        GeometryReader { geometry in
            ZStack {
                LaboratoryBackground()
                if usesDashboardLayout, geometry.size.width >= 680 {
                    HStack(alignment: .top, spacing: 18) {
                        VStack(alignment: .leading, spacing: 12) {
                            header
                            modelEntryView
                            compactVoiceSelector(vertical: true)
                            Spacer(minLength: 4)
                            footer
                        }
                        .frame(width: min(310, geometry.size.width * 0.34))
                        .frame(maxHeight: .infinity, alignment: .top)

                        ScrollView(.vertical) {
                            utteranceCard(compact: true)
                                .frame(maxWidth: .infinity, alignment: .topLeading)
                        }
                        .scrollIndicators(.hidden)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 14)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 18) {
                            header
                            modelEntryView
                            compactVoiceSelector(vertical: false)
                            utteranceCard(compact: false)
                            footer
                        }
                        .frame(maxWidth: 760)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 16)
                }
            }
            .catalystReadableType()
        }
    }

    private var usesDashboardLayout: Bool {
#if targetEnvironment(macCatalyst)
        true
#else
        UIDevice.current.userInterfaceIdiom == .pad
#endif
    }

    private var keyboardAwareView: some View {
        workspaceView
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { focusedField = nil }
            }
        }
        .coordinateSpace(name: "voice-forge-text-space")
        .onPreferenceChange(VoiceForgeTextEntryFrameKey.self) { frames in
            textEntryFrames = frames
        }
        .simultaneousGesture(
            SpatialTapGesture(coordinateSpace: .named("voice-forge-text-space"))
                .onEnded { tap in
                    guard focusedField != nil else { return }
                    let tappedEditor = textEntryFrames.values.contains { $0.contains(tap.location) }
                    if !tappedEditor { focusedField = nil }
                }
        )
    }

    private var sheetView: some View {
        keyboardAwareView
        .sheet(isPresented: $showEnrollment) {
            EnrollmentSheet(model: model)
        }
        .sheet(isPresented: $showSpecimen) {
            NavigationStack {
                ScrollView { specimenCard.padding(18) }
                    .background(LaboratoryBackground())
                    .navigationTitle("Model & storage")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showSpecimen = false }
                        }
                    }
            }
            .preferredColorScheme(.dark)
            .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showVoiceLab) {
            NavigationStack {
                ScrollView { voicesCard.padding(18) }
                    .background(LaboratoryBackground())
                    .navigationTitle("Voice laboratory")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showVoiceLab = false }
                        }
                    }
            }
            .preferredColorScheme(.dark)
            .presentationDetents([.large])
        }
        .sheet(isPresented: $showGalaxy) {
            VoiceGalaxyView(presets: model.presets, enrolled: model.library.voices)
        }
        .sheet(item: $cardVoice) { voice in
            VoiceCardSheet(voice: voice)
        }
    }

    private var importView: some View {
        sheetView
        .fileImporter(
            isPresented: $showDesktopImporter,
            allowedContentTypes: [.plainText, .image],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { model.importDesktopFile(url) }
            case .failure(let error):
                model.lastError = error.localizedDescription
            }
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
    }

    private var voiceManagementView: some View {
        importView
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
    }

    private var lifecycleView: some View {
        voiceManagementView
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIApplication.didReceiveMemoryWarningNotification)
        ) { _ in
            model.unloadEngineForMemoryPressure()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background {
                model.unloadEngineForMemoryPressure()
            } else if phase == .active {
                if !profilingRequested { model.warmEngineIfPossible() }
                consumeStagedText()
            }
        }
        .onChange(of: model.store.phase) { _, phase in
            if phase == .ready, !profilingRequested {
                model.warmEngineIfPossible()
            }
        }
        .sensoryFeedback(.selection, trigger: model.selectedVoice)
        .sensoryFeedback(.success, trigger: model.lastAudio?.count)
        .sensoryFeedback(.success, trigger: model.enrollmentSaved) { _, saved in saved }
        .sensoryFeedback(.success, trigger: importCount) { _, count in count > 0 }
        .onAppear(perform: debugCardHook)
        .onAppear(perform: debugVideoHook)
        .task {
            if profilingRequested {
                await model.runProfilingBenchmarkIfRequested()
            } else {
                model.warmEngineIfPossible()
            }
        }
        .task { consumeStagedText() }
    }

    private var systemIntegrationView: some View {
        lifecycleView
        .onOpenURL(perform: handleDeepLink)
        .onAppear {
            #if DEBUG
                // Screenshot harness: FTTS_DEBUG_GALAXY=1 opens the constellation.
                if ProcessInfo.processInfo.environment["FTTS_DEBUG_GALAXY"] != nil {
                    showGalaxy = true
                }
            #endif
        }
        .scrollDismissesKeyboard(.interactively)
        .userActivity("com.frankentts.voice-forge") { activity in
            activity.title = "FrankenTTS Voice Forge"
            activity.isEligibleForHandoff = true
            activity.userInfo = ["route": "forge"]
        }
        .onContinueUserActivity("com.frankentts.voice-forge") { _ in
            consumeStagedText()
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            model.importDesktopFile(url)
            return true
        }
        .focusedSceneValue(
            \.voiceForgeCommands,
            VoiceForgeCommandActions(
                importFile: { showDesktopImporter = true },
                synthesize: {
                    focusedField = nil
                    model.synthesize()
                },
                stop: { model.cancelSynthesis() },
                togglePlayback: {
                    model.togglePlayback()
                    playbackTick += 1
                },
                canSynthesize: model.canSynthesizeFromCommand,
                canStop: model.isSynthesizing,
                canTogglePlayback: model.player != nil && focusedField == nil
            )
        )
    }

    private func consumeStagedText() {
        guard let staged = FrankenTTSSharedStore.consumeStagedText(), !staged.isEmpty else { return }
        model.text = String(staged.prefix(600))
        focusedField = .utterance
    }

    private func handleDeepLink(_ url: URL) {
        guard url.scheme == "frankentts" else { return }
        if url.host == "cancel" { model.cancelSynthesis() }
        consumeStagedText()
    }

    private var header: some View {
        HStack(spacing: 12) {
            MonsterStatusMark(mood: monsterMood, instrument: .voice)
                .frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 2) {
                Text("FrankenTTS")
                    .font(.system(size: Lab.typeSize(22), weight: .black))
                    .foregroundStyle(Lab.textPrimary)
                Text("VOICE_ALIVE")
                    .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                    .kerning(2)
                    .foregroundStyle(Lab.emerald)
            }
            Spacer()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("FrankenTTS, the monster voice engine")
    }

    private var monsterMood: MonsterMood {
        if model.lastError != nil { return .error }
        if model.isSynthesizing { return .working }
        if model.isLoadingModel { return .waking }
        if model.lastAudio != nil { return .success }
        return .idle
    }

    private var modelStatusButton: some View {
        Button { showSpecimen = true } label: {
            StatusCapsule(
                title: modelStatusTitle,
                detail: modelStatusDetail,
                systemImage: modelStatusImage,
                tint: model.store.phase == .ready ? Lab.emerald : Lab.amber
            )
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens model download, storage, and readiness details")
    }

    @ViewBuilder
    private var modelEntryView: some View {
        switch model.store.phase {
        case .ready:
            modelStatusButton
        case .idle, .downloading, .verifying, .failed:
            specimenCard
        }
    }

    private var modelStatusTitle: String {
        switch model.store.phase {
        case .ready:
            model.isEngineWarm ? "Voice core warm" : (model.isLoadingModel ? "Voice core waking" : "Model on device")
        case .downloading: "Downloading the local model"
        case .verifying: "Verifying the local model"
        case .failed: "Model needs attention"
        case .idle: "Model required"
        }
    }

    private var modelStatusDetail: String {
        switch model.store.phase {
        case .ready:
            model.isEngineWarm ? "Private · ready to synthesize" : "Private · warming automatically"
        case .downloading(_, let done, let total, _):
            "\(Self.gigabytes(done)) of \(Self.gigabytes(total)) GB"
        case .verifying(let asset): "Checking \(asset)"
        case .failed: "Tap to inspect and retry"
        case .idle: "One-time 2.0 GB download"
        }
    }

    private var modelStatusImage: String {
        switch model.store.phase {
        case .ready: model.isEngineWarm ? "bolt.fill" : "brain.head.profile"
        case .downloading: "arrow.down.circle"
        case .verifying: "checkmark.shield"
        case .failed: "exclamationmark.triangle"
        case .idle: "externaldrive.badge.plus"
        }
    }

    @ViewBuilder
    private func compactVoiceSelector(vertical: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                LabLabel(text: "Voice specimen")
                Spacer()
                Button("Manage") { showVoiceLab = true }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Lab.emerald)
                    .buttonStyle(.plain)
            }

            if vertical {
                Button { showVoiceLab = true } label: {
                    HStack(spacing: 12) {
                        VoiceOrb(name: model.currentVoiceLabel, selected: true)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(model.currentVoiceLabel)
                                .font(.headline)
                                .foregroundStyle(Lab.textPrimary)
                            Text("Selected voice · tap to open the library")
                                .font(.caption)
                                .foregroundStyle(Lab.textSecondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(Lab.textSecondary)
                    }
                    .padding(12)
                    .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(Lab.emerald.opacity(0.28), lineWidth: 1)
                    }
                }
                .buttonStyle(.plain)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 9) {
                        ForEach(model.presets) { preset in
                            CompactVoiceChip(
                                name: preset.name,
                                selected: model.selectedVoice == preset.name
                            ) { model.selectedVoice = preset.name }
                        }
                        ForEach(model.library.voices) { voice in
                            CompactVoiceChip(
                                name: voice.name,
                                selected: model.enrolledSelection() == voice.id,
                                isPersonal: true
                            ) { model.selectedVoice = "voice:\(voice.id.uuidString)" }
                        }
                        Button { openEnrollment(target: nil) } label: {
                            Label("Clone", systemImage: "waveform.badge.plus")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(Lab.emerald)
                                .padding(.horizontal, 13)
                                .frame(height: 42)
                                .background(Lab.emerald.opacity(0.08), in: Capsule())
                                .overlay(Capsule().strokeBorder(Lab.emerald.opacity(0.25), lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.vertical, 2)
                }
                .scrollClipDisabled()
            }
        }
    }

    private var specimenCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 10) {
                    ZStack {
                        Circle()
                            .fill(Lab.emerald.opacity(0.12))
                        Image(systemName: model.store.cachedBytes > 0
                              ? "arrow.clockwise.circle.fill" : "brain.head.profile.fill")
                            .font(.system(size: Lab.typeSize(19), weight: .bold))
                            .foregroundStyle(Lab.emerald)
                    }
                    .frame(width: 40, height: 40)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.store.cachedBytes > 0 ? "FINISH VOICE SETUP" : "BRING THE VOICE CORE TO LIFE")
                            .font(.system(size: Lab.typeSize(12), weight: .black, design: .monospaced))
                            .kerning(0.7)
                            .foregroundStyle(Lab.textPrimary)
                        Text("One setup · then everything runs privately here")
                            .font(.system(size: Lab.typeSize(11)))
                            .foregroundStyle(Lab.textSecondary)
                    }
                }

                switch model.store.phase {
                case .idle:
                    Text(
                        model.store.cachedBytes > 0
                            ? "Your saved progress is intact. Resume the remaining static model files—completed pieces are never downloaded twice."
                            : "FrankenTTS needs a one-time 2.0 GB voice model. It downloads from the FrankenTTS GitHub release, is cryptographically verified, and stays in this app's private storage."
                    )
                    .font(.system(size: Lab.typeSize(14)))
                    .foregroundStyle(Lab.textSecondary)

                    HStack(spacing: 12) {
                        ModelPromise(systemImage: "lock.shield", text: "No data sent")
                        ModelPromise(systemImage: "arrow.clockwise", text: "Resumable")
                        ModelPromise(systemImage: "checkmark.seal", text: "Verified")
                    }

                    if model.lowMemoryDevice {
                        Text(
                            "This device reports under 6 GB of memory; the engine may not fit. A recent device with at least 6 GB is recommended."
                        )
                        .font(.system(size: Lab.typeSize(13)))
                        .foregroundStyle(Lab.danger)
                    }
                    Button(
                        model.store.cachedBytes > 0
                            ? "Resume setup" : "Download & set up"
                    ) { model.store.startDownload() }
                        .buttonStyle(PrimaryButtonStyle())
                case .downloading(let asset, let done, let total, let eta):
                    HStack(spacing: 14) {
                        ZStack {
                            Circle()
                                .stroke(Lab.emerald.opacity(0.15), lineWidth: 6)
                            Circle()
                                .trim(from: 0, to: total > 0 ? Double(done) / Double(total) : 0)
                                .stroke(
                                    Lab.emerald,
                                    style: StrokeStyle(lineWidth: 6, lineCap: .round)
                                )
                                .rotationEffect(.degrees(-90))
                                .animation(.smooth, value: done)
                            Text("\(Int((total > 0 ? Double(done) / Double(total) : 0) * 100))%")
                                .font(.system(size: Lab.typeSize(11), weight: .bold, design: .monospaced))
                                .foregroundStyle(Lab.textPrimary)
                        }
                        .frame(width: 58, height: 58)

                        VStack(alignment: .leading, spacing: 4) {
                            Text(asset)
                                .font(.system(size: Lab.typeSize(15), weight: .bold))
                                .foregroundStyle(Lab.textPrimary)
                            Text("Part \(max(1, model.store.currentFileIndex)) of \(model.store.currentFileCount)")
                                .font(.system(size: Lab.typeSize(10), weight: .bold, design: .monospaced))
                                .foregroundStyle(Lab.emerald)
                            Text("\(Self.gigabytes(done)) of \(Self.gigabytes(total)) GB · \(eta)")
                                .font(.system(size: Lab.typeSize(11)))
                                .foregroundStyle(Lab.textSecondary)
                            if model.store.downloadRateBytesPerSecond > 0 {
                                Text(Self.downloadRate(model.store.downloadRateBytesPerSecond))
                                    .font(.system(size: Lab.typeSize(10), design: .monospaced))
                                    .foregroundStyle(Lab.textSecondary)
                            }
                        }
                    }
                    ProgressView(value: Double(done), total: Double(total))
                        .tint(Lab.emerald)
                    HStack {
                        Label("Progress is saved automatically", systemImage: "externaldrive.fill.badge.checkmark")
                            .font(.system(size: Lab.typeSize(10)))
                            .foregroundStyle(Lab.textSecondary)
                        Spacer()
                        Button("Pause") { model.store.pauseDownload() }
                            .buttonStyle(GhostButtonStyle())
                    }
                case .verifying(let asset):
                    HStack(spacing: 12) {
                        ProgressView().tint(Lab.emerald).controlSize(.regular)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Securing \(asset)")
                                .font(.system(size: Lab.typeSize(14), weight: .bold))
                                .foregroundStyle(Lab.textPrimary)
                            Text("Checking every byte before the engine can use it")
                                .font(.system(size: Lab.typeSize(11)))
                                .foregroundStyle(Lab.textSecondary)
                        }
                    }
                case .ready:
                    VStack(alignment: .leading, spacing: 9) {
                        HStack {
                            Image(systemName: "checkmark.seal.fill").foregroundStyle(Lab.emerald)
                            Text("Model on device · \(Self.gigabytes(model.store.cachedBytes)) GB")
                                .font(.system(size: Lab.typeSize(13), design: .monospaced))
                                .foregroundStyle(Lab.textPrimary)
                            Spacer()
                            Button("Clear") { model.clearModel() }
                                .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                        }
                        HStack(spacing: 8) {
                            if model.isLoadingModel {
                                ProgressView().tint(Lab.emerald).controlSize(.small)
                            } else {
                                Image(systemName: model.isEngineWarm
                                      ? "bolt.circle.fill" : "bolt.circle")
                                    .foregroundStyle(model.isEngineWarm
                                                     ? Lab.emerald : Lab.textSecondary)
                            }
                            Text(
                                model.isLoadingModel
                                    ? "Warming automatically…"
                                    : (model.isEngineWarm
                                       ? "Engine warm · ready to synthesize"
                                       : "Engine will warm automatically")
                            )
                            .font(.system(size: Lab.typeSize(11), design: .monospaced))
                            .foregroundStyle(model.isEngineWarm
                                             ? Lab.emerald : Lab.textSecondary)
                        }
                    }
                case .failed(let message):
                    Label("Setup paused", systemImage: "exclamationmark.triangle.fill")
                        .font(.system(size: Lab.typeSize(14), weight: .bold))
                        .foregroundStyle(Lab.danger)
                    Text(message)
                        .font(.system(size: Lab.typeSize(13)))
                        .foregroundStyle(Lab.textSecondary)
                    Button(model.store.cachedBytes > 0 ? "Resume setup" : "Try again") {
                        model.store.startDownload()
                    }
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
                                openEnrollment(target: voice.id)
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
                            openEnrollment(target: nil)
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
                    .font(.system(size: Lab.typeSize(14), weight: .semibold))
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
                    "Cloning runs the speaker encoder on this device; the recording is discarded once the 4 KB voice vector exists. A voice card someone sends you works the same way: the picture holds their voiceprint, and importing it never touches the internet. Clone or import only voices you have the right to use."
                )
                .font(.system(size: Lab.typeSize(12)))
                .foregroundStyle(Lab.textSecondary)
            }
        }
    }

    private func utteranceCard(compact: Bool) -> some View {
        LabPanel {
            VStack(alignment: .leading, spacing: compact ? 9 : 12) {
                HStack(spacing: 8) {
                    LabLabel(text: "03 · The Utterance")
                    Spacer()
                    Button("Select all") { selectAllUtterance() }
                        .buttonStyle(GhostButtonStyle())
                        .disabled(model.text.isEmpty)
                        .accessibilityHint("Selects the entire utterance so typing replaces it")
                    Button {
                        model.text = ""
                        focusedField = .utterance
                    } label: {
                        Label("Clear", systemImage: "xmark.circle.fill")
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    .disabled(model.text.isEmpty)
                    .accessibilityHint("Removes the current utterance")
                }

                ZStack(alignment: .topLeading) {
                    if model.text.isEmpty {
                        Text("Type or paste what FrankenTTS should say…")
                            .font(.system(size: Lab.typeSize(16)))
                            .foregroundStyle(Lab.textSecondary.opacity(0.72))
                            .padding(.horizontal, 13)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: Binding(
                        get: { model.text },
                        set: { model.text = String($0.prefix(600)) }
                    ))
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .foregroundStyle(Lab.textPrimary)
                    .font(.system(size: Lab.typeSize(16)))
                    .lineSpacing(4)
                    .focused($focusedField, equals: .utterance)
                    .reportVoiceForgeTextEntry(.utterance)
                    .textInputAutocapitalization(.sentences)
                    .autocorrectionDisabled(false)
                }
                // More vertical room makes drag handles and the magnifier
                // practical on a phone instead of fighting a three-line box.
                .frame(
                    minHeight: compact ? 104 : 138,
                    maxHeight: compact ? 132 : 190
                )
                .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(
                            focusedField == .utterance
                                ? Lab.emerald.opacity(0.62) : Color.clear,
                            lineWidth: 1
                        )
                )
                HStack {
                    Text("\(model.text.count) / 600")
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    Spacer()
                    Text("seed")
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    TextField(
                        "0",
                        text: Binding(
                            get: { String(model.seed) },
                            set: { model.seed = UInt64($0.filter(\.isNumber).prefix(10)) ?? 0 }
                        )
                    )
                    .keyboardType(.numberPad)
                    .focused($focusedField, equals: .seed)
                    .reportVoiceForgeTextEntry(.seed)
                    .font(.system(size: Lab.typeSize(12), design: .monospaced))
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
                    focusedField = nil
                    model.synthesize()
                } label: {
                    HStack(spacing: 8) {
                        if model.isSynthesizing {
                            ProgressView().tint(.white).controlSize(.small)
                        }
                        Text(
                            model.isSynthesizing
                                ? "Synthesizing"
                                : (model.isLoadingModel ? "Warming model…" : "⚡ Synthesize")
                        )
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(
                    !model.canSynthesizeFromCommand
                )
                if model.isSynthesizing {
                    GalvanicVoiceForge(
                        telemetry: model.forge,
                        elapsed: model.synthesisSeconds,
                        estimatedRemainingSeconds: model.estimatedRemainingSeconds,
                        compact: compact,
                        cancel: model.isSynthesizing ? { model.cancelSynthesis() } : nil
                    )
                    .transition(.opacity.combined(with: .scale(scale: 0.985, anchor: .top)))
                }
                if let audio = model.lastAudio {
                    PlaybackSignalView(samples: audio, player: model.player)
                        .frame(height: compact ? 112 : 148)
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
                                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
                                    .foregroundStyle(Lab.textSecondary)
                            }
                        } else {
                            Button("Make video") { model.exportVideo() }
                                .buttonStyle(GhostButtonStyle())
                        }
                        Spacer()
                        if let factor = model.lastRealTimeFactor {
                            Text(
                                factor >= 1
                                    ? String(format: "%.2f× real-time speed", factor)
                                    : String(
                                        format: "%.2f× speed · %.1fs per 1s audio",
                                        factor, 1 / max(factor, 0.001)
                                    )
                            )
                                .font(.system(size: Lab.typeSize(11), design: .monospaced))
                                .foregroundStyle(factor >= 1 ? Lab.emerald : Lab.textSecondary)
                                .lineLimit(1)
                                .minimumScaleFactor(0.7)
                        }
                    }
                    if let profile = model.lastProfile {
                        DisclosureGroup(isExpanded: $showMachineProfile) {
                            synthesisProfile(profile)
                                .padding(.top, 8)
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "xray")
                                    .foregroundStyle(Lab.cyan)
                                Text("Inside the machine")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(Lab.textPrimary)
                                Spacer()
                                Text("timings · kernels · frames")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(Lab.textSecondary)
                            }
                        }
                        .tint(Lab.cyan)
                    }
                }
                if let error = model.lastError {
                    Text(error).font(.system(size: Lab.typeSize(13))).foregroundStyle(Lab.danger)
                }
            }
        }
    }

    private func synthesisProfile(_ profile: SynthesisProfile) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("ON-DEVICE PROFILE")
                    .foregroundStyle(Lab.emerald)
                Spacer()
                Text("\(profile.teamPartitions)-way · \(profile.frames) frames")
                    .foregroundStyle(Lab.textSecondary)
            }
            Text(
                String(
                    format: "total %.0f ms · generation %.0f ms · codec active %.0f ms (overlapped)",
                    profile.totalMs,
                    profile.generationMs,
                    profile.codecActiveMs
                )
            )
            .foregroundStyle(Lab.textPrimary)
            Text(
                String(
                    format: "prefill %.0f · talker %.0f · residual %.0f · feedback %.0f · other %.0f ms",
                    profile.prefillMs,
                    profile.talkerMs,
                    profile.microdecoderMs,
                    profile.feedbackMs,
                    profile.otherGenerationMs
                )
            )
            .foregroundStyle(Lab.textSecondary)
        }
        .font(.system(size: Lab.typeSize(10), design: .monospaced))
        .padding(10)
        .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 9))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "On-device synthesis profile. \(profile.teamPartitions) workers, "
                + "\(profile.frames) frames, total \(Int(profile.totalMs)) milliseconds."
        )
    }

    private func selectAllUtterance() {
        focusedField = .utterance
        // Focus lands on the underlying UITextView on the next main-loop turn.
        // Sending the standard responder action preserves UIKit's native
        // selection handles, edit menu, undo stack, and replacement behavior.
        DispatchQueue.main.async {
            UIApplication.shared.sendAction(
                #selector(UIResponder.selectAll(_:)),
                to: nil,
                from: nil,
                for: nil
            )
        }
    }

    private func openEnrollment(target: UUID?) {
        guard model.store.phase == .ready else {
            showSpecimen = true
            return
        }
        model.enrollmentTarget = target
        showVoiceLab = false
        Task { @MainActor in
            // Let a presented voice-library sheet dismiss before asking SwiftUI to
            // present the recorder; otherwise the request can be dropped on iPhone.
            try? await Task.sleep(for: .milliseconds(180))
            showEnrollment = true
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
                .font(.system(size: Lab.typeSize(11), design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
            Text(
                "If you like this free app, please show your appreciation by trying out my paid skills site at [JeffreysSkills.md](https://jeffreys-skills.md)."
            )
            .font(.system(size: Lab.typeSize(10), design: .monospaced))
            .foregroundStyle(Lab.textSecondary.opacity(0.72))
            .tint(Lab.emerald.opacity(0.8))
            .multilineTextAlignment(.center)
            .frame(maxWidth: 320)
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

    private static func downloadRate(_ bytesPerSecond: Double) -> String {
        let formatted = ByteCountFormatter.string(
            fromByteCount: Int64(bytesPerSecond), countStyle: .file)
        return "\(formatted)/s"
    }
}

private struct ModelPromise: View {
    let systemImage: String
    let text: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: systemImage)
                .font(.system(size: Lab.typeSize(13), weight: .bold))
                .foregroundStyle(Lab.emerald)
            Text(text)
                .font(.system(size: Lab.typeSize(9), weight: .semibold))
                .foregroundStyle(Lab.textSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct VoiceOrb: View {
    let name: String
    let selected: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(
                    AngularGradient(
                        colors: [Lab.emeraldDeep, Lab.emerald, Lab.cyan, Lab.emeraldDeep],
                        center: .center
                    )
                )
            Circle()
                .fill(Color.black.opacity(0.62))
                .padding(3)
            Text(String(name.prefix(1)).uppercased())
                .font(.system(size: 14, weight: .black, design: .rounded))
                .foregroundStyle(selected ? Lab.emerald : Lab.textPrimary)
        }
        .frame(width: 40, height: 40)
        .shadow(color: selected ? Lab.emerald.opacity(0.36) : .clear, radius: 9)
        .accessibilityHidden(true)
    }
}

private struct CompactVoiceChip: View {
    let name: String
    let selected: Bool
    var isPersonal = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                VoiceOrb(name: name, selected: selected)
                    .frame(width: 30, height: 30)
                    .scaleEffect(0.75)
                    .frame(width: 26, height: 26)
                Text(name.capitalized)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                if isPersonal {
                    Image(systemName: "person.wave.2.fill")
                        .font(.system(size: 9, weight: .bold))
                }
            }
            .foregroundStyle(selected ? Lab.textPrimary : Lab.textSecondary)
            .padding(.horizontal, 10)
            .frame(height: 42)
            .background(
                selected ? Lab.emerald.opacity(0.14) : Color.black.opacity(0.34),
                in: Capsule()
            )
            .overlay {
                Capsule().strokeBorder(
                    selected ? Lab.emerald.opacity(0.52) : Color.white.opacity(0.07),
                    lineWidth: 1
                )
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(name) voice")
        .accessibilityValue(selected ? "Selected" : "Not selected")
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
                    "Read this aloud. The first few sentences are enough for a good clone; the whole script polishes it slightly. Background noise is removed automatically before your voice is learned."
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
