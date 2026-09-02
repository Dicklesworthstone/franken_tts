// The laboratory: one scrolling screen mirroring the site's playground.

import AVFoundation
import CryptoKit
import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

private enum VoiceLibraryFilter: String, CaseIterable, Identifiable {
    case all = "All"
    case feminine = "Feminine"
    case masculine = "Masculine"
    case personal = "My voices"

    var id: Self { self }

    var symbol: String {
        switch self {
        case .all: "waveform"
        case .feminine: "sparkles"
        case .masculine: "waveform.path"
        case .personal: "person.wave.2.fill"
        }
    }

    func includes(_ preset: Preset) -> Bool {
        switch self {
        case .all: true
        case .feminine: preset.character.localizedCaseInsensitiveContains("feminine")
        case .masculine: preset.character.localizedCaseInsensitiveContains("masculine")
        case .personal: false
        }
    }
}

private struct PhoneWorkspaceHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// UIKit owns the mature text-editing behaviors that matter for a long utterance:
/// selection handles, edit menus, undo, dictation, marked text, and cursor-preserving
/// replacement. A SwiftUI `TextEditor` whose binding rewrites the entire string on
/// every keystroke can disturb those behaviors, especially for large pastes and IMEs.
private struct NativeUtteranceEditor: UIViewRepresentable {
    @Binding var text: String
    @Binding var isFocused: Bool
    let maximumLength: Int
    let focusRequest: Int
    let selectAllRequest: Int
    let externalRevision: Int

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.delegate = context.coordinator
        textView.backgroundColor = .clear
        textView.textColor = UIColor(Lab.textPrimary)
        textView.tintColor = UIColor(Lab.emerald)
        textView.font = .systemFont(ofSize: Lab.typeSize(16))
        textView.adjustsFontForContentSizeCategory = true
        textView.autocapitalizationType = .sentences
        textView.autocorrectionType = .yes
        textView.smartDashesType = .yes
        textView.smartQuotesType = .yes
        textView.keyboardDismissMode = .interactive
        textView.keyboardAppearance = .dark
        textView.textContainerInset = UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        textView.textContainer.lineFragmentPadding = 0
        textView.accessibilityIdentifier = "utteranceEditor"
        textView.text = text
        return textView
    }

    func updateUIView(_ textView: UITextView, context: Context) {
        context.coordinator.parent = self

        // A delegate edit updates the binding before this method is called, so the
        // strings normally match. Explicit external actions carry a revision: they own
        // the requested replacement and must also work while autocorrect/IME has marked
        // text. Commit that composition before applying Clear/import/joke, rather than
        // ignoring the action and letting the old document overwrite the model later.
        let externalChange = context.coordinator.lastExternalRevision != externalRevision
        if externalChange {
            context.coordinator.isApplyingExternalChange = true
            context.coordinator.lastExternalRevision = externalRevision
            if textView.markedTextRange != nil { textView.unmarkText() }
        }
        if textView.text != text, externalChange || textView.markedTextRange == nil {
            let oldSelection = textView.selectedRange
            textView.text = text
            let utf16Count = (text as NSString).length
            let location = min(oldSelection.location, utf16Count)
            textView.selectedRange = NSRange(
                location: location,
                length: min(oldSelection.length, max(0, utf16Count - location))
            )
        }
        if externalChange { context.coordinator.isApplyingExternalChange = false }

        // UIKit remains the source of truth for ordinary tap focus. SwiftUI view
        // updates can arrive with a stale boolean during the same event; using that
        // boolean to resign here made the editor visibly select text but reject typing.
        // Only an explicit, monotonically increasing request may take focus.
        if context.coordinator.lastFocusRequest != focusRequest {
            context.coordinator.lastFocusRequest = focusRequest
            DispatchQueue.main.async {
                textView.becomeFirstResponder()
            }
        }

        if context.coordinator.lastSelectAllRequest != selectAllRequest {
            context.coordinator.lastSelectAllRequest = selectAllRequest
            DispatchQueue.main.async {
                textView.becomeFirstResponder()
                textView.selectAll(nil)
            }
        }
    }

    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: NativeUtteranceEditor
        var lastFocusRequest: Int
        var lastSelectAllRequest: Int
        var lastExternalRevision: Int
        var isApplyingExternalChange = false

        init(parent: NativeUtteranceEditor) {
            self.parent = parent
            lastFocusRequest = parent.focusRequest
            lastSelectAllRequest = parent.selectAllRequest
            lastExternalRevision = parent.externalRevision
        }

        func textViewDidBeginEditing(_ textView: UITextView) {
            if !parent.isFocused { parent.isFocused = true }
        }

        func textViewDidEndEditing(_ textView: UITextView) {
            if parent.isFocused { parent.isFocused = false }
        }

        func textViewDidChange(_ textView: UITextView) {
            guard !isApplyingExternalChange else { return }
            // Do not normalize or truncate marked text mid-composition. The delegate
            // range gate below handles ordinary edits; this is a defensive backstop
            // for input systems that commit marked text in one operation.
            guard textView.markedTextRange == nil else { return }
            let value = textView.text ?? ""
            if value.count <= parent.maximumLength {
                if parent.text != value { parent.text = value }
                return
            }

            let selection = textView.selectedRange
            let limited = String(value.prefix(parent.maximumLength))
            textView.text = limited
            textView.selectedRange = NSRange(
                location: min(selection.location, (limited as NSString).length),
                length: 0
            )
            if parent.text != limited { parent.text = limited }
        }

        func textView(
            _ textView: UITextView,
            shouldChangeTextIn range: NSRange,
            replacementText replacement: String
        ) -> Bool {
            // Let marked-text composition proceed naturally. It is checked when the
            // input method commits, avoiding broken accents and CJK entry.
            if textView.markedTextRange != nil { return true }

            let current = textView.text ?? ""
            let candidate = (current as NSString).replacingCharacters(in: range, with: replacement)
            if candidate.count <= parent.maximumLength { return true }

            // Oversized pastes still insert everything that fits. Replace only the
            // edited range so the rest of the document and its selection stay intact.
            let withoutSelection = (current as NSString).replacingCharacters(in: range, with: "")
            let available = max(0, parent.maximumLength - withoutSelection.count)
            let accepted = String(replacement.prefix(available))
            guard !accepted.isEmpty else { return false }

            textView.textStorage.replaceCharacters(in: range, with: accepted)
            textView.selectedRange = NSRange(
                location: range.location + (accepted as NSString).length,
                length: 0
            )
            textViewDidChange(textView)
            return false
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

    mutating func markMeasuredWork(elapsed: TimeInterval) {
        guard elapsed >= 0 else { return }
        hasMeasuredWork = true
        if predictedFinishElapsed == nil {
            predictedFinishElapsed = max(
                elapsed + 0.75,
                learnedSecondsPerWord * Double(words)
            )
        }
    }

    mutating func observeCompletedFraction(_ fraction: Double, elapsed: TimeInterval) {
        guard fraction > 0, fraction < 1, elapsed > 0 else { return }
        hasMeasuredWork = true
        let observedTotal = elapsed / fraction
        let priorTotal = learnedSecondsPerWord * Double(words)
        let candidate = max(elapsed + 0.75, priorTotal * 0.30 + observedTotal * 0.70)
        if let old = predictedFinishElapsed {
            predictedFinishElapsed = min(old * 0.62 + candidate * 0.38, old + 5)
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

    var text = JokeLibrary.random()
    /// Bumped only for non-keyboard replacements so UIKit can reliably distinguish
    /// an explicit Clear/import/joke from an ordinary delegate-driven keystroke.
    var utteranceExternalRevision = 0
    var seed: UInt64 = 0

    var isSynthesizing = false
    /// True while Voice Lab owns the engine for an all-voice comparison run.
    var isComparingVoices = false
    /// True while the hidden physical-device profiler owns the engine.
    ///
    /// The profiler deliberately exercises the shipping engine, but it is not a visible
    /// synthesis run. Without a separate ownership bit, an ordinary memory-warning or
    /// background notification can unload the engine underneath a benchmark and turn an
    /// otherwise valid sample window into a misleading cancellation receipt.
    private var isProfilingBenchmark = false
    var isLoadingModel = false
    var isEngineWarm = false
    var synthesisSeconds = 0.0
    var estimatedRemainingSeconds: Int?
    var lastError: String?
    var textImportNotice: String?
    var isImportingText = false
    var isClearingModel = false
    var lastAudio: [Float]?
    /// The voice that produced `lastAudio`, snapshotted when the run starts.
    ///
    /// `selectedVoice` remains freely browsable after synthesis. Export metadata must
    /// describe the audio the user actually hears, not whichever tile they tapped later.
    var lastAudioVoiceLabel: String?
    var lastRealTimeFactor: Double?
    var lastProfile: SynthesisProfile?
    var synthesisChunkIndex = 1
    var synthesisChunkCount = 1
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
    private var videoExportTask: Task<Void, Never>?
    private var importGeneration = 0
    private var importTask: Task<Void, Never>?
    private var cancellationRequested = false
    private var completedSynthesisFrames: UInt64 = 0
    private var engineWarmTask: Task<Void, Never>?
    /// Allocated before lifecycle tasks are spawned so actor mailbox reordering cannot
    /// let an old memory-pressure unload close a newly claimed engine.
    private var engineLifecycleToken: UInt64 = 0
    /// Fences native callbacks from completed/cancelled warm and synthesis runs.
    private var progressGeneration = 0
    private let activity = VoiceForgeActivityController.shared
    private var eta = TTSAdaptiveETA()

    func nextEngineLifecycleToken() -> UInt64 {
        engineLifecycleToken &+= 1
        return engineLifecycleToken
    }

    var lowMemoryDevice: Bool {
        ProcessInfo.processInfo.physicalMemory < 6 * 1024 * 1024 * 1024
    }

    var canSynthesizeFromCommand: Bool {
        !isSynthesizing
            && !isComparingVoices
            && !isClearingModel
            && !isLoadingModel
            && store.phase == .ready
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canCompareVoices: Bool {
        !isSynthesizing
            && !isComparingVoices
            && !isClearingModel
            && !isLoadingModel
            && store.phase == .ready
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !presets.isEmpty
    }

    var canClearModel: Bool {
        !isSynthesizing
            && !isComparingVoices
            && !isEnrolling
            && !recorder.isRecording
            && !isProfilingBenchmark
            && !isClearingModel
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
        let text = self.text
        let chunks = UtteranceChunker.split(text)
        guard !chunks.isEmpty else { return }
        let speaker: [Float]
        let voiceLabel = currentVoiceLabel
        do {
            // Snapshot the exact conditioning at button press. Resolving it after the
            // asynchronous task starts lets a fast voice-tile tap silently change the run.
            speaker = try speakerVector()
        } catch {
            lastError = error.localizedDescription
            return
        }
        player?.pause()
        isSynthesizing = true
        progressGeneration &+= 1
        let progressRun = progressGeneration
        cancellationRequested = false
        completedSynthesisFrames = 0
        synthesisChunkIndex = 1
        synthesisChunkCount = chunks.count
        lastError = nil
        lastProfile = nil
        synthesisSeconds = 0
        estimatedRemainingSeconds = nil
        nativeProgressEvents.removeAll(keepingCapacity: true)
        let beganWarm = isEngineWarm
        forge.reset(for: beganWarm ? .checkingMemory : .readingBundle)
        eta.reset(text: text, warm: beganWarm)
        activity.begin()
        let seed = self.seed
        let lifecycleToken = nextEngineLifecycleToken()
        let runStartedAt = ProcessInfo.processInfo.systemUptime
        let ticker = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.synthesisSeconds = max(
                    0, ProcessInfo.processInfo.systemUptime - runStartedAt)
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
                isLoadingModel = await !engine.isLoaded
                try await engine.load(
                    modelDirectory: store.modelDirectory,
                    lifecycleToken: lifecycleToken
                ) { [weak self] event in
                    DispatchQueue.main.async {
                        self?.receive(event, generation: progressRun)
                    }
                }
                isEngineWarm = true
                isLoadingModel = false
                let denoiseAvailable = await engine.denoiseAvailable
                var pcm: [Float] = []
                var aggregateProfile: SynthesisProfile?
                var everyProfileAvailable = true
                var synthesisElapsed = 0.0
                var generatedSampleCount = 0
                var completedCharacters = 0

                for (index, chunk) in chunks.enumerated() {
                    guard !cancellationRequested else { throw EngineError.cancelled }
                    synthesisChunkIndex = index + 1
                    let frameOffset = completedSynthesisFrames
                    let started = ProcessInfo.processInfo.systemUptime
                    let output = try await engine.synthesize(
                        text: chunk.text,
                        speaker: speaker,
                        seed: seed &+ UInt64(index)
                    ) { [weak self] event in
                        DispatchQueue.main.async {
                            self?.receive(
                                event,
                                chunkOrdinal: index + 1,
                                frameOffset: frameOffset,
                                generation: progressRun
                            )
                        }
                    }
                    synthesisElapsed += max(
                        0, ProcessInfo.processInfo.systemUptime - started)
                    guard !cancellationRequested else { throw EngineError.cancelled }

                    if let profile = output.profile {
                        aggregateProfile = aggregateProfile?.adding(profile) ?? profile
                        completedSynthesisFrames += profile.frames
                    } else {
                        everyProfileAvailable = false
                        completedSynthesisFrames += UInt64(
                            output.pcm.count / 1_920
                        )
                    }
                    generatedSampleCount += output.pcm.count

                    var chunkPCM = output.pcm
                    if denoiseAvailable {
                        forge.phase = .denoising
                        if index == chunks.count - 1 {
                            eta.beginDenoise(
                                elapsed: max(
                                    0, ProcessInfo.processInfo.systemUptime - runStartedAt))
                        }
                        chunkPCM = (try? await engine.denoise(pcm: chunkPCM)) ?? chunkPCM
                    }
                    guard !cancellationRequested else { throw EngineError.cancelled }

                    chunkPCM = await Task.detached(priority: .userInitiated) {
                        SpeechMastering.process(chunkPCM)
                    }.value
                    pcm.append(contentsOf: chunkPCM)
                    if index < chunks.count - 1 {
                        let pauseSamples = Int(
                            chunk.trailingPauseSeconds * Double(WavWriter.sampleRate)
                        )
                        pcm.append(contentsOf: repeatElement(Float.zero, count: pauseSamples))
                    }

                    completedCharacters += chunk.text.count
                    eta.observeCompletedFraction(
                        Double(completedCharacters) / Double(max(1, text.count)),
                        elapsed: max(0, ProcessInfo.processInfo.systemUptime - runStartedAt)
                    )
                }

                guard !cancellationRequested else { throw EngineError.cancelled }
                let completedProfile = everyProfileAvailable ? aggregateProfile : nil
                forge.generatedFrames = completedSynthesisFrames
                forge.decodedFrames = max(forge.decodedFrames, completedSynthesisFrames)
                forge.decodedSamples = UInt64(generatedSampleCount)
                let factor = (Double(generatedSampleCount) / Double(WavWriter.sampleRate))
                    / max(0.000_001, synthesisElapsed)
                synthesisSeconds = max(
                    0, ProcessInfo.processInfo.systemUptime - runStartedAt)
                eta.finish(
                    elapsed: synthesisSeconds,
                    frames: completedSynthesisFrames
                )
                estimatedRemainingSeconds = nil
                try startPlayback(of: pcm, voiceLabel: voiceLabel)
                lastProfile = completedProfile
                lastRealTimeFactor = factor
                UserDefaults.standard.set(factor, forKey: "measuredRealTimeFactor")
                progressGeneration &+= 1
                forge.phase = .complete
                activity.finish(
                    status: .complete,
                    headline: "Voice alive",
                    detail: "Your private on-device audio is ready"
                )
            } catch EngineError.cancelled {
                progressGeneration &+= 1
                forge.phase = .cancelled
                activity.finish(
                    status: .cancelled,
                    headline: "Forge stopped",
                    detail: "No partial audio was published"
                )
            } catch {
                progressGeneration &+= 1
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

    /// The two fixed benchmark utterances: a phonetically rich short sentence and a
    /// connected-prose long paragraph, so cold/warm and voice scenarios stay comparable
    /// across runs and trees.
    private static let shortProfileText =
        "Frankenstein listened carefully while the rain tapped softly against the laboratory window."
    private static let longProfileText =
        "The creature spoke slowly at first, choosing each word as though it had cost him years of listening to learn. Every sentence carried the weight of that long winter in the mountains, when he had watched the family through the crack in the wall and taught himself the music of human speech. By the time the rain stopped, he could shape grief and hope in the same breath, and the doctor's lamp burned low over the unfinished notes."

    func runProfilingBenchmarkIfRequested() async {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FTTS_IOS_PROFILE"] == "1" else { return }
        guard !isProfilingBenchmark else { return }
        isProfilingBenchmark = true
        defer { isProfilingBenchmark = false }
        let lifecycleToken = nextEngineLifecycleToken()

        let requestedRuns = Int(environment["FTTS_IOS_PROFILE_RUNS"] ?? "20") ?? 20
        let runs = max(1, min(100, requestedRuns))
        let scenario = environment["FTTS_IOS_PROFILE_SCENARIO"] ?? "short"
        let benchmarkText = environment["FTTS_IOS_PROFILE_TEXT"]
            ?? (scenario == "long" ? Self.longProfileText : Self.shortProfileText)
        let voice = environment["FTTS_IOS_PROFILE_VOICE"] ?? "matt"
        let streaming = environment["FTTS_IOS_PROFILE_STREAMING"] == "1"
        let requestedPacketFrames = Int(
            environment["FTTS_IOS_PROFILE_PACKET_FRAMES"] ?? "1") ?? 1
        let packetFrames = max(1, min(1_024, requestedPacketFrames))
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
            await store.waitForLaunchValidation()
            guard store.phase == .ready else {
                throw EngineError.native("profiling requires a verified downloaded model")
            }
            let speaker: [Float]
            if voice.hasPrefix("enrolled") {
                let wanted = voice.split(separator: ":").dropFirst().first.map(String.init)
                let match = library.voices.first { candidate in
                    wanted.map { candidate.name.caseInsensitiveCompare($0) == .orderedSame }
                        ?? true
                }
                guard let match else {
                    throw EngineError.native(
                        wanted.map { "no enrolled voice named \($0) on this device" }
                            ?? "no enrolled voice on this device; enroll one to profile the cloned-voice scenario"
                    )
                }
                speaker = match.vector
            } else {
                speaker = try Engine.presetVector(named: voice)
            }
            try appendReceipt([
                "event": "run_start",
                "schema_version": 2,
                "runs": runs,
                "scenario": scenario,
                "voice": voice,
                "seed": benchmarkSeed,
                "streaming_packet_frames": packetFrames,
                "device_model": UIDevice.current.model,
                "system_name": UIDevice.current.systemName,
                "system_version": UIDevice.current.systemVersion,
                "active_processors": ProcessInfo.processInfo.activeProcessorCount,
                "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "team_threads": environment["FTTS_INT8_THREADS"] ?? "unset",
                "codec_queue_frames": environment["FTTS_CODEC_QUEUE_FRAMES"] ?? "unset",
                "codec_user_initiated_qos": environment["FTTS_CODEC_USER_INITIATED_QOS"] ?? "unset",
                "accelerate_row_block": environment["FTTS_ACCELERATE_ROW_BLOCK"] ?? "default",
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

            let loadStartedUptime = ProcessInfo.processInfo.systemUptime
            try await engine.load(
                modelDirectory: store.modelDirectory,
                lifecycleToken: lifecycleToken
            )
            try appendReceipt([
                "event": "engine_loaded",
                "load_ms": (ProcessInfo.processInfo.systemUptime - loadStartedUptime) * 1_000,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])

            var firstDigest: String?
            var validRuns = 0
            var allAudioIdentical = true
            for index in 0..<runs {
                let startedUptime = ProcessInfo.processInfo.systemUptime
                let output = try await engine.synthesize(
                    text: benchmarkText,
                    speaker: speaker,
                    seed: benchmarkSeed
                )
                let wallMs = (ProcessInfo.processInfo.systemUptime - startedUptime) * 1_000
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
                var ttfaMs = 0.0
                var streamingMatches = false
                var streamingEquivalent = false
                var streamingDigest = ""
                var streamingSampleCount = 0
                var firstStreamingMismatch = -1
                var maxStreamingSampleDelta = 0.0
                if streaming {
                    let streamed = try await engine.synthesizeStreaming(
                        text: benchmarkText,
                        speaker: speaker,
                        seed: benchmarkSeed,
                        packetFrames: packetFrames
                    )
                    ttfaMs = Double(streamed.firstAudioNanos) / 1_000_000
                    let streamingWav = WavWriter.data(from: streamed.pcm)
                    streamingDigest = SHA256.hash(data: streamingWav)
                        .map { String(format: "%02x", $0) }.joined()
                    streamingSampleCount = streamed.pcm.count
                    streamingMatches = streamingDigest == digest
                    for index in 0..<min(output.pcm.count, streamed.pcm.count) {
                        let whole = output.pcm[index]
                        let packetized = streamed.pcm[index]
                        if firstStreamingMismatch < 0,
                           whole.bitPattern != packetized.bitPattern
                        {
                            firstStreamingMismatch = index
                        }
                        maxStreamingSampleDelta = max(
                            maxStreamingSampleDelta,
                            Double(abs(whole - packetized))
                        )
                    }
                    if firstStreamingMismatch < 0,
                       output.pcm.count != streamed.pcm.count
                    {
                        firstStreamingMismatch = min(output.pcm.count, streamed.pcm.count)
                    }
                    streamingEquivalent = output.pcm.count == streamed.pcm.count
                        && maxStreamingSampleDelta <= 0.000_001
                }
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
                    "codec_backpressure_ms": profile.codecBackpressureMs,
                    "codec_tail_ms": profile.codecTailMs,
                    "codec_user_initiated_qos_applied": profile.codecUserInitiatedQos,
                    "other_generation_ms": profile.otherGenerationMs,
                    "generator_glue_ms": profile.generatorGlueMs,
                    "frames": profile.frames,
                    "team_partitions": profile.teamPartitions,
                    "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                    "ttfa_ms": ttfaMs,
                    "streaming_measured": streaming,
                    "streaming_matches_whole_buffer": streamingMatches,
                    "streaming_pcm_equivalent": streamingEquivalent,
                    "streaming_wav_sha256": streamingDigest,
                    "streaming_sample_count": streamingSampleCount,
                    "streaming_first_mismatch_sample": firstStreamingMismatch,
                    "streaming_max_abs_sample_delta": maxStreamingSampleDelta,
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
        cancellationRequested = true
        forge.phase = .cancelling
        engine.cancelCurrentWork()
    }

    private func receive(
        _ event: EngineProgress,
        chunkOrdinal: Int? = nil,
        frameOffset: UInt64 = 0,
        generation: Int
    ) {
        guard generation == progressGeneration,
              !(isSynthesizing && cancellationRequested)
        else { return }
        if let chunkOrdinal, chunkOrdinal != synthesisChunkIndex { return }
        forge.apply(event)
        if synthesisChunkCount > 1, chunkOrdinal != nil {
            if event.kind == .unit, event.stage == .frames {
                forge.generatedFrames = frameOffset + event.current
                forge.predictedMaximumFrames = 0
                if event.current > 0 { eta.markMeasuredWork(elapsed: synthesisSeconds) }
            } else if event.kind == .unit, event.stage == .codec {
                forge.decodedFrames = frameOffset + event.current
                forge.predictedMaximumFrames = 0
            }
        } else {
            eta.observe(event, elapsed: synthesisSeconds)
        }
        estimatedRemainingSeconds = eta.remainingSeconds(at: synthesisSeconds)
        activity.update(from: forge, elapsed: synthesisSeconds)
        nativeProgressEvents.append(event)
        if nativeProgressEvents.count > 160 {
            nativeProgressEvents.removeFirst(nativeProgressEvents.count - 160)
        }
    }

    private func startPlayback(of pcm: [Float], voiceLabel: String) throws {
        let wav = WavWriter.data(from: pcm)
        // Unique per synthesis: an in-flight video export reads the previous WAV for its
        // audio track, and overwriting it mid-read would mux corrupt audio.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "franken_tts-\(ProcessInfo.processInfo.globallyUniqueString).wav")
        var committed = false
        defer {
            if !committed { try? FileManager.default.removeItem(at: url) }
        }
        try wav.write(to: url)
        try AVAudioSession.sharedInstance().setCategory(.playback)
        let preparedPlayer = try AVAudioPlayer(contentsOf: url)

        // Everything that can fail is complete. Publish the PCM, player, URLs, and
        // producing voice as one coherent result so the UI can never pair new samples
        // with the previous run's player or share artifacts.
        // A completed synthesis supersedes the previous clip. Stop only now—not when
        // synthesis begins—so a failed/cancelled forge leaves its prior export usable.
        videoExportTask?.cancel()
        videoExportTask = nil
        isExportingVideo = false
        videoProgress = 0
        wavUrl = url
        m4aUrl = nil
        videoUrl = nil
        synthesisGeneration += 1
        let generation = synthesisGeneration
        lastAudio = pcm
        player = preparedPlayer
        lastAudioVoiceLabel = voiceLabel
        player?.play()
        committed = true
        // The share default is the small file; convert as soon as audio exists.
        Task {
            let converted = try? await MediaExporter.exportM4A(fromWav: url)
            if generation == synthesisGeneration { m4aUrl = converted }
        }
    }

    func togglePlayback() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
        } else {
            if player.duration > 0, player.currentTime >= player.duration - 0.05 {
                player.currentTime = 0
            }
            player.play()
        }
    }

    func seekPlayback(to progress: Double) {
        guard let player, player.duration > 0 else { return }
        player.currentTime = player.duration * min(1, max(0, progress))
    }

    /// The label stamped on the video's voice pill.
    var currentVoiceLabel: String {
        if let id = enrolledSelection(), let voice = library.voice(id: id) {
            return voice.name
        }
        return selectedVoice.capitalized
    }

    var lastAudioExportVoiceLabel: String {
        lastAudioVoiceLabel ?? currentVoiceLabel
    }

    func exportVideo() {
        guard let wavUrl, let audio = lastAudio, !isExportingVideo else { return }
        isExportingVideo = true
        videoProgress = 0
        let label = lastAudioExportVoiceLabel
        let generation = synthesisGeneration
        videoExportTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if generation == self.synthesisGeneration {
                    self.isExportingVideo = false
                    self.videoExportTask = nil
                }
            }
            do {
                let rendered = try await MediaExporter.exportVideo(
                    pcm: audio, voiceLabel: label, wavUrl: wavUrl
                ) { [weak self] fraction in
                    Task { @MainActor in
                        guard let self, generation == self.synthesisGeneration else { return }
                        self.videoProgress = fraction
                    }
                }
                if generation == self.synthesisGeneration { self.videoUrl = rendered }
            } catch is CancellationError {
                // A newer successful synthesis owns the share surface now.
            } catch {
                if generation == self.synthesisGeneration {
                    self.lastError = error.localizedDescription
                }
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
        let lifecycleToken = nextEngineLifecycleToken()
        Task {
            defer { isEnrolling = false }
            do {
                let pcm = try await Task.detached(priority: .userInitiated) {
                    try Self.conditioned(raw)
                }.value
                guard pcm.count >= 3 * Int(AudioRecorder.targetRate) else {
                    throw EngineError.native(
                        "recording too short; a few sentences of the script is all it takes")
                }
                try await engine.load(
                    modelDirectory: store.modelDirectory,
                    lifecycleToken: lifecycleToken
                )
                // The denoiser is not optional: a profile built from un-denoised audio
                // carries the recording's noise into every synthesis. Its absence
                // means the model download is incomplete — refuse and say so.
                guard await engine.denoiseAvailable else {
                    throw EngineError.native(
                        "the noise-removal file is missing; it downloads automatically — check the connection, relaunch, and try again")
                }
                let vector = try await engine.enroll(pcm: pcm)
                isEngineWarm = true
                let selected: UUID
                if let target = enrollmentTarget {
                    try library.update(id: target, name: trimmedName, vector: vector)
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

    /// Trim edge silence, remove rumble, gently correct microphone coloration, and
    /// normalize active speech before enrollment. The encoder embeds whatever it is
    /// given; conditioning the reference keeps room and microphone differences from
    /// dominating the resulting voiceprint. Refusing outright silence beats enrolling it.
    nonisolated private static func conditioned(_ pcm: [Float]) throws -> [Float] {
        var peak: Float = 0
        for value in pcm where value.isFinite { peak = max(peak, abs(value)) }
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
        return SpeechMastering.process(
            Array(pcm[low..<high]),
            sampleRate: Float(AudioRecorder.targetRate),
            maximumGain: 32
        )
    }

    /// Frees the ~2.3 GB engine heap; the next synthesis reloads it.
    func unloadEngineForMemoryPressure() {
        guard !isSynthesizing,
              !isComparingVoices,
              !isEnrolling,
              !isClearingModel,
              !isProfilingBenchmark
        else { return }
        // A callback already queued onto the main actor can outlive cancellation of
        // the warm task. Fence it before changing the visible warm/loading state.
        progressGeneration &+= 1
        engine.cancelCurrentWork()
        engineWarmTask?.cancel()
        engineWarmTask = nil
        isEngineWarm = false
        isLoadingModel = false
        let lifecycleToken = nextEngineLifecycleToken()
        Task { await engine.unload(lifecycleToken: lifecycleToken) }
    }

    /// Hydrate as soon as verified model files are present. Users should wait
    /// for the model once at app/foreground entry, not discover the same cold
    /// start only after pressing Synthesize.
    func warmEngineIfPossible() {
        guard store.phase == .ready,
              !isEngineWarm,
              engineWarmTask == nil,
              !isSynthesizing,
              !isComparingVoices,
              !isEnrolling,
              !isClearingModel,
              !recorder.isRecording
        else { return }

        isLoadingModel = true
        progressGeneration &+= 1
        let progressRun = progressGeneration
        let lifecycleToken = nextEngineLifecycleToken()
        engineWarmTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.isLoadingModel = false
                self.engineWarmTask = nil
            }
            do {
                if await !self.engine.isLoaded {
                    self.forge.reset(for: .readingBundle)
                }
                try await self.engine.load(
                    modelDirectory: self.store.modelDirectory,
                    lifecycleToken: lifecycleToken
                ) { [weak self] event in
                    DispatchQueue.main.async {
                        self?.receive(event, generation: progressRun)
                    }
                }
                try Task.checkCancellation()
                self.progressGeneration &+= 1
                self.isEngineWarm = true
                if !self.isSynthesizing { self.forge.phase = .idle }
            } catch EngineError.cancelled {
                self.progressGeneration &+= 1
                self.isEngineWarm = false
                self.forge.phase = .cancelled
            } catch is CancellationError {
                self.progressGeneration &+= 1
                self.isEngineWarm = false
                self.forge.phase = .cancelled
            } catch {
                self.progressGeneration &+= 1
                self.isEngineWarm = false
                self.forge.phase = .failed
                self.lastError = "Could not warm the model: \(error.localizedDescription)"
            }
        }
    }

    func clearModel() {
        guard canClearModel else {
            lastError = "Wait for the current voice operation to finish before clearing the model."
            return
        }

        isClearingModel = true
        progressGeneration &+= 1
        engine.cancelCurrentWork()
        engineWarmTask?.cancel()
        engineWarmTask = nil
        isEngineWarm = false
        isLoadingModel = false
        let lifecycleToken = nextEngineLifecycleToken()
        Task {
            await engine.unload(lifecycleToken: lifecycleToken)
            store.clear()
            isClearingModel = false
        }
    }

    /// A microphone capture must never continue after the app leaves the foreground.
    /// Enrollment computation, however, already owns a completed in-memory take; keep
    /// that work alive and let the engine actor finish when iOS resumes the process.
    func prepareForBackground() {
        if recorder.isRecording {
            _ = recorder.stop()
            lastError = "Recording stopped when FrankenTTS left the foreground. Tap Start recording to try again."
        }
        unloadEngineForMemoryPressure()
    }

    func importDesktopFile(_ url: URL) {
        let supportedText = ["txt", "md", "markdown"]
        let ext = url.pathExtension.lowercased()
        let isTextFile = supportedText.contains(ext)
            || UTType(filenameExtension: ext)?.conforms(to: .text) == true
        let scoped = url.startAccessingSecurityScopedResource()
        importTask?.cancel()
        importGeneration += 1
        let generation = importGeneration
        isImportingText = true
        importTask = Task {
            defer {
                if scoped { url.stopAccessingSecurityScopedResource() }
                if generation == importGeneration {
                    isImportingText = false
                    importTask = nil
                }
            }
            do {
                try Task.checkCancellation()
                if isTextFile {
                    let imported = try await Task.detached(priority: .userInitiated) {
                        try TextImportLoader.readTextFile(from: url)
                    }.value
                    guard generation == importGeneration else { return }
                    replaceUtterance(with: imported.text, wasTruncated: imported.wasTruncated)
                    return
                }
                if ext == "pdf" {
                    let imported = try await Task.detached(priority: .userInitiated) {
                        try TextImportLoader.extractPDF(from: url)
                    }.value
                    guard generation == importGeneration else { return }
                    replaceUtterance(with: imported.text, wasTruncated: imported.wasTruncated)
                    return
                }
                let data = try await Task.detached(priority: .userInitiated) {
                    try Data(contentsOf: url, options: .mappedIfSafe)
                }.value
                guard let (name, vector) = await Task.detached(priority: .userInitiated, operation: {
                    VoicePrintCard.decode(data)
                }).value else {
                    throw EngineError.native("that image does not contain a FrankenTTS voice card")
                }
                guard generation == importGeneration else { return }
                if let existing = library.voices.first(where: { $0.vector == vector }) {
                    selectedVoice = "voice:\(existing.id.uuidString)"
                } else {
                    let voice = try library.add(name: name, vector: vector)
                    selectedVoice = "voice:\(voice.id.uuidString)"
                }
            } catch {
                guard generation == importGeneration, !Task.isCancelled else { return }
                if isTextFile || ext == "pdf" {
                    textImportNotice = error.localizedDescription
                } else {
                    lastError = error.localizedDescription
                }
            }
        }
    }

    func importRemoteText(_ rawURL: String) {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed) else {
            textImportNotice = "Enter a complete HTTPS URL for a web page, text file, or PDF."
            return
        }
        importTask?.cancel()
        importGeneration += 1
        let generation = importGeneration
        isImportingText = true
        importTask = Task {
            defer {
                if generation == importGeneration {
                    isImportingText = false
                    importTask = nil
                }
            }
            do {
                let imported = try await TextImportLoader.download(from: url)
                guard generation == importGeneration else { return }
                replaceUtterance(with: imported.text, wasTruncated: imported.wasTruncated)
            } catch {
                if generation == importGeneration, !Task.isCancelled {
                    textImportNotice = error.localizedDescription
                }
            }
        }
    }

    /// Replaces the editor contents in response to an explicit user action.
    ///
    /// PDFKit extraction is deliberately off the main actor and can outlive the tap that
    /// started it. A manual edit, Clear, joke selection, or system handoff must revoke that
    /// task's publication right; otherwise a late PDF result can resurrect text the user
    /// already removed and make the editor appear impossible to clear.
    func replaceUtteranceFromUser(with value: String) {
        cancelTextImport()
        utteranceExternalRevision &+= 1
        text = String(value.prefix(JokeLibrary.maximumUtteranceLength))
    }

    func updateUtteranceFromEditor(_ value: String) {
        cancelTextImport()
        text = String(value.prefix(JokeLibrary.maximumUtteranceLength))
    }

    func clearUtterance() {
        cancelTextImport()
        textImportNotice = nil
        utteranceExternalRevision &+= 1
        text = ""
    }

    private func cancelTextImport() {
        guard importTask != nil || isImportingText else { return }
        importGeneration &+= 1
        importTask?.cancel()
        importTask = nil
        isImportingText = false
    }

    private func replaceUtterance(with imported: String, wasTruncated: Bool = false) {
        utteranceExternalRevision &+= 1
        text = String(imported.prefix(JokeLibrary.maximumUtteranceLength))
        if wasTruncated || imported.count > JokeLibrary.maximumUtteranceLength {
            textImportNotice = "Imported the first 50,000 characters. The source was longer than the utterance limit."
        }
    }
}

struct LabView: View {
    @AppStorage(LabAppearance.storageKey) private var appearance = LabAppearance.dark.rawValue
    private enum EditorFocus: Hashable {
        case seed
    }

    @State private var model = LabModel()
    @State private var showEnrollment =
        ProcessInfo.processInfo.environment["FTTS_DEBUG_ENROLLMENT"] == "1"
    @State private var showGalaxy = false
    @State private var showSpecimen = false
    @State private var showVoiceLab = ProcessInfo.processInfo.environment["FTTS_OPEN_VOICE_LIBRARY"] == "1"
    @State private var showVoiceComparison =
        ProcessInfo.processInfo.environment["FTTS_OPEN_VOICE_COMPARISON"] == "1"
    #if DEBUG
        @State private var showSynthesisInstrument =
            ProcessInfo.processInfo.environment["FTTS_DEBUG_SYNTHESIS_UI"] == "1"
    #endif
    @State private var renameTarget: EnrolledVoice?
    @State private var renameText = ""
    @State private var cardVoice: EnrolledVoice?
    @State private var importItem: PhotosPickerItem?
    @State private var importFailed = false
    @State private var importCount = 0
    @State private var voiceCardImportGeneration = 0
    @State private var showDesktopImporter = false
    @State private var showURLImporter = false
    @State private var importURLText = ""
    @State private var voiceSearchText = ""
    @State private var voiceLibraryFilter: VoiceLibraryFilter = .all
    @State private var utteranceFocusRequest = 0
    @State private var utteranceSelectAllRequest = 0
    @State private var isUtteranceFocused = false
    @State private var phoneWorkspaceHeight: CGFloat = 0
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

    /// Unit-test host launches should exercise the requested Swift test, not
    /// silently hydrate a 2.3 GB native engine in the background. Besides
    /// making the default suite needlessly slow, those deliberate app-lifetime
    /// worker threads are reported as leaks by Thread Sanitizer when XCTest
    /// terminates its host process.
    private var automaticWarmSuppressed: Bool {
        profilingRequested
            || ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    var body: some View {
        systemIntegrationView
    }

    private var workspaceView: some View {
        GeometryReader { geometry in
            ZStack {
                LaboratoryBackground()
                    .contentShape(Rectangle())
                    .onTapGesture { dismissKeyboard() }
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
                        .scrollDismissesKeyboard(.interactively)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 14)
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            phoneWorkspace(compact: true)
                                .background {
                                    GeometryReader { contentGeometry in
                                        Color.clear.preference(
                                            key: PhoneWorkspaceHeightKey.self,
                                            value: contentGeometry.size.height
                                        )
                                    }
                                }
                        }
                        .scrollIndicators(.hidden)
                        .scrollDismissesKeyboard(.interactively)
                        // Keep the normal ready screen as a fixed instrument panel.
                        // Scrolling switches on only for genuinely taller content or
                        // while editing, without replacing the UIKit editor hierarchy.
                        .scrollDisabled(
                            !isUtteranceFocused && focusedField == nil
                                && phoneWorkspaceHeight <= geometry.size.height + 1
                        )
                        .onPreferenceChange(PhoneWorkspaceHeightKey.self) { height in
                            phoneWorkspaceHeight = height
                        }
                        .onChange(of: isUtteranceFocused) { _, focused in
                            guard focused else { return }
                            Task { @MainActor in
                                // Wait for the keyboard's safe-area animation, then
                                // keep the complete composer (not merely its cursor)
                                // visible above it.
                                try? await Task.sleep(for: .milliseconds(260))
                                withAnimation(.snappy(duration: 0.22)) {
                                    proxy.scrollTo("utterance-card", anchor: .bottom)
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                }
            }
            .catalystReadableType()
        }
    }

    private func phoneWorkspace(compact: Bool) -> some View {
        VStack(alignment: .leading, spacing: compact ? 10 : 16) {
            header
            modelEntryView
            compactVoiceSelector(vertical: false)
            utteranceCard(compact: compact)
            footer
        }
        .frame(maxWidth: 760)
        .fixedSize(horizontal: false, vertical: true)
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
            .presentationDetents([.medium, .large])
        }
        .fullScreenCover(isPresented: $showVoiceLab) {
            NavigationStack {
                ScrollView { voicesCard.padding(.horizontal, 18).padding(.vertical, 20) }
                    .background(LaboratoryBackground())
                    .navigationTitle("Voice Library")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showVoiceLab = false }
                        }
                    }
            }
            // The share button lives inside this full-screen cover, so this cover
            // must own the card sheet. Asking the obscured root view to present a
            // second modal could be delayed or silently dropped by UIKit.
            .sheet(item: $cardVoice) { voice in
                VoiceCardSheet(voice: voice)
            }
        }
        .fullScreenCover(isPresented: $showVoiceComparison) {
            NavigationStack {
                VoiceComparisonView(model: model) {
                    showVoiceComparison = false
                }
            }
        }
        #if DEBUG
            .fullScreenCover(isPresented: $showSynthesisInstrument) {
                SynthesisInstrumentHarness {
                    showSynthesisInstrument = false
                }
            }
        #endif
        .sheet(isPresented: $showGalaxy) {
            VoiceGalaxyView(presets: model.presets, enrolled: model.library.voices)
        }
    }

    private var importView: some View {
        sheetView
        .fileImporter(
            isPresented: $showDesktopImporter,
            allowedContentTypes: [.plainText, .pdf, .image],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { model.importDesktopFile(url) }
            case .failure(let error):
                model.lastError = error.localizedDescription
            }
        }
        .alert("Import text from URL", isPresented: $showURLImporter) {
            TextField("https://example.com/script.txt", text: $importURLText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Import") {
                model.importRemoteText(importURLText)
                importURLText = ""
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("FrankenTTS extracts the main reading text from a web page, text file, or PDF. The resulting utterance stays on this device.")
        }
        .alert(
            "Text import",
            isPresented: Binding(
                get: { model.textImportNotice != nil },
                set: { if !$0 { model.textImportNotice = nil } }
            )
        ) {
            Button("OK", role: .cancel) { model.textImportNotice = nil }
        } message: {
            Text(model.textImportNotice ?? "")
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
            voiceCardImportGeneration &+= 1
            let generation = voiceCardImportGeneration
            Task {
                let data = try? await item.loadTransferable(type: Data.self)
                guard generation == voiceCardImportGeneration, !Task.isCancelled else { return }
                let decoded = await Task.detached(priority: .userInitiated) {
                    data.flatMap { VoicePrintCard.decode($0) }
                }.value
                // Photo-provider reads and million-pixel card decoding can
                // complete out of order. Only the latest selection may add a
                // voice or surface an error for the user.
                guard generation == voiceCardImportGeneration, !Task.isCancelled else { return }
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
                model.prepareForBackground()
            } else if phase == .active {
                if !automaticWarmSuppressed { model.warmEngineIfPossible() }
                consumeStagedText()
            }
        }
        .onChange(of: model.store.phase) { _, phase in
            if phase == .ready, !automaticWarmSuppressed {
                model.warmEngineIfPossible()
            }
        }
        .sensoryFeedback(.selection, trigger: model.selectedVoice)
        .sensoryFeedback(.success, trigger: model.lastAudio?.count)
        .sensoryFeedback(.success, trigger: model.enrollmentSaved) { _, saved in saved }
        .sensoryFeedback(.success, trigger: importCount) { _, count in count > 0 }
        .onAppear(perform: debugCardHook)
        .onAppear(perform: debugLongVoiceNameHook)
        .onAppear(perform: debugVideoHook)
        .task {
            if profilingRequested {
                await model.runProfilingBenchmarkIfRequested()
            } else if !automaticWarmSuppressed {
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
                    dismissKeyboard()
                    model.synthesize()
                },
                stop: { model.cancelSynthesis() },
                togglePlayback: {
                    model.togglePlayback()
                    playbackTick += 1
                },
                canSynthesize: model.canSynthesizeFromCommand,
                canStop: model.isSynthesizing,
                canTogglePlayback:
                    model.player != nil && focusedField == nil && !isUtteranceFocused
            )
        )
        .preferredColorScheme((LabAppearance(rawValue: appearance) ?? .dark).colorScheme)
    }

    private func consumeStagedText() {
        guard let staged = FrankenTTSSharedStore.consumeStagedText(), !staged.isEmpty else { return }
        model.replaceUtteranceFromUser(with: staged)
        focusUtteranceAfterCurrentTap()
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
                FrankenWordmark(
                    productInitial: "TTS",
                    productRemainder: "",
                    fullName: "FrankenTTS"
                )
                Text("VOICE_ALIVE")
                    .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                    .kerning(2)
                    .foregroundStyle(Lab.emerald)
            }
            Spacer()
            LabAppearanceButton(selection: $appearance)
        }
        .accessibilityElement(children: .contain)
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
    private func compactVoiceSelector(vertical _: Bool) -> some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    LabLabel(text: "Voice specimen")
                    Spacer()
                    Text("\(model.presets.count) BUILT-IN · \(model.library.voices.count) YOURS")
                        .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                        .kerning(0.8)
                        .foregroundStyle(Lab.textSecondary)
                }

                VoiceEnrollmentCallout(
                    isReady: model.store.phase == .ready,
                    compact: true
                ) {
                    openEnrollment(target: nil)
                }

                Button { showVoiceLab = true } label: {
                    HStack(spacing: 12) {
                        VoiceOrb(name: model.currentVoiceLabel, selected: true)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(model.currentVoiceLabel)
                                .font(.headline)
                                .foregroundStyle(Lab.textPrimary)
                                .lineLimit(1)
                                .minimumScaleFactor(0.68)
                                .allowsTightening(true)
                            Text("Selected voice")
                                .font(.caption)
                                .foregroundStyle(Lab.textSecondary)
                        }
                        Spacer()
                        HStack(spacing: 5) {
                            Text("MANAGE VOICES")
                            Image(systemName: "chevron.right")
                        }
                        .font(.system(size: Lab.typeSize(9), weight: .black, design: .monospaced))
                        .foregroundStyle(Color.black.opacity(0.84))
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)
                        .padding(.horizontal, 11)
                        .frame(minHeight: 36)
                        .background(Lab.emerald, in: Capsule())
                    }
                    .padding(12)
                    .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(Lab.emerald.opacity(0.28), lineWidth: 1)
                    }
                }
                .buttonStyle(.plain)
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
                            Button {
                                model.clearModel()
                            } label: {
                                if model.isClearingModel {
                                    ProgressView().controlSize(.small).tint(Lab.danger)
                                } else {
                                    Text("Clear")
                                }
                            }
                            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                            .disabled(!model.canClearModel)
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
                    Button(model.store.cachedBytes > 0 ? "Repair model" : "Try again") {
                        model.store.startDownload()
                    }
                        .buttonStyle(PrimaryButtonStyle())
                }
            }
        }
    }

    private var voicesCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            voiceArchiveHero
            VoiceEnrollmentCallout(
                isReady: model.store.phase == .ready,
                compact: false
            ) {
                openEnrollment(target: nil)
            }
            voiceSearchField
            voiceFilterBar

            if voiceLibraryFilter != .personal {
                voiceSectionHeader(
                    "BUILT-IN SPECIMENS",
                    detail: "\(filteredPresetVoices.count) of \(model.presets.count) voices"
                )
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 190), spacing: 12)],
                    spacing: 12
                ) {
                    ForEach(filteredPresetVoices) { preset in
                        VoiceTile(
                            name: preset.name,
                            character: preset.character,
                            selected: model.selectedVoice == preset.name
                        ) {
                            withAnimation(.snappy) { model.selectedVoice = preset.name }
                        }
                    }
                }
            }

            if voiceLibraryFilter == .all || voiceLibraryFilter == .personal {
                voiceSectionHeader(
                    "YOUR VOICEPRINTS",
                    detail: model.library.voices.isEmpty
                        ? "private · ready when you are"
                        : "\(filteredPersonalVoices.count) saved on this device"
                )
                if filteredPersonalVoices.isEmpty {
                    personalVoiceEmptyState
                } else {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 190), spacing: 12)],
                        spacing: 12
                    ) {
                        ForEach(filteredPersonalVoices) { voice in
                            EnrolledVoiceTile(
                                voice: voice,
                                selected: model.enrolledSelection() == voice.id,
                                select: { model.selectedVoice = "voice:\(voice.id.uuidString)" },
                                rename: { renameTarget = voice },
                                reRecord: { openEnrollment(target: voice.id) },
                                share: { cardVoice = voice },
                                delete: {
                                    model.library.delete(id: voice.id)
                                    if model.enrolledSelection() == voice.id {
                                        model.selectedVoice = "matt"
                                    }
                                }
                            )
                        }
                    }
                    .animation(.snappy, value: model.library.voices)
                }
            }

            if filteredPresetVoices.isEmpty, filteredPersonalVoices.isEmpty {
                Label("No voices match that search", systemImage: "waveform.badge.magnifyingglass")
                    .font(.system(size: Lab.typeSize(14), weight: .bold))
                    .foregroundStyle(Lab.textSecondary)
                    .frame(maxWidth: .infinity, minHeight: 110)
                    .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 18))
            }

            PhotosPicker(
                selection: $importItem, matching: .images, photoLibrary: .shared()
            ) {
                HStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12).fill(Lab.cyan.opacity(0.12))
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 22, weight: .bold))
                            .foregroundStyle(Lab.cyan)
                    }
                    .frame(width: 46, height: 46)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Import a voice card")
                            .font(.system(size: Lab.typeSize(15), weight: .black))
                            .foregroundStyle(Lab.textPrimary)
                        Text("The picture itself carries the private 4 KB voiceprint")
                            .font(.system(size: Lab.typeSize(11), weight: .medium))
                            .foregroundStyle(Lab.textSecondary)
                            .multilineTextAlignment(.leading)
                    }
                    Spacer()
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(Lab.cyan)
                }
                .padding(14)
                .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 17))
                .overlay(RoundedRectangle(cornerRadius: 17).stroke(Lab.cyan.opacity(0.28)))
            }
            .accessibilityHint("Pick a voice card someone sent you; the voice joins your library")

            Label(
                "Everything here stays on this device. Recordings are discarded after the 4 KB voiceprint exists. Clone or import only voices you have the right to use.",
                systemImage: "lock.shield.fill"
            )
            .font(.system(size: Lab.typeSize(11), weight: .medium))
            .foregroundStyle(Lab.textSecondary)
            .padding(.horizontal, 4)
            .padding(.bottom, 16)
        }
        .frame(maxWidth: 1_180, alignment: .leading)
        .frame(maxWidth: .infinity)
    }

    private var filteredPresetVoices: [Preset] {
        let query = voiceSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return model.presets.filter { preset in
            voiceLibraryFilter.includes(preset)
                && (query.isEmpty
                    || preset.name.localizedCaseInsensitiveContains(query)
                    || preset.character.localizedCaseInsensitiveContains(query))
        }
    }

    private var filteredPersonalVoices: [EnrolledVoice] {
        guard voiceLibraryFilter == .all || voiceLibraryFilter == .personal else { return [] }
        let query = voiceSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return model.library.voices.filter { query.isEmpty || $0.name.localizedCaseInsensitiveContains(query) }
    }

    private var selectedVoiceCharacter: String {
        if model.enrolledSelection() != nil { return "Your private, locally cloned voiceprint" }
        return model.presets.first(where: { $0.name == model.selectedVoice })?.character
            ?? "Ready to become the voice of your next utterance"
    }

    private var voiceArchiveHero: some View {
        ZStack(alignment: .trailing) {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Lab.emeraldDeep.opacity(0.78), Lab.panelStrong, Lab.cyan.opacity(0.08)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(RoundedRectangle(cornerRadius: 24).stroke(Lab.emerald.opacity(0.38)))

            VoiceLibraryHalo()
                .frame(width: 240)
                .opacity(0.8)
                .allowsHitTesting(false)

            HStack(spacing: 18) {
                VoiceOrb(name: model.currentVoiceLabel, selected: true)
                    .scaleEffect(1.65)
                    .frame(width: 74, height: 74)
                VStack(alignment: .leading, spacing: 5) {
                    Text("\(model.presets.count) BUILT-IN VOICES · \(model.library.voices.count) YOURS")
                        .font(.system(size: Lab.typeSize(9), weight: .black, design: .monospaced))
                        .kerning(1.2)
                        .foregroundStyle(Lab.emerald)
                    Text(model.currentVoiceLabel.capitalized)
                        .font(.system(
                            size: Lab.typeSize(voiceArchiveNamePointSize),
                            weight: .black,
                            design: .rounded
                        ))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)
                        .allowsTightening(true)
                        .layoutPriority(1)
                    Text(selectedVoiceCharacter)
                        .font(.system(size: Lab.typeSize(12), weight: .medium))
                        .foregroundStyle(Lab.textPrimary.opacity(0.78))
                        .fixedSize(horizontal: false, vertical: true)
                    Text("CURRENT SPECIMEN")
                        .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                        .foregroundStyle(Lab.cyan)
                }
                Spacer(minLength: 0)
            }
            .padding(20)
        }
        .frame(minHeight: 170)
        .shadow(color: Lab.emerald.opacity(0.15), radius: 24, y: 10)
    }

    /// Keep the hero name complete without letting an intrinsically wide Text
    /// enlarge the containing ScrollView. The coefficient is tuned to the space
    /// remaining beside the 74-point orb on a 390-point compact screen.
    private var voiceArchiveNamePointSize: CGFloat {
        let glyphCount = max(1, model.currentVoiceLabel.count)
        return min(28, max(12, 420 / CGFloat(glyphCount)))
    }

    private var voiceSearchField: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(Lab.emerald)
            TextField("Search by name or character", text: $voiceSearchText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .foregroundStyle(Lab.textPrimary)
            if !voiceSearchText.isEmpty {
                Button { voiceSearchText = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(Lab.textSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 50)
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 15))
        .overlay(RoundedRectangle(cornerRadius: 15).stroke(Lab.stroke))
    }

    private var voiceFilterBar: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(VoiceLibraryFilter.allCases) { filter in
                        Button {
                            withAnimation(.snappy) { voiceLibraryFilter = filter }
                        } label: {
                            Label(filter.rawValue, systemImage: filter.symbol)
                                .font(.system(size: Lab.typeSize(11), weight: .bold))
                                .foregroundStyle(voiceLibraryFilter == filter ? Color.black : Lab.textPrimary)
                                .padding(.horizontal, 13)
                                .frame(height: 38)
                                .background(
                                    voiceLibraryFilter == filter ? Lab.emerald : Color.black.opacity(0.38),
                                    in: Capsule()
                                )
                                .overlay(Capsule().stroke(voiceLibraryFilter == filter ? .clear : Lab.stroke))
                        }
                        .buttonStyle(.plain)
                        .id(filter)
                    }
                }
                .padding(.vertical, 2)
            }
            .scrollClipDisabled()
            .onAppear {
                proxy.scrollTo(voiceLibraryFilter, anchor: .trailing)
            }
            .onChange(of: voiceLibraryFilter) { _, filter in
                withAnimation(.snappy) { proxy.scrollTo(filter, anchor: .center) }
            }
        }
    }

    private func voiceSectionHeader(_ title: String, detail: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            LabLabel(text: title)
            Spacer()
            Text(detail)
                .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
        }
    }

    private var personalVoiceEmptyState: some View {
        Button { openEnrollment(target: nil) } label: {
            HStack(spacing: 13) {
                Image(systemName: "person.crop.circle.badge.plus")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(Lab.emerald)
                VStack(alignment: .leading, spacing: 3) {
                    Text("No personal voiceprints yet")
                        .font(.system(size: Lab.typeSize(15), weight: .black))
                        .foregroundStyle(Lab.textPrimary)
                    Text("Thirty seconds of reading creates one—privately, on this device.")
                        .font(.system(size: Lab.typeSize(11), weight: .medium))
                        .foregroundStyle(Lab.textSecondary)
                        .multilineTextAlignment(.leading)
                }
                Spacer()
                Image(systemName: "arrow.right")
                    .fontWeight(.black)
                    .foregroundStyle(Lab.emerald)
            }
            .padding(15)
            .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 17))
            .overlay(RoundedRectangle(cornerRadius: 17).stroke(Lab.emerald.opacity(0.22)))
        }
        .buttonStyle(.plain)
    }

    private func utteranceCard(compact: Bool) -> some View {
        LabPanel {
            VStack(alignment: .leading, spacing: compact ? 9 : 12) {
                HStack(spacing: 8) {
                    LabLabel(text: "03 · The Utterance")
                    Spacer()
                    if isUtteranceFocused {
                        Button {
                            selectAllUtterance()
                        } label: {
                            Image(systemName: "text.badge.checkmark")
                        }
                        .buttonStyle(GhostButtonStyle())
                        .disabled(model.text.isEmpty)
                        .accessibilityLabel("Select all")
                        .accessibilityHint("Selects the entire utterance so typing replaces it")
                        Button {
                            model.clearUtterance()
                            focusUtteranceAfterCurrentTap()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                        .disabled(model.text.isEmpty)
                        .accessibilityLabel("Clear")
                        .accessibilityHint("Removes the current utterance")
                        Button {
                            dismissKeyboard()
                        } label: {
                            Label("Done", systemImage: "checkmark")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        .accessibilityHint("Finishes editing and hides the keyboard")
                    } else {
                        Button("Select all") { selectAllUtterance() }
                            .buttonStyle(GhostButtonStyle())
                            .disabled(model.text.isEmpty)
                            .accessibilityHint("Selects the entire utterance so typing replaces it")
                        Button {
                            model.clearUtterance()
                            focusUtteranceAfterCurrentTap()
                        } label: {
                            Label("Clear", systemImage: "xmark.circle.fill")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                        .disabled(model.text.isEmpty)
                        .accessibilityHint("Removes the current utterance")
                    }
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
                    NativeUtteranceEditor(
                        text: Binding(
                            get: { model.text },
                            set: { model.updateUtteranceFromEditor($0) }
                        ),
                        isFocused: $isUtteranceFocused,
                        maximumLength: JokeLibrary.maximumUtteranceLength,
                        focusRequest: utteranceFocusRequest,
                        selectAllRequest: utteranceSelectAllRequest,
                        externalRevision: model.utteranceExternalRevision
                    )
                }
                // More vertical room makes drag handles and the magnifier
                // practical on a phone instead of fighting a three-line box.
                // Never let keyboard-driven relayout stretch the empty editor into
                // a large black void. Long utterances scroll inside this native field.
                .frame(height: isUtteranceFocused ? 108 : (compact ? 104 : 138))
                .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(
                            isUtteranceFocused
                                ? Lab.emerald.opacity(0.62) : Color.clear,
                            lineWidth: 1
                        )
                )
                HStack {
                    Text("\(model.text.count) / 50k")
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    Button {
                        model.replaceUtteranceFromUser(
                            with: JokeLibrary.random(excluding: model.text)
                        )
                        dismissKeyboard()
                    } label: {
                        Image(systemName: "dice.fill")
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.cyan))
                    .accessibilityLabel("Choose another random joke")
                    .accessibilityHint("Replaces the utterance with a different bundled joke")
                    Menu {
                        Button {
                            showDesktopImporter = true
                        } label: {
                            Label("Text, PDF, or voice card from Files", systemImage: "folder")
                        }
                        Button {
                            showURLImporter = true
                        } label: {
                            Label("Web page, text, or PDF URL", systemImage: "link")
                        }
                    } label: {
                        if model.isImportingText {
                            ProgressView().controlSize(.small).tint(Lab.emerald)
                        } else {
                            Image(systemName: "square.and.arrow.down")
                        }
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                    .disabled(model.isImportingText)
                    .accessibilityLabel("Import utterance")
                    .accessibilityHint("Imports text from Files or an HTTPS URL")
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
                    .font(.system(size: Lab.typeSize(12), design: .monospaced))
                    .foregroundStyle(Lab.textPrimary)
                    .frame(width: 74)
                    .padding(.vertical, 5)
                    .padding(.horizontal, 8)
                    .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel("Seed")
                    Button {
                        model.seed = UInt64.random(in: 0..<100_000)
                    } label: {
                        Image(systemName: "dice")
                    }
                    .buttonStyle(GhostButtonStyle())
                    .accessibilityLabel("Randomize seed")
                }
                HStack(spacing: 10) {
                    Button {
                        dismissKeyboard()
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
                            .lineLimit(1)
                            .minimumScaleFactor(0.72)
                        }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(!model.canSynthesizeFromCommand)

                    Button {
                        dismissKeyboard()
                        showVoiceComparison = true
                    } label: {
                        Label("Voice Lab", systemImage: "person.3.sequence.fill")
                            .lineLimit(1)
                            .minimumScaleFactor(0.72)
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.cyan))
                    .disabled(!model.canCompareVoices)
                    .accessibilityHint("Speaks the same excerpt with every available voice")
                }
                if model.isSynthesizing {
                    GalvanicVoiceForge(
                        telemetry: model.forge,
                        elapsed: model.synthesisSeconds,
                        estimatedRemainingSeconds: model.estimatedRemainingSeconds,
                        chunkIndex: model.synthesisChunkIndex,
                        chunkCount: model.synthesisChunkCount,
                        compact: compact,
                        cancel: model.isSynthesizing ? { model.cancelSynthesis() } : nil
                    )
                    .transition(.opacity.combined(with: .scale(scale: 0.985, anchor: .top)))
                }
                if let audio = model.lastAudio {
                    PlaybackSignalView(
                        samples: audio,
                        player: model.player,
                        analysisID: model.wavUrl?.lastPathComponent ?? "\(audio.count)",
                        refreshToken: playbackTick
                    ) { progress in
                        model.seekPlayback(to: progress)
                    }
                        .frame(height: compact ? 112 : 148)
                    HStack(spacing: 10) {
                        Button {
                            model.togglePlayback()
                            playbackTick += 1
                        } label: {
                            Image(
                                systemName: model.player?.isPlaying == true
                                    ? "pause.fill" : "play.fill")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        .accessibilityLabel(
                            model.player?.isPlaying == true ? "Pause" : "Play")
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
        .id("utterance-card")
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
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 9))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "On-device synthesis profile. \(profile.teamPartitions) workers, "
                + "\(profile.frames) frames, total \(Int(profile.totalMs)) milliseconds."
        )
    }

    private func selectAllUtterance() {
        DispatchQueue.main.async {
            isUtteranceFocused = true
            utteranceFocusRequest &+= 1
            utteranceSelectAllRequest &+= 1
        }
    }

    private func focusUtteranceAfterCurrentTap() {
        DispatchQueue.main.async {
            isUtteranceFocused = true
            utteranceFocusRequest &+= 1
        }
    }

    private func dismissKeyboard() {
        isUtteranceFocused = false
        focusedField = nil
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
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
            let debugVoice = EnrolledVoice(id: UUID(), name: "Jeff", vector: vector)
            showVoiceLab = true
            Task { @MainActor in
                // The library cover must finish presenting before it can own the
                // nested card sheet.
                try? await Task.sleep(for: .milliseconds(250))
                cardVoice = debugVoice
            }
            // Round-trip the REAL composed PNG through both import paths: the raw
            // bytes exercise the lossless chunk; a re-encode via UIImage strips the
            // private chunk, forcing the pixel decoder against the exact pixels
            // ImageRenderer produced (which the off-device harness only approximates).
            Task {
                guard let png = try? await VoicePrintCard.pngData(name: "Jeff", vector: vector)
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

    /// Layout harness for the narrowest supported phone. It does not touch the
    /// persisted voice archive, so repeatedly running the screenshot test cannot
    /// leave synthetic voices behind in the simulator or a developer build.
    private func debugLongVoiceNameHook() {
        #if DEBUG
            guard ProcessInfo.processInfo.environment["FTTS_DEBUG_LONG_VOICE_NAME"] == "1"
            else { return }
            let voice = EnrolledVoice(
                id: UUID(uuidString: "A11CE000-0000-4000-8000-000000000001")!,
                name: "Alexandria-Cassandra Nightingale",
                vector: Array(repeating: 0, count: Engine.speakerWidth)
            )
            model.library.installDebugVoices([voice])
            model.selectedVoice = "voice:\(voice.id.uuidString)"
            voiceLibraryFilter = .personal
            showVoiceLab = true
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
            let started = ProcessInfo.processInfo.systemUptime
            NSLog("FTTS_DEBUG_VIDEO starting: \(pcm.count) samples")
            Task {
                do {
                    let out = try await MediaExporter.exportVideo(
                        pcm: pcm, voiceLabel: "debug", wavUrl: url
                    ) { fraction in
                        let percent = Int(fraction * 100)
                        if percent % 10 == 0 {
                            NSLog("FTTS_DEBUG_VIDEO %d%% at %.1fs", percent,
                                ProcessInfo.processInfo.systemUptime - started)
                        }
                    }
                    NSLog("FTTS_DEBUG_VIDEO done in %.1fs: %@",
                        ProcessInfo.processInfo.systemUptime - started, out.path)
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
        .background(Lab.panelSoft, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// Enrollment is the app's signature capability, so it gets a real call to
/// action instead of masquerading as one more preset in the voice grid.
private struct VoiceEnrollmentCallout: View {
    let isReady: Bool
    let compact: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ViewThatFits(in: .horizontal) {
                calloutLabel(showActionTitle: true)
                calloutLabel(showActionTitle: false)
            }
            .padding(.horizontal, compact ? 13 : 16)
            .padding(.vertical, compact ? 11 : 14)
            .frame(maxWidth: .infinity, minHeight: compact ? 72 : 84, alignment: .leading)
            .background {
                LinearGradient(
                    colors: [
                        Lab.emeraldDeep.opacity(0.90),
                        Lab.emerald.opacity(0.13),
                        Lab.cyan.opacity(0.055),
                        Lab.panelStrong
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
            // Force the complete composited button through one final mask. A
            // blurred sublayer here previously escaped the rounded outline on
            // physical iPhones even though the background itself was clipped.
            .compositingGroup()
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [Lab.emerald.opacity(0.9), Lab.cyan.opacity(0.34), Lab.emerald.opacity(0.18)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1.25
                    )
            }
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(VoiceEnrollmentCalloutStyle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Create your own voice")
        .accessibilityValue(
            isReady
                ? "Ready. Record about thirty seconds. Processing stays on this device."
                : "Voice model setup required first."
        )
        .accessibilityHint(isReady ? "Opens the private voice recorder" : "Opens model setup")
    }

    private func calloutLabel(showActionTitle: Bool) -> some View {
        HStack(spacing: compact ? 11 : 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.black.opacity(0.34))
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.13), lineWidth: 1)
                Image(systemName: "waveform.badge.plus")
                    .font(.system(size: Lab.typeSize(compact ? 19 : 22), weight: .bold))
                    .foregroundStyle(.white)
                    .symbolRenderingMode(.hierarchical)
            }
            .frame(width: compact ? 46 : 52, height: compact ? 46 : 52)
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                Text("SIGNATURE FEATURE")
                    .font(.system(size: Lab.typeSize(compact ? 8 : 9), weight: .black, design: .monospaced))
                    .kerning(1.35)
                    .foregroundStyle(Lab.emerald)
                    .lineLimit(1)
                Text("Create your own voice")
                    .font(.system(size: Lab.typeSize(compact ? 15 : 17), weight: .black))
                    .foregroundStyle(.white)
                    .lineLimit(compact ? 2 : 1)
                    .minimumScaleFactor(0.78)
                Text(
                    isReady
                        ? "Read for about 30 seconds · private and on-device"
                        : "Finish model setup, then record privately on-device"
                )
                .font(.system(size: Lab.typeSize(compact ? 10 : 11), weight: .medium))
                .foregroundStyle(Lab.textPrimary.opacity(0.78))
                // The dashboard sidebar is narrower than an iPhone. Keep the
                // headline intact and let this supporting promise wrap instead.
                .lineLimit(2)
                .minimumScaleFactor(0.78)
            }

            Spacer(minLength: 4)

            if showActionTitle {
                HStack(spacing: 5) {
                    Text(isReady ? "Start" : "Set up")
                    Image(systemName: "arrow.right")
                }
                .font(.system(size: Lab.typeSize(10), weight: .black, design: .monospaced))
                .textCase(.uppercase)
                .foregroundStyle(Color.black.opacity(0.82))
                .padding(.horizontal, 11)
                .frame(minHeight: 34)
                .background(Lab.emerald, in: Capsule())
            } else {
                Image(systemName: "arrow.right.circle.fill")
                    .font(.system(size: Lab.typeSize(24), weight: .bold))
                    .foregroundStyle(Lab.emerald)
            }
        }
    }
}

private struct VoiceEnrollmentCalloutStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.988 : 1)
            .brightness(configuration.isPressed ? -0.04 : 0)
            .animation(.easeOut(duration: 0.14), value: configuration.isPressed)
            .hoverEffect(.highlight)
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

private struct VoiceLibraryHalo: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 24.0, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let center = CGPoint(x: size.width * 0.58, y: size.height / 2)
                let time = timeline.date.timeIntervalSinceReferenceDate
                for ring in 0..<4 {
                    let radius = CGFloat(30 + ring * 22)
                    let circle = CGRect(
                        x: center.x - radius,
                        y: center.y - radius,
                        width: radius * 2,
                        height: radius * 2
                    )
                    let tint = ring.isMultiple(of: 2) ? Lab.emerald : Lab.cyan
                    context.stroke(
                        Path(ellipseIn: circle),
                        with: .color(tint.opacity(0.1)),
                        lineWidth: 1
                    )
                }
                for index in 0..<7 {
                    let angle = Double(index) / 7 * .pi * 2 + time * 0.12
                    let orbit = CGFloat(index.isMultiple(of: 2) ? 92 : 68)
                    let point = CGPoint(
                        x: center.x + CGFloat(cos(angle)) * orbit,
                        y: center.y + CGFloat(sin(angle)) * orbit
                    )
                    let tint = index.isMultiple(of: 2) ? Lab.emerald : Lab.cyan
                    let node = CGRect(x: point.x - 4, y: point.y - 4, width: 8, height: 8)
                    context.fill(Path(ellipseIn: node), with: .color(tint.opacity(0.55)))
                }
            }
        }
        .mask(LinearGradient(colors: [.clear, .white], startPoint: .leading, endPoint: .trailing))
        .accessibilityHidden(true)
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
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    VoiceOrb(name: name, selected: selected)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(name.capitalized)
                            .font(.system(size: Lab.typeSize(16), weight: .black))
                            .foregroundStyle(Lab.textPrimary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.62)
                            .allowsTightening(true)
                            .layoutPriority(1)
                        Text(character.localizedCaseInsensitiveContains("feminine") ? "FEMININE" : "MASCULINE")
                            .font(.system(size: Lab.typeSize(7), weight: .black, design: .monospaced))
                            .kerning(1)
                            .foregroundStyle(selected ? Lab.emerald : Lab.textSecondary)
                    }
                    Spacer()
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 19, weight: .bold))
                        .foregroundStyle(selected ? Lab.emerald : Lab.textSecondary.opacity(0.38))
                }
                Text(character)
                    .font(.system(size: Lab.typeSize(11), weight: .medium))
                    .foregroundStyle(Lab.textSecondary)
                    .multilineTextAlignment(.leading)
                    .frame(minHeight: 32, alignment: .top)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Text(selected ? "SELECTED" : "TAP TO SELECT")
                        .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                    Spacer()
                    Image(systemName: "waveform")
                }
                .foregroundStyle(selected ? Lab.emerald : Lab.textSecondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 154, alignment: .leading)
            .background(
                LinearGradient(
                    colors: [
                        selected ? Lab.emerald.opacity(0.13) : Lab.panelStrong,
                        Lab.panel
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 17, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 17)
                    .stroke(
                        selected
                            ? Lab.emerald
                            : (accent ? Lab.emerald.opacity(0.35) : Lab.stroke),
                        lineWidth: selected ? 1.5 : 1)
            )
        }
        .buttonStyle(.plain)
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
        VStack(alignment: .leading, spacing: 9) {
            Button(action: select) {
                HStack(spacing: 10) {
                    VoiceOrb(name: voice.name, selected: selected)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(voice.name)
                            .font(.system(size: Lab.typeSize(15), weight: .black))
                            .foregroundStyle(Lab.textPrimary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.62)
                            .allowsTightening(true)
                            .layoutPriority(1)
                        Text("PRIVATE VOICEPRINT")
                            .font(.system(size: Lab.typeSize(7), weight: .black, design: .monospaced))
                            .foregroundStyle(Lab.emerald)
                    }
                    Spacer()
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selected ? Lab.emerald : Lab.textSecondary.opacity(0.35))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
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
        .padding(13)
        .background(
            selected ? Lab.emerald.opacity(0.11) : Lab.panelStrong,
            in: RoundedRectangle(cornerRadius: 17)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 17)
                .stroke(
                    selected ? Lab.emerald : Lab.emerald.opacity(0.35),
                    lineWidth: selected ? 1.5 : 1))
    }
}

struct EnrollmentSheet: View {
    @Bindable var model: LabModel
    @Environment(\.dismiss) private var dismiss
    @State private var cloneName: String

    init(model: LabModel) {
        self.model = model
        if let target = model.enrollmentTarget,
           let voice = model.library.voice(id: target)
        {
            _cloneName = State(initialValue: voice.name)
        } else {
            _cloneName = State(initialValue: model.library.suggestedEnrollmentName())
        }
    }

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
                .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 12))
                TextField("name your voice", text: $cloneName)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 10))
                    .foregroundStyle(Lab.textPrimary)
                    // Locked once recording starts: the name is required to save, and
                    // clearing it mid-read would cost the whole take at auto-stop.
                    .disabled(model.isEnrolling || model.recorder.isRecording)
                    .accessibilityIdentifier("enrollment-name")
                if model.recorder.isRecording {
                    // The live meter is the tell that the microphone is actually hearing
                    // you; a silent bar during the script means stop and fix it now, not
                    // after a minute of reading.
                    HStack(spacing: 10) {
                        Image(systemName: "waveform")
                            .foregroundStyle(Lab.emerald)
                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Lab.panelSoft)
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
                            // Starting a recording must never be a silent no-op. If the
                            // user cleared the suggested label, restore a unique one;
                            // it remains editable until the microphone actually starts.
                            if cloneName.trimmingCharacters(in: .whitespacesAndNewlines)
                                .isEmpty
                            {
                                cloneName = model.library.suggestedEnrollmentName()
                            }
                            do {
                                try model.recorder.start()
                            } catch {
                                model.lastError = error.localizedDescription
                            }
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .accessibilityIdentifier("enrollment-start-recording")
                        .accessibilityHint(
                            "Starts recording immediately. You can edit the suggested voice name first."
                        )
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
