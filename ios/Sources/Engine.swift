// The Rust engine behind an actor: one handle, strictly serialized access, exactly the
// concurrency contract crates/ftts-ffi documents.

import FttsCore
import Foundation

struct Preset: Identifiable, Decodable {
    let name: String
    let character: String
    var id: String { name }
}

struct SynthesisProfile: Decodable {
    let totalMs: Double
    let generationMs: Double
    let prefillMs: Double
    let microdecoderMs: Double
    let feedbackMs: Double
    let talkerMs: Double
    let codecActiveMs: Double
    let frames: UInt64
    let teamPartitions: Int

    enum CodingKeys: String, CodingKey {
        case totalMs = "total_ms"
        case generationMs = "generation_ms"
        case prefillMs = "prefill_ms"
        case microdecoderMs = "microdecoder_ms"
        case feedbackMs = "feedback_ms"
        case talkerMs = "talker_ms"
        case codecActiveMs = "codec_active_ms"
        case frames
        case teamPartitions = "team_partitions"
    }

    var otherGenerationMs: Double {
        max(0, generationMs - prefillMs - microdecoderMs - feedbackMs - talkerMs)
    }

    func adding(_ other: SynthesisProfile) -> SynthesisProfile {
        SynthesisProfile(
            totalMs: totalMs + other.totalMs,
            generationMs: generationMs + other.generationMs,
            prefillMs: prefillMs + other.prefillMs,
            microdecoderMs: microdecoderMs + other.microdecoderMs,
            feedbackMs: feedbackMs + other.feedbackMs,
            talkerMs: talkerMs + other.talkerMs,
            codecActiveMs: codecActiveMs + other.codecActiveMs,
            frames: frames + other.frames,
            teamPartitions: max(teamPartitions, other.teamPartitions)
        )
    }
}

struct SynthesisOutput {
    let pcm: [Float]
    let profile: SynthesisProfile?
}

struct EngineProgress: Sendable, Equatable {
    enum Kind: UInt32, Sendable {
        case stageStarted = 1
        case stageFinished = 2
        case unit = 3
        case admission = 4
        case health = 5
    }

    enum Stage: UInt32, Sendable {
        case modelBundle = 1
        case modelWeights = 2
        case runtime = 3
        case synthesis = 4
        case text = 5
        case frames = 6
        case codec = 7
        case resourceAdmission = 8
        case health = 9

        var shortLabel: String {
            switch self {
            case .modelBundle: "Reading specimen map"
            case .modelWeights: "Hydrating neural tissue"
            case .runtime: "Charging the runtime"
            case .synthesis: "Forging the voice"
            case .text: "Binding the utterance"
            case .frames: "Growing semantic frames"
            case .codec: "Turning frames into sound"
            case .resourceAdmission: "Checking memory headroom"
            case .health: "Watching signal health"
            }
        }
    }

    static let totalIsUpperBound: UInt32 = 1
    static let invalidatesOutput: UInt32 = 2

    let kind: Kind
    let stage: Stage
    let flags: UInt32
    let current: UInt64
    let total: UInt64
    let detail: UInt64
    let elapsedMilliseconds: Double

    var hasEstimatedTotal: Bool { flags & Self.totalIsUpperBound != 0 }
    var outputInvalid: Bool { flags & Self.invalidatesOutput != 0 }

    init?(_ native: FttsProgressEvent) {
        guard native.abi_version == UInt32(FTTS_PROGRESS_ABI_VERSION),
              let kind = Kind(rawValue: native.kind),
              let stage = Stage(rawValue: native.stage)
        else { return nil }
        self.kind = kind
        self.stage = stage
        flags = native.flags
        current = native.current
        total = native.total
        detail = native.detail
        elapsedMilliseconds = native.elapsed_ms
    }
}

private final class ProgressCallbackBox: @unchecked Sendable {
    private let lock = NSLock()
    private var cancellationRequested = false
    let publish: @Sendable (EngineProgress) -> Void

    init(publish: @escaping @Sendable (EngineProgress) -> Void) {
        self.publish = publish
    }

    func requestCancellation() {
        lock.lock()
        cancellationRequested = true
        lock.unlock()
    }

    func callbackVerdict() -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        return cancellationRequested ? 1 : 0
    }
}

private final class EngineCancellationController: @unchecked Sendable {
    private let lock = NSLock()
    private weak var active: ProgressCallbackBox?

    func begin(_ box: ProgressCallbackBox) {
        lock.lock()
        active = box
        lock.unlock()
    }

    func end(_ box: ProgressCallbackBox) {
        lock.lock()
        if active === box { active = nil }
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        let box = active
        lock.unlock()
        box?.requestCancellation()
    }
}

private let nativeProgressCallback: @convention(c) (
    UnsafeMutableRawPointer?, UnsafePointer<FttsProgressEvent>?
) -> Int32 = { context, eventPointer in
    guard let context, let eventPointer else { return 0 }
    let box = Unmanaged<ProgressCallbackBox>.fromOpaque(context).takeUnretainedValue()
    if let progress = EngineProgress(eventPointer.pointee) {
        box.publish(progress)
    }
    return box.callbackVerdict()
}

enum EngineError: LocalizedError {
    case native(String)
    case cancelled
    var errorDescription: String? {
        switch self {
        case .native(let message): message
        case .cancelled: "operation cancelled"
        }
    }

    static func lastFromNative() -> EngineError {
        .native(String(cString: ftts_last_error_message()))
    }
}

/// All engine access lives here. The Rust handle is not thread-safe; an actor's
/// serialization is the whole safety argument, so no engine call may leave this type.
actor Engine {
    static let speakerWidth = Int(FTTS_SPEAKER_WIDTH)
    private var handle: OpaquePointer?
    nonisolated private let cancellationController = EngineCancellationController()

    static func presets() -> [Preset] {
        let json = String(cString: ftts_presets_json())
        return (try? JSONDecoder().decode([Preset].self, from: Data(json.utf8))) ?? []
    }

    static func presetVector(named name: String) throws -> [Float] {
        var out = [Float](repeating: 0, count: speakerWidth)
        let code = out.withUnsafeMutableBufferPointer { buffer in
            ftts_preset_vector(name, buffer.baseAddress)
        }
        guard code == 0 else { throw EngineError.lastFromNative() }
        return out
    }

    var isLoaded: Bool { handle != nil }

    /// Hydrates the model. Multi-second; callers show progress copy of their own.
    func load(
        modelDirectory: URL,
        onProgress: @escaping @Sendable (EngineProgress) -> Void = { _ in }
    ) throws {
        guard handle == nil else { return }
        let callback = ProgressCallbackBox(publish: onProgress)
        cancellationController.begin(callback)
        defer { cancellationController.end(callback) }
        let context = Unmanaged.passUnretained(callback).toOpaque()
        guard let opened = ftts_engine_open_with_progress(
            modelDirectory.path,
            nativeProgressCallback,
            context
        ) else {
            if callback.callbackVerdict() != 0 { throw EngineError.cancelled }
            throw EngineError.lastFromNative()
        }
        handle = opened
    }

    /// Drops the model, freeing its ~2.3 GB of heap. Safe to call at any idle moment.
    func unload() {
        if let handle {
            ftts_engine_close(handle)
        }
        handle = nil
    }

    func synthesize(
        text: String,
        speaker: [Float],
        seed: UInt64,
        onProgress: @escaping @Sendable (EngineProgress) -> Void = { _ in }
    ) throws -> SynthesisOutput {
        guard let handle else { throw EngineError.native("engine not loaded") }
        guard speaker.count == Self.speakerWidth else {
            throw EngineError.native("speaker vector has wrong width")
        }
        var pcm: UnsafeMutablePointer<Float>?
        var length = 0
        let callback = ProgressCallbackBox(publish: onProgress)
        cancellationController.begin(callback)
        defer { cancellationController.end(callback) }
        let context = Unmanaged.passUnretained(callback).toOpaque()
        let code = speaker.withUnsafeBufferPointer { buffer in
            ftts_synthesize_with_progress(
                handle,
                text,
                buffer.baseAddress,
                buffer.count,
                seed,
                nativeProgressCallback,
                context,
                &pcm,
                &length
            )
        }
        if code == FTTS_SYNTH_CANCELLED { throw EngineError.cancelled }
        guard code == 0, let pcm else { throw EngineError.lastFromNative() }
        defer { ftts_pcm_free(pcm, length) }
        let profileJSON = String(cString: ftts_last_synthesis_profile_json(handle))
        let profile = try? JSONDecoder().decode(
            SynthesisProfile.self,
            from: Data(profileJSON.utf8)
        )
        return SynthesisOutput(
            pcm: Array(UnsafeBufferPointer(start: pcm, count: length)),
            profile: profile
        )
    }

    /// Thread-safe and nonisolated so a main-actor Cancel button can signal a native
    /// synchronous call while the engine actor itself is occupied by that call.
    nonisolated func cancelCurrentWork() {
        cancellationController.cancel()
    }

    /// Whether the neural denoiser artifact is in the model directory — asked of the
    /// engine itself, the same check its enrollment pipeline makes.
    var denoiseAvailable: Bool {
        guard let handle else { return false }
        return ftts_denoise_available(handle) == 1
    }

    /// Runs the neural denoiser over mono 24 kHz PCM. Throws when unavailable or on
    /// failure; callers keep their original audio in that case.
    func denoise(pcm samples: [Float]) throws -> [Float] {
        guard let handle else { throw EngineError.native("engine not loaded") }
        var cleaned: UnsafeMutablePointer<Float>?
        let code = samples.withUnsafeBufferPointer { buffer in
            ftts_denoise(handle, buffer.baseAddress, buffer.count, &cleaned)
        }
        guard code == 0, let cleaned else { throw EngineError.lastFromNative() }
        defer { ftts_pcm_free(cleaned, samples.count) }
        return Array(UnsafeBufferPointer(start: cleaned, count: samples.count))
    }

    func enroll(pcm samples: [Float]) throws -> [Float] {
        guard let handle else { throw EngineError.native("engine not loaded") }
        var out = [Float](repeating: 0, count: Self.speakerWidth)
        let code = samples.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                ftts_enroll(handle, input.baseAddress, input.count, output.baseAddress)
            }
        }
        guard code == 0 else { throw EngineError.lastFromNative() }
        return out
    }

    deinit {
        if let handle {
            ftts_engine_close(handle)
        }
    }
}

/// Mono 16-bit 24 kHz RIFF/WAV, the CLI's own output layout, for sharing.
enum WavWriter {
    static let sampleRate = 24_000

    static func data(from samples: [Float]) -> Data {
        var data = Data(capacity: 44 + samples.count * 2)
        func chunk(_ tag: String) { data.append(contentsOf: Array(tag.utf8)) }
        func u32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        chunk("RIFF")
        u32(UInt32(36 + samples.count * 2))
        chunk("WAVE")
        chunk("fmt ")
        u32(16)
        u16(1) // PCM
        u16(1) // mono
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * 2)) // byte rate
        u16(2) // block align
        u16(16) // bits
        chunk("data")
        u32(UInt32(samples.count * 2))
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            u16(UInt16(bitPattern: Int16((clamped * 32767.0).rounded())))
        }
        return data
    }
}

/// Conservative, deterministic speech mastering for synthesized 24 kHz mono
/// audio. It corrects recording-dependent tonal extremes, normalizes active
/// speech rather than leading/trailing silence, and catches peaks without
/// changing the timing or sample count used by playback and exports.
enum SpeechMastering {
    private static let targetActiveRMS: Float = 0.112 // approximately -19 dBFS
    private static let limiterKnee: Float = 0.82
    private static let limiterCeiling: Float = 0.95

    static func process(
        _ input: [Float],
        sampleRate: Float = 24_000,
        maximumGain: Float = 24.0
    ) -> [Float] {
        guard input.count > 1, sampleRate > 0 else { return input }

        var peak: Float = 0
        for sample in input where sample.isFinite {
            peak = max(peak, abs(sample))
        }
        guard peak > 0.001 else { return input.map { $0.isFinite ? $0 : 0 } }

        // DC/rumble removal. The one-pole high-pass is intentionally below the
        // useful speech fundamental, so it removes recording bias without
        // thinning ordinary voices.
        let highPassPole = exp(-2 * Float.pi * 70 / sampleRate)
        var cleaned = [Float](repeating: 0, count: input.count)
        var previousInput: Float = 0
        var previousOutput: Float = 0
        for index in input.indices {
            let sample = input[index].isFinite ? input[index] : 0
            let output = sample - previousInput + highPassPole * previousOutput
            cleaned[index] = output
            previousInput = sample
            previousOutput = output
        }

        // Split into reconstructing low/mid/high bands. Analysis determines a
        // gentle, bounded correction; no band moves more than 2.5 dB, so this
        // reduces microphone coloration without replacing the voice's timbre.
        let lowCoefficient = onePoleCoefficient(cutoff: 220, sampleRate: sampleRate)
        let presenceCoefficient = onePoleCoefficient(cutoff: 3_400, sampleRate: sampleRate)
        var lowState: Float = 0
        var presenceState: Float = 0
        var lowEnergy: Double = 0
        var midEnergy: Double = 0
        var highEnergy: Double = 0
        let analysisGate = max(0.003, peak * 0.012)

        for index in cleaned.indices {
            let sample = cleaned[index]
            lowState += lowCoefficient * (sample - lowState)
            presenceState += presenceCoefficient * (sample - presenceState)
            let low = lowState
            let high = sample - presenceState
            let mid = presenceState - low
            if abs(sample) >= analysisGate {
                lowEnergy += Double(low * low)
                midEnergy += Double(mid * mid)
                highEnergy += Double(high * high)
            }
        }

        let totalEnergy = lowEnergy + midEnergy + highEnergy
        let lowShare = totalEnergy > 0 ? Float(lowEnergy / totalEnergy) : 0.22
        let highShare = totalEnergy > 0 ? Float(highEnergy / totalEnergy) : 0.12
        let lowGain = decibelsToGain(clamp((0.22 - lowShare) * 10, -2.5, 2.0))
        let highGain = decibelsToGain(clamp((0.12 - highShare) * 12, -2.5, 2.0))

        var equalized = [Float](repeating: 0, count: cleaned.count)
        lowState = 0
        presenceState = 0
        for index in equalized.indices {
            let sample = cleaned[index]
            lowState += lowCoefficient * (sample - lowState)
            presenceState += presenceCoefficient * (sample - presenceState)
            let low = lowState
            let high = sample - presenceState
            let mid = presenceState - low
            equalized[index] = low * lowGain + mid + high * highGain
        }

        let activeRMS = gatedRMS(equalized, sampleRate: sampleRate)
        guard activeRMS > 0.000_1 else { return equalized }
        let loudnessGain = clamp(targetActiveRMS / activeRMS, 0.25, max(0.25, maximumGain))

        for index in equalized.indices {
            equalized[index] = softLimit(equalized[index] * loudnessGain)
        }
        let fadeSamples = min(equalized.count / 2, max(1, Int(sampleRate * 0.004)))
        if fadeSamples > 1 {
            for index in 0..<fadeSamples {
                let gain = Float(index) / Float(fadeSamples - 1)
                equalized[index] *= gain
                equalized[equalized.count - 1 - index] *= gain
            }
        }
        return equalized
    }

    private static func gatedRMS(_ samples: [Float], sampleRate: Float) -> Float {
        let window = max(1, Int(sampleRate * 0.020))
        var windowPowers: [Double] = []
        windowPowers.reserveCapacity((samples.count + window - 1) / window)
        var index = 0
        while index < samples.count {
            let end = min(samples.count, index + window)
            var power: Double = 0
            for sample in samples[index..<end] { power += Double(sample * sample) }
            windowPowers.append(power / Double(end - index))
            index = end
        }
        guard let maximum = windowPowers.max(), maximum > 0 else { return 0 }
        let gate = max(0.000_01, maximum * 0.01) // -50 dBFS or 20 dB below this clip's peak window
        let active = windowPowers.filter { $0 >= gate }
        guard !active.isEmpty else { return 0 }
        return Float(sqrt(active.reduce(0, +) / Double(active.count)))
    }

    private static func onePoleCoefficient(cutoff: Float, sampleRate: Float) -> Float {
        1 - exp(-2 * Float.pi * cutoff / sampleRate)
    }

    private static func decibelsToGain(_ decibels: Float) -> Float {
        pow(10, decibels / 20)
    }

    private static func softLimit(_ sample: Float) -> Float {
        let magnitude = abs(sample)
        guard magnitude > limiterKnee else { return sample }
        let span = limiterCeiling - limiterKnee
        let limited = limiterKnee + span * (1 - exp(-(magnitude - limiterKnee) / span))
        return sample.sign == .minus ? -limited : limited
    }

    private static func clamp(_ value: Float, _ lower: Float, _ upper: Float) -> Float {
        min(upper, max(lower, value))
    }
}
