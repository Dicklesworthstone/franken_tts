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
}

struct SynthesisOutput {
    let pcm: [Float]
    let profile: SynthesisProfile?
}

enum EngineError: LocalizedError {
    case native(String)
    var errorDescription: String? {
        if case .native(let message) = self { return message }
        return nil
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
    func load(modelDirectory: URL) throws {
        guard handle == nil else { return }
        guard let opened = ftts_engine_open(modelDirectory.path) else {
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

    func synthesize(text: String, speaker: [Float], seed: UInt64) throws -> SynthesisOutput {
        guard let handle else { throw EngineError.native("engine not loaded") }
        guard speaker.count == Self.speakerWidth else {
            throw EngineError.native("speaker vector has wrong width")
        }
        var pcm: UnsafeMutablePointer<Float>?
        var length = 0
        let code = speaker.withUnsafeBufferPointer { buffer in
            ftts_synthesize(handle, text, buffer.baseAddress, buffer.count, seed, &pcm, &length)
        }
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
