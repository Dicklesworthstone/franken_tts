// Enrollment capture: microphone → mono Float32 at 24 kHz, the exact PCM the speaker
// encoder consumes. The recording lives only in memory and is handed to the engine,
// matching the app's microphone permission copy.
//
// Threading: the AVAudioEngine tap fires on a realtime audio thread, which must never
// block on the main thread. Conversion and accumulation therefore happen inside a small
// lock-protected sink the tap owns; the UI observes only the main-actor fields.

import AVFoundation
import Foundation

/// Owned by the audio tap: converts to 24 kHz mono and accumulates under a lock.
private final class CaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    let converter: AVAudioConverter
    let format: AVAudioFormat

    init?(from input: AVAudioFormat) {
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: AudioRecorder.targetRate,
                channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: input, to: target)
        else { return nil }
        self.converter = converter
        self.format = target
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            return
        }
        var served = false
        _ = converter.convert(to: out, error: nil) { _, status in
            if served {
                status.pointee = .noDataNow
                return nil
            }
            served = true
            status.pointee = .haveData
            return buffer
        }
        guard let channel = out.floatChannelData?[0], out.frameLength > 0 else { return }
        let chunk = UnsafeBufferPointer(start: channel, count: Int(out.frameLength))
        lock.lock()
        samples.append(contentsOf: chunk)
        lock.unlock()
    }

    func drain() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let out = samples
        samples = []
        return out
    }
}

@MainActor
@Observable
final class AudioRecorder {
    static let targetRate: Double = 24_000

    var isRecording = false
    var seconds: Double = 0

    private let audioEngine = AVAudioEngine()
    private var sink: CaptureSink?
    private var timer: Timer?

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true)

        let input = audioEngine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let sink = CaptureSink(from: inputFormat) else {
            throw EngineError.native("cannot build the 24 kHz capture pipeline")
        }
        self.sink = sink
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            sink.consume(buffer)
        }
        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true
        seconds = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.seconds += 0.25 }
        }
    }

    /// Stops and returns the captured 24 kHz mono PCM.
    func stop() -> [Float] {
        timer?.invalidate()
        timer = nil
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRecording = false
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        let samples = sink?.drain() ?? []
        sink = nil
        return samples
    }
}
