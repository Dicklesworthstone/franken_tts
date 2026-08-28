// Share formats: M4A audio (small, the default) and the branded MP4 video.
//
// The video frames come from the same Rust renderer behind `ftts make-video`, through
// the FFI, so the phone's clip is pixel-identical to the desktop's; the encode differs
// by platform on purpose — ffmpeg on desktop, the hardware H.264/AAC encoders here.

import AVFoundation
import FttsCore
import Foundation

/// The audio side of one AVAssetWriter session.
///
/// AVFoundation's writer/reader classes predate Swift concurrency and are not
/// annotated Sendable, even though a writer is explicitly designed to accept
/// its independent media inputs concurrently. This context has one owner and
/// confines the reader, reader output, and audio input to exactly one async
/// feed. The video path shares only the writer's documented thread-safe status
/// and error reporting, so the unchecked conformance records that invariant in
/// one audited place instead of scattering warning suppressions through the
/// export loop.
private final class AudioFeedContext: @unchecked Sendable {
    private let reader: AVAssetReader
    private let readerOutput: AVAssetReaderTrackOutput
    private let audioInput: AVAssetWriterInput
    private let writer: AVAssetWriter

    init(
        reader: AVAssetReader,
        readerOutput: AVAssetReaderTrackOutput,
        audioInput: AVAssetWriterInput,
        writer: AVAssetWriter
    ) {
        self.reader = reader
        self.readerOutput = readerOutput
        self.audioInput = audioInput
        self.writer = writer
    }

    func feed() async throws {
        while let sample = readerOutput.copyNextSampleBuffer() {
            while !audioInput.isReadyForMoreMediaData {
                guard writer.status == .writing else {
                    throw writer.error
                        ?? EngineError.native("video writer stopped while waiting for audio")
                }
                try await Task.sleep(for: .milliseconds(4))
            }
            guard audioInput.append(sample) else {
                throw writer.error ?? EngineError.native("appending audio failed")
            }
        }
        audioInput.markAsFinished()
        // nil from copyNextSampleBuffer means EITHER end-of-track or failure;
        // only the status separates a finished read from truncated audio.
        if reader.status == .failed {
            throw reader.error
                ?? EngineError.native("reading the share WAV failed midway")
        }
    }
}

enum MediaExporter {
    /// Transcode the played WAV into an AAC .m4a (an order of magnitude smaller).
    ///
    /// Outputs get unique names: a fixed name would let a new synthesis overwrite a file
    /// an in-flight share sheet still references.
    static func exportM4A(fromWav wavUrl: URL) async throws -> URL {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("franken_tts-\(ProcessInfo.processInfo.globallyUniqueString).m4a")
        let asset = AVURLAsset(url: wavUrl)
        guard
            let session = AVAssetExportSession(
                asset: asset, presetName: AVAssetExportPresetAppleM4A)
        else { throw EngineError.native("cannot create the M4A export session") }
        session.outputFileType = .m4a
        session.outputURL = output
        await session.export()
        if let error = session.error { throw error }
        return output
    }

    /// Render and encode the branded share video. `progress` is 0...1 on any thread.
    static func exportVideo(
        pcm: [Float], voiceLabel: String, wavUrl: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "franken_tts-\(ProcessInfo.processInfo.globallyUniqueString).mp4")

        let width = Int(ftts_video_width())
        let height = Int(ftts_video_height())
        let fps = Int32(ftts_video_fps())

        let renderer: OpaquePointer? = pcm.withUnsafeBufferPointer { buffer in
            ftts_video_open(buffer.baseAddress, buffer.count, 24_000, voiceLabel)
        }
        guard let renderer else { throw EngineError.lastFromNative() }
        defer { ftts_video_close(renderer) }
        let frames = ftts_video_frame_count(renderer)
        guard width > 0, height > 0, fps > 0, frames > 0 else {
            throw EngineError.native("video renderer returned invalid dimensions or no frames")
        }

        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
            ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        // No explicit bitrate: mono AAC at 24 kHz rejects 96 kbps outright (the
        // encoder's per-channel ceiling at that sample rate is lower), and error
        // -11861 from that mismatch killed the whole export. The encoder's own
        // default for this format is well within range and sounds fine for speech.
        let audioInput = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 24_000,
                AVNumberOfChannelsKey: 1,
            ])
        audioInput.expectsMediaDataInRealTime = false
        writer.add(videoInput)
        writer.add(audioInput)
        guard writer.startWriting() else {
            throw writer.error ?? EngineError.native("video writer failed to start")
        }
        writer.startSession(atSourceTime: .zero)

        // Audio reader, set up before feeding starts. One asset instance throughout: a
        // reader can only consume tracks belonging to the exact asset it was created
        // over, not an equal asset at the same URL.
        let audioAsset = AVURLAsset(url: wavUrl)
        let reader = try AVAssetReader(asset: audioAsset)
        guard let track = try await audioAsset.loadTracks(withMediaType: .audio).first
        else { throw EngineError.native("share WAV has no audio track") }
        let readerOutput = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [AVFormatIDKey: kAudioFormatLinearPCM])
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw reader.error ?? EngineError.native("cannot read the share WAV")
        }

        // Audio and video MUST feed concurrently: with two inputs the writer
        // interleaves media, so after buffering a fraction of a second of video it
        // stops accepting more until audio for those timestamps arrives. Feeding all
        // video first deadlocks at the buffer depth — the "stuck at 19%" bug.
        do {
            let audioFeed = AudioFeedContext(
                reader: reader,
                readerOutput: readerOutput,
                audioInput: audioInput,
                writer: writer
            )
            async let audioDone: Void = audioFeed.feed()

            // The Rust renderer is a pure function of (frame, immutable state), so a
            // chunk of frames renders in parallel across cores — straight into BGRA
            // (Rust does the swizzle; Swift only copies rows). Appends stay in order.
            let bgraStride = width * 4
            let window = 4
            var chunkStart = 0
            while chunkStart < frames {
                let chunk = Array(chunkStart..<min(chunkStart + window, frames))
                let rendered = try await withThrowingTaskGroup(
                    of: (Int, [UInt8]).self
                ) { group in
                    for frame in chunk {
                        group.addTask {
                            var bgra = [UInt8](repeating: 0, count: bgraStride * height)
                            let code = bgra.withUnsafeMutableBufferPointer { buffer in
                                ftts_video_render_frame_bgra(
                                    renderer, frame, buffer.baseAddress, bgraStride)
                            }
                            guard code == 0 else { throw EngineError.lastFromNative() }
                            return (frame, bgra)
                        }
                    }
                    var out = [Int: [UInt8]]()
                    for try await (frame, bgra) in group {
                        out[frame] = bgra
                    }
                    return out
                }
                for frame in chunk {
                    guard let bgra = rendered[frame] else {
                        throw EngineError.native("frame \(frame) went missing")
                    }
                    while !videoInput.isReadyForMoreMediaData {
                        guard writer.status == .writing else {
                            throw writer.error
                                ?? EngineError.native("video writer stopped while waiting for frames")
                        }
                        try await Task.sleep(for: .milliseconds(4))
                    }
                    guard let pool = adaptor.pixelBufferPool else {
                        throw EngineError.native("no pixel buffer pool")
                    }
                    var slot: CVPixelBuffer?
                    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &slot)
                    guard let pixelBuffer = slot else {
                        throw EngineError.native("cannot allocate a pixel buffer")
                    }
                    let lockStatus = CVPixelBufferLockBaseAddress(pixelBuffer, [])
                    guard lockStatus == kCVReturnSuccess else {
                        throw EngineError.native("cannot lock a video pixel buffer")
                    }
                    do {
                        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
                        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                            throw EngineError.native("video pixel buffer has no writable storage")
                        }
                        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
                        bgra.withUnsafeBufferPointer { source in
                            if stride == bgraStride {
                                base.copyMemory(
                                    from: source.baseAddress!, byteCount: bgraStride * height)
                            } else {
                                for row in 0..<height {
                                    (base + row * stride).copyMemory(
                                        from: source.baseAddress! + row * bgraStride,
                                        byteCount: bgraStride)
                                }
                            }
                        }
                    }
                    let time = CMTime(value: CMTimeValue(frame), timescale: fps)
                    guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                        throw writer.error
                            ?? EngineError.native("appending frame \(frame) failed")
                    }
                    progress(Double(frame + 1) / Double(frames) * 0.9)
                    try Task.checkCancellation()
                }
                chunkStart += window
            }
            videoInput.markAsFinished()
            try await audioDone
        } catch {
            writer.cancelWriting()
            throw error
        }

        await writer.finishWriting()
        if writer.status != .completed {
            throw writer.error ?? EngineError.native("video encode did not complete")
        }
        progress(1.0)
        return output
    }
}
