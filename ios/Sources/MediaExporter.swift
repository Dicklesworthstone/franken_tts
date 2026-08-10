// Share formats: M4A audio (small, the default) and the branded MP4 video.
//
// The video frames come from the same Rust renderer behind `ftts make-video`, through
// the FFI, so the phone's clip is pixel-identical to the desktop's; the encode differs
// by platform on purpose — ffmpeg on desktop, the hardware H.264/AAC encoders here.

import AVFoundation
import FttsCore
import Foundation

enum MediaExporter {
    /// Transcode the played WAV into an AAC .m4a (an order of magnitude smaller).
    static func exportM4A(fromWav wavUrl: URL) async throws -> URL {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("franken_tts.m4a")
        try? FileManager.default.removeItem(at: output)
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
            .appendingPathComponent("franken_tts_video.mp4")
        try? FileManager.default.removeItem(at: output)

        let width = Int(ftts_video_width())
        let height = Int(ftts_video_height())
        let fps = Int32(ftts_video_fps())

        let renderer: OpaquePointer? = pcm.withUnsafeBufferPointer { buffer in
            ftts_video_open(buffer.baseAddress, buffer.count, 24_000, voiceLabel)
        }
        guard let renderer else { throw EngineError.lastFromNative() }
        defer { ftts_video_close(renderer) }
        let frames = ftts_video_frame_count(renderer)

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
        let audioInput = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 24_000,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 96_000,
            ])
        audioInput.expectsMediaDataInRealTime = false
        writer.add(videoInput)
        writer.add(audioInput)
        guard writer.startWriting() else {
            throw writer.error ?? EngineError.native("video writer failed to start")
        }
        writer.startSession(atSourceTime: .zero)

        // Video first: Rust renders RGB24, converted here into the adaptor's BGRA pool.
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for frame in 0..<frames {
            let code = rgb.withUnsafeMutableBufferPointer { buffer in
                ftts_video_render_frame(renderer, frame, buffer.baseAddress)
            }
            guard code == 0 else { throw EngineError.lastFromNative() }
            while !videoInput.isReadyForMoreMediaData {
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
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
                let destination = base.assumingMemoryBound(to: UInt8.self)
                rgb.withUnsafeBufferPointer { source in
                    for row in 0..<height {
                        var from = row * width * 3
                        var to = row * stride
                        for _ in 0..<width {
                            destination[to] = source[from + 2] // B
                            destination[to + 1] = source[from + 1] // G
                            destination[to + 2] = source[from] // R
                            destination[to + 3] = 255 // A
                            from += 3
                            to += 4
                        }
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            let time = CMTime(value: CMTimeValue(frame), timescale: fps)
            guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                throw writer.error ?? EngineError.native("appending frame \(frame) failed")
            }
            progress(Double(frame + 1) / Double(frames) * 0.9)
            try Task.checkCancellation()
        }
        videoInput.markAsFinished()

        // Then the audio track, decoded from the WAV and AAC-encoded by the writer.
        // One asset instance throughout: a reader can only consume tracks belonging to
        // the exact asset it was created over, not an equal asset at the same URL.
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
        while let sample = readerOutput.copyNextSampleBuffer() {
            while !audioInput.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(4))
            }
            guard audioInput.append(sample) else {
                throw writer.error ?? EngineError.native("appending audio failed")
            }
        }
        audioInput.markAsFinished()

        await writer.finishWriting()
        if writer.status != .completed {
            throw writer.error ?? EngineError.native("video encode did not complete")
        }
        progress(1.0)
        return output
    }
}
