// Model download, verification, and storage.
//
// The manifest constants mirror site/model-manifest.js, which mirrors the CLI's pinned
// manifest — those files are the source of truth and the three must move together if the
// model release is ever re-pinned. Files download in 32 MiB HTTP ranges (resumable from
// the bytes already on disk), hash incrementally as they arrive, and are refused on any
// digest mismatch. Storage is Application Support, excluded from iCloud backup.

import CryptoKit
import Foundation

struct ModelFile {
    let asset: String
    /// Where the engine expects the file, relative to the model directory.
    let relativePath: String
    let bytes: Int64
    let sha256: String
}

enum ModelManifest {
    static let releaseBase =
        "https://github.com/Dicklesworthstone/franken_tts/releases/download/model-qwen3-tts-v1/"

    static let files: [ModelFile] = [
        ModelFile(
            asset: "qwen3-tts-12hz-0.6b-base.fttsq",
            relativePath: "qwen3-tts-12hz-0.6b-base.fttsq",
            bytes: 1_312_015_713,
            sha256: "597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824"),
        ModelFile(
            asset: "speech_tokenizer_model.safetensors",
            relativePath: "speech_tokenizer/model.safetensors",
            bytes: 682_293_092,
            sha256: "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"),
        ModelFile(
            asset: "vocab.json", relativePath: "vocab.json", bytes: 2_776_833,
            sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        ModelFile(
            asset: "merges.txt", relativePath: "merges.txt", bytes: 1_671_839,
            sha256: "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        ModelFile(
            asset: "tokenizer_config.json", relativePath: "tokenizer_config.json", bytes: 7344,
            sha256: "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"),
        // The FastEnhancer denoiser. Its presence is what switches enrollment cleanup to
        // the neural route, matching `ftts pull` + `ftts enroll` on the desktop.
        ModelFile(
            asset: "fastenhancer-s-48k-denoise.safetensors",
            relativePath: "denoise/fastenhancer-s-48k.safetensors",
            bytes: 838_440,
            sha256: "28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b"),
    ]

    static let totalBytes = files.reduce(Int64(0)) { $0 + $1.bytes }
    static let chunkBytes: Int64 = 32 * 1024 * 1024
}

enum DownloadPhase: Equatable {
    case idle
    case downloading(asset: String, done: Int64, total: Int64, eta: String)
    case verifying(asset: String)
    case ready
    case failed(String)
}

@MainActor
@Observable
final class ModelStore {
    var phase: DownloadPhase = .idle
    var cachedBytes: Int64 = 0

    private var task: Task<Void, Never>?

    let modelDirectory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("franken_tts/model", isDirectory: true)
    }()

    init() {
        refreshCachedBytes()
        // Size-only trust at launch, the same tradeoff the website makes with its memo
        // file: every byte was digest-verified when it was downloaded, and the .fttsq
        // artifact re-verifies its own digests at engine load. A full re-hash of 2 GB on
        // every app start would cost tens of seconds for corruption this storage does
        // not produce in practice.
        if isComplete { phase = .ready }
    }

    var isComplete: Bool {
        ModelManifest.files.allSatisfy { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
            return size == file.bytes
        }
    }

    func refreshCachedBytes() {
        cachedBytes = ModelManifest.files.reduce(Int64(0)) { total, file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
            return total + (size ?? 0)
        }
    }

    func startDownload() {
        guard task == nil else { return }
        task = Task { [weak self] in
            await self?.run()
            self?.task = nil
        }
    }

    func clear() {
        task?.cancel()
        task = nil
        try? FileManager.default.removeItem(at: modelDirectory)
        refreshCachedBytes()
        phase = .idle
    }

    private func run() async {
        do {
            try FileManager.default.createDirectory(
                at: modelDirectory, withIntermediateDirectories: true)
            var directory = modelDirectory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? directory.setResourceValues(values)

            var doneBytes: Int64 = 0
            for file in ModelManifest.files {
                try await ensure(file: file, alreadyDone: doneBytes)
                doneBytes += file.bytes
                refreshCachedBytes()
            }
            phase = .ready
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func ensure(file: ModelFile, alreadyDone: Int64) async throws {
        let destination = modelDirectory.appendingPathComponent(file.relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let existing =
            (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int64)
            ?? nil
        if existing == file.bytes {
            // Size-complete from an earlier session: verify once before trusting.
            phase = .verifying(asset: file.asset)
            if try await digest(of: destination) == file.sha256 { return }
            try FileManager.default.removeItem(at: destination)
        }

        var offset: Int64 = 0
        if let existing, existing < file.bytes {
            offset = existing // resume
        } else if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
            offset = 0
        }

        let sink = try FileHandle(forWritingTo: destination)
        defer { try? sink.close() }
        try sink.seekToEnd()

        // A resume cannot reuse a streamed hash for the prefix; hash the whole file after.
        // A from-zero download hashes as it goes and needs no second read.
        var live: SHA256? = offset == 0 ? SHA256() : nil
        let started = Date()
        let startedOffset = offset

        while offset < file.bytes {
            try Task.checkCancellation()
            let end = min(offset + ModelManifest.chunkBytes, file.bytes) - 1
            var request = URLRequest(url: URL(string: ModelManifest.releaseBase + file.asset)!)
            request.setValue("bytes=\(offset)-\(end)", forHTTPHeaderField: "Range")
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse,
                http.statusCode == 206 || (http.statusCode == 200 && offset == 0)
            else {
                throw EngineError.native("range fetch for \(file.asset) failed")
            }
            try sink.write(contentsOf: data)
            live?.update(data: data)
            offset += Int64(data.count)

            let elapsed = Date().timeIntervalSince(started)
            let rate = elapsed > 1 ? Double(offset - startedOffset) / elapsed : 0
            let remaining = rate > 0 ? Double(ModelManifest.totalBytes - alreadyDone - offset) / rate : 0
            let eta = rate > 0 ? Self.formatEta(seconds: remaining) : "estimating…"
            phase = .downloading(
                asset: file.asset, done: alreadyDone + offset, total: ModelManifest.totalBytes,
                eta: eta)
        }
        try sink.close()

        phase = .verifying(asset: file.asset)
        let digestHex: String
        if let live {
            digestHex = live.finalize().map { String(format: "%02x", $0) }.joined()
        } else {
            digestHex = try await digest(of: destination)
        }
        guard digestHex == file.sha256 else {
            try? FileManager.default.removeItem(at: destination)
            throw EngineError.native("\(file.asset): digest mismatch; cleared for retry")
        }
    }

    /// Streaming SHA-256 of a file, 8 MiB at a time, off the main actor.
    private nonisolated func digest(of url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            while true {
                let data = try handle.read(upToCount: 8 * 1024 * 1024) ?? Data()
                if data.isEmpty { break }
                hasher.update(data: data)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    private static func formatEta(seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let minutes = total / 60
        return minutes > 0 ? "~\(minutes)m \(total % 60)s left" : "~\(total)s left"
    }
}
