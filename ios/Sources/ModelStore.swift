// Model download, verification, and storage.
//
// The manifest constants mirror site/model-manifest.js, which mirrors the CLI's pinned
// manifest — those files are the source of truth and the three must move together if the
// model release is ever re-pinned. Files download in small HTTP ranges to URLSession
// temporary files, then append to the resumable file on disk. This is deliberately a
// download-task path rather than `data(for:)`: a CDN that ignores Range must never make
// iPadOS materialize a 1.3 GB response in the app's synthesis heap. Every response range
// and final digest is validated. Storage is Application Support, excluded from backup.

import CryptoKit
import Foundation

struct ModelFile {
    let asset: String
    /// Where the engine expects the file, relative to the model directory.
    let relativePath: String
    let bytes: Int64
    let sha256: String
    /// Every current file is required — the denoiser included, because enrollment
    /// refuses to run without it. The flag stays for any future asset that is
    /// genuinely decorative; migration for small late-added files is handled by the
    /// silent completion in `ModelStore.init`.
    var required = true
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
        // The FastEnhancer denoiser. REQUIRED: enrollment refuses to run without it,
        // because a profile built from un-denoised audio carries the recording's noise
        // into every synthesis. Existing installs that predate this file get it
        // auto-completed silently at launch (it is under a megabyte).
        ModelFile(
            asset: "fastenhancer-s-48k-denoise.safetensors",
            relativePath: "denoise/fastenhancer-s-48k.safetensors",
            bytes: 838_440,
            sha256: "28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b"),
    ]

    static let totalBytes = files.reduce(Int64(0)) { $0 + $1.bytes }
    /// Small enough to make progress feel live and retry cheaply on an interrupted iPad
    /// connection. `URLSession.download(for:)` keeps even an unexpectedly large CDN
    /// response out of the process heap.
    static let chunkBytes: Int64 = 8 * 1024 * 1024
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
    var downloadRateBytesPerSecond: Double = 0
    var currentFileIndex = 0
    var currentFileCount = ModelManifest.files.count

    private var task: Task<Void, Never>?
    private var activeTaskID: UUID?
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = true
        configuration.timeoutIntervalForRequest = 90
        configuration.timeoutIntervalForResource = 60 * 60 * 24
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: configuration)
    }()

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
        if isComplete {
            phase = .ready
        } else if missingFiles().allSatisfy({ $0.bytes < 5_000_000 }), cachedBytes > 0 {
            // An install from before a small file (the denoiser) joined the manifest:
            // everything big is here, so completing the sub-megabyte remainder needs
            // no fresh consent — the user consented to this model. The phone stays
            // usable meanwhile; enrollment checks the engine directly before running.
            phase = .ready
            fetchSmallMissingFiles()
        }
    }

    private func missingFiles() -> [ModelFile] {
        ModelManifest.files.filter { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
            return size != file.bytes
        }
    }

    private func fetchSmallMissingFiles() {
        Task { [weak self] in
            guard let self else { return }
            for file in self.missingFiles() {
                do {
                    let data = try await URLSession.shared.data(
                        from: URL(string: ModelManifest.releaseBase + file.asset)!).0
                    let digest = SHA256.hash(data: data)
                        .map { String(format: "%02x", $0) }.joined()
                    guard digest == file.sha256 else { continue }
                    let destination = self.modelDirectory
                        .appendingPathComponent(file.relativePath)
                    try FileManager.default.createDirectory(
                        at: destination.deletingLastPathComponent(),
                        withIntermediateDirectories: true)
                    try data.write(to: destination, options: .atomic)
                    self.refreshCachedBytes()
                } catch {
                    // Transient network failure; the next launch retries.
                }
            }
        }
    }

    var isComplete: Bool {
        ModelManifest.files.filter(\.required).allSatisfy { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
            return size == file.bytes
        }
    }

    func refreshCachedBytes() {
        cachedBytes = ModelManifest.files.reduce(Int64(0)) { total, file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
            // Oversized/corrupt remnants will be replaced by `ensure`; do not let them
            // produce a progress value above 100% in the meantime.
            return total + (size <= file.bytes ? size : 0)
        }
    }

    func startDownload() {
        guard task == nil else { return }
        refreshCachedBytes()
        downloadRateBytesPerSecond = 0
        currentFileIndex = firstIncompleteFileIndex
        phase = .downloading(
            asset: "Preparing the voice engine",
            done: cachedBytes,
            total: ModelManifest.totalBytes,
            eta: "Checking storage…"
        )
        let taskID = UUID()
        activeTaskID = taskID
        task = Task { [weak self] in
            await self?.run(taskID: taskID)
            guard self?.activeTaskID == taskID else { return }
            self?.task = nil
            self?.activeTaskID = nil
        }
    }

    func pauseDownload() {
        task?.cancel()
        task = nil
        activeTaskID = nil
        refreshCachedBytes()
        downloadRateBytesPerSecond = 0
        phase = .idle
    }

    func clear() {
        task?.cancel()
        task = nil
        activeTaskID = nil
        try? FileManager.default.removeItem(at: modelDirectory)
        refreshCachedBytes()
        phase = .idle
    }

    private func run(taskID: UUID) async {
        do {
            try requireActive(taskID)
            try FileManager.default.createDirectory(
                at: modelDirectory, withIntermediateDirectories: true)
            var directory = modelDirectory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? directory.setResourceValues(values)
            try requireEnoughFreeStorage()

            let started = Date()
            let startingBytes = cachedBytes
            for (index, file) in ModelManifest.files.enumerated() {
                try requireActive(taskID)
                currentFileIndex = index + 1
                try await ensure(
                    file: file,
                    started: started,
                    startingBytes: startingBytes,
                    taskID: taskID
                )
                refreshCachedBytes()
            }
            try requireActive(taskID)
            downloadRateBytesPerSecond = 0
            phase = .ready
        } catch is CancellationError {
            guard activeTaskID == taskID else { return }
            refreshCachedBytes()
            downloadRateBytesPerSecond = 0
            phase = .idle
        } catch {
            guard activeTaskID == taskID else { return }
            refreshCachedBytes()
            downloadRateBytesPerSecond = 0
            phase = .failed(error.localizedDescription)
        }
    }

    private func ensure(
        file: ModelFile,
        started: Date,
        startingBytes: Int64,
        taskID: UUID
    ) async throws {
        try requireActive(taskID)
        let destination = modelDirectory.appendingPathComponent(file.relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let existing =
            (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int64)
            ?? nil
        if existing == file.bytes {
            // Size-complete from an earlier session: verify once before trusting.
            phase = .verifying(asset: file.displayName)
            if try await digest(of: destination) == file.sha256 { return }
            try requireActive(taskID)
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

        phase = .downloading(
            asset: file.displayName,
            done: cachedBytes,
            total: ModelManifest.totalBytes,
            eta: downloadRateBytesPerSecond > 0 ? "Updating estimate…" : "Connecting…"
        )

        while offset < file.bytes {
            try requireActive(taskID)
            let end = min(offset + ModelManifest.chunkBytes, file.bytes) - 1
            var request = URLRequest(url: URL(string: ModelManifest.releaseBase + file.asset)!)
            request.setValue("bytes=\(offset)-\(end)", forHTTPHeaderField: "Range")
            let (temporary, response) = try await downloadWithRetry(request)
            try requireActive(taskID)
            guard let http = response as? HTTPURLResponse else {
                throw DownloadError.invalidResponse("The model server returned an unreadable response.")
            }

            let temporaryBytes = try Self.fileSize(temporary)
            if http.statusCode == 206 {
                try Self.validatePartialResponse(
                    http, requestedStart: offset, requestedEnd: end,
                    expectedTotal: file.bytes, actualBytes: temporaryBytes)
                try Self.appendFile(at: temporary, to: destination)
                offset += temporaryBytes
            } else if http.statusCode == 200, offset == 0 {
                // Some CDNs ignore Range and send the whole asset. A download task keeps
                // that response on disk; verify it before atomically adopting it.
                guard temporaryBytes == file.bytes else {
                    throw DownloadError.invalidResponse(
                        "The model server ignored resume and returned the wrong file size.")
                }
                phase = .verifying(asset: file.displayName)
                guard try await digest(of: temporary) == file.sha256 else {
                    throw DownloadError.invalidResponse(
                        "\(file.displayName) did not pass its security check. Please retry.")
                }
                try requireActive(taskID)
                try Self.adoptDownloadedFile(temporary, at: destination)
                offset = file.bytes
            } else {
                throw DownloadError.httpStatus(http.statusCode)
            }

            refreshCachedBytes()
            updateProgress(
                asset: file.displayName,
                started: started,
                startingBytes: startingBytes
            )
        }

        phase = .verifying(asset: file.displayName)
        let digestHex = try await digest(of: destination)
        try requireActive(taskID)
        guard digestHex == file.sha256 else {
            try? FileManager.default.removeItem(at: destination)
            throw DownloadError.invalidResponse(
                "\(file.displayName) did not pass its security check. Its partial copy was cleared; retry to fetch a clean copy.")
        }
    }

    private func downloadWithRetry(_ request: URLRequest) async throws -> (URL, URLResponse) {
        var lastError: Error?
        for attempt in 1...3 {
            do {
                return try await session.download(for: request)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                try Task.checkCancellation()
                lastError = error
                guard attempt < 3 else { break }
                try await Task.sleep(for: .seconds(Double(attempt)))
            }
        }
        throw DownloadError.network(lastError?.localizedDescription ?? "unknown network error")
    }

    private func requireActive(_ taskID: UUID) throws {
        try Task.checkCancellation()
        guard activeTaskID == taskID else { throw CancellationError() }
    }

    private func updateProgress(asset: String, started: Date, startingBytes: Int64) {
        let elapsed = Date().timeIntervalSince(started)
        let transferred = max(0, cachedBytes - startingBytes)
        let measuredRate = elapsed > 0.8 ? Double(transferred) / elapsed : 0
        if measuredRate > 0 {
            downloadRateBytesPerSecond = downloadRateBytesPerSecond == 0
                ? measuredRate
                : downloadRateBytesPerSecond * 0.65 + measuredRate * 0.35
        }
        let remainingBytes = max(0, ModelManifest.totalBytes - cachedBytes)
        let remaining = downloadRateBytesPerSecond > 0
            ? Double(remainingBytes) / downloadRateBytesPerSecond
            : 0
        phase = .downloading(
            asset: asset,
            done: cachedBytes,
            total: ModelManifest.totalBytes,
            eta: remaining > 0 ? Self.formatEta(seconds: remaining) : "Estimating time…"
        )
    }

    private var firstIncompleteFileIndex: Int {
        guard let index = ModelManifest.files.firstIndex(where: { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
            return size != file.bytes
        }) else { return ModelManifest.files.count }
        return index + 1
    }

    private func requireEnoughFreeStorage() throws {
        refreshCachedBytes()
        let remaining = max(0, ModelManifest.totalBytes - cachedBytes)
        let safetyMargin: Int64 = 384 * 1024 * 1024
        let values = try modelDirectory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey
        ])
        guard let available = values.volumeAvailableCapacityForImportantUsage else { return }
        guard available >= remaining + safetyMargin else {
            let needed = Self.storageString(remaining + safetyMargin)
            let free = Self.storageString(available)
            throw DownloadError.insufficientStorage(needed: needed, available: free)
        }
    }

    private nonisolated static func validatePartialResponse(
        _ response: HTTPURLResponse,
        requestedStart: Int64,
        requestedEnd: Int64,
        expectedTotal: Int64,
        actualBytes: Int64
    ) throws {
        guard let value = response.value(forHTTPHeaderField: "Content-Range"),
              let range = ParsedContentRange(value),
              range.start == requestedStart,
              range.end == requestedEnd,
              range.total == expectedTotal,
              actualBytes == requestedEnd - requestedStart + 1
        else {
            throw DownloadError.invalidResponse(
                "The model server returned an invalid resume range. Please retry.")
        }
    }

    private nonisolated static func appendFile(at source: URL, to destination: URL) throws {
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        try output.seekToEnd()
        while true {
            let data = try input.read(upToCount: 1024 * 1024) ?? Data()
            if data.isEmpty { break }
            try output.write(contentsOf: data)
        }
        try output.synchronize()
    }

    private nonisolated static func adoptDownloadedFile(_ source: URL, at destination: URL) throws {
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: source)
        } else {
            try FileManager.default.moveItem(at: source, to: destination)
        }
    }

    private nonisolated static func fileSize(_ url: URL) throws -> Int64 {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values.fileSize ?? 0)
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
        return minutes > 0 ? "About \(minutes)m \(total % 60)s left" : "About \(total)s left"
    }

    private static func storageString(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

private extension ModelFile {
    var displayName: String {
        switch asset {
        case "qwen3-tts-12hz-0.6b-base.fttsq": "Voice engine"
        case "speech_tokenizer_model.safetensors": "Speech decoder"
        case "vocab.json": "Language vocabulary"
        case "merges.txt": "Tokenizer rules"
        case "tokenizer_config.json": "Tokenizer settings"
        case "fastenhancer-s-48k-denoise.safetensors": "Voice cleanup"
        default: "Model component"
        }
    }
}

private struct ParsedContentRange {
    let start: Int64
    let end: Int64
    let total: Int64

    init?(_ value: String) {
        // RFC 9110 form: "bytes 0-8388607/1312015713".
        let components = value.split(separator: " ", maxSplits: 1)
        guard components.count == 2, components[0].lowercased() == "bytes" else { return nil }
        let rangeAndTotal = components[1].split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2, let total = Int64(rangeAndTotal[1]) else { return nil }
        let bounds = rangeAndTotal[0].split(separator: "-", maxSplits: 1)
        guard bounds.count == 2,
              let start = Int64(bounds[0]),
              let end = Int64(bounds[1]),
              start >= 0,
              end >= start
        else { return nil }
        self.start = start
        self.end = end
        self.total = total
    }
}

private enum DownloadError: LocalizedError {
    case network(String)
    case httpStatus(Int)
    case invalidResponse(String)
    case insufficientStorage(needed: String, available: String)

    var errorDescription: String? {
        switch self {
        case .network(let detail):
            "The download was interrupted (\(detail)). Your progress is saved; tap Resume to continue."
        case .httpStatus(let status):
            "The model server returned HTTP \(status). Your progress is saved; wait a moment and tap Resume."
        case .invalidResponse(let message):
            message
        case .insufficientStorage(let needed, let available):
            "Not enough free space. FrankenTTS needs \(needed) available to finish safely; this device currently has \(available)."
        }
    }
}
