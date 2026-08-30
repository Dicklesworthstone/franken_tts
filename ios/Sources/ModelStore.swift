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
import CoreFoundation
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
    /// Hashing these files at launch costs only a few milliseconds and closes the
    /// exact-size corruption hole that a byte-count-only readiness check leaves open.
    static let launchDigestLimit: Int64 = 5_000_000
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
    private var launchValidationTask: Task<Void, Never>?
    private var launchValidationID: UUID?
    private var activeTaskID: UUID?
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = true
        configuration.timeoutIntervalForRequest = 90
        configuration.timeoutIntervalForResource = 60 * 60 * 24
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: configuration)
    }()

    let modelDirectory: URL

    init(modelDirectory: URL? = nil) {
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        self.modelDirectory = modelDirectory
            ?? applicationSupport.appendingPathComponent("franken_tts/model", isDirectory: true)
        refreshCachedBytes()
        if largeArtifactsHaveExpectedSizes {
            // A full 2 GB digest on every launch would be hostile to battery and startup.
            // Instead, hash every small tokenizer/config artifact and parse the large
            // containers' bounded headers before advertising readiness. The Rust loader
            // performs the deeper section checks when the engine warms.
            phase = .verifying(asset: "installed model")
            let validationID = UUID()
            launchValidationID = validationID
            launchValidationTask = Task { [weak self] in
                await self?.validateInstalledModelAtLaunch(validationID: validationID)
            }
        }
    }

    private func missingFiles() -> [ModelFile] {
        ModelManifest.files.filter { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
            return size != file.bytes
        }
    }

    private var largeArtifactsHaveExpectedSizes: Bool {
        ModelManifest.files
            .filter { $0.bytes >= ModelManifest.launchDigestLimit }
            .allSatisfy { installedSize(of: $0) == $0.bytes }
    }

    private func installedSize(of file: ModelFile) -> Int64? {
        let path = modelDirectory.appendingPathComponent(file.relativePath).path
        return (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
    }

    private func validateInstalledModelAtLaunch(validationID: UUID) async {
        defer {
            if launchValidationID == validationID {
                launchValidationID = nil
                launchValidationTask = nil
            }
        }
        do {
            let largeFiles = ModelManifest.files.filter {
                $0.bytes >= ModelManifest.launchDigestLimit
            }
            for file in largeFiles {
                try requireLaunchValidation(validationID)
                let url = modelDirectory.appendingPathComponent(file.relativePath)
                let hasExpectedSize = installedSize(of: file) == file.bytes
                let hasValidHeader = hasExpectedSize
                    ? try await Self.hasValidContainerHeader(file: file, at: url)
                    : false
                let valid = hasExpectedSize && hasValidHeader
                try requireLaunchValidation(validationID)
                guard valid else {
                    phase = .failed(
                        "A saved model component is damaged. Tap Repair model to verify and replace only the affected file."
                    )
                    return
                }
            }

            let smallFiles = ModelManifest.files.filter {
                $0.bytes < ModelManifest.launchDigestLimit
            }
            var filesToRepair: [ModelFile] = []
            for file in smallFiles {
                try requireLaunchValidation(validationID)
                let url = modelDirectory.appendingPathComponent(file.relativePath)
                let hasExpectedSize = installedSize(of: file) == file.bytes
                let hasValidDigest = hasExpectedSize
                    ? try await digest(of: url) == file.sha256
                    : false
                let valid = hasExpectedSize && hasValidDigest
                try requireLaunchValidation(validationID)
                guard valid else {
                    filesToRepair.append(file)
                    continue
                }
            }

            for file in filesToRepair {
                try requireLaunchValidation(validationID)
                phase = .verifying(asset: file.displayName)
                try await repairSmallFile(file)
                try requireLaunchValidation(validationID)
                refreshCachedBytes()
            }

            try requireLaunchValidation(validationID)
            guard isComplete else {
                phase = .idle
                return
            }
            phase = .ready
        } catch is CancellationError {
            // An explicit download, pause, or clear owns the next phase.
        } catch {
            guard launchValidationID == validationID else { return }
            phase = .failed(
                "The model's support files could not be repaired: \(error.localizedDescription)"
            )
        }
    }

    private func repairSmallFile(_ file: ModelFile) async throws {
        guard file.bytes < ModelManifest.launchDigestLimit,
              let source = URL(string: ModelManifest.releaseBase + file.asset)
        else {
            throw DownloadError.invalidResponse("The requested repair was not a small model file.")
        }
        let (temporary, response) = try await session.download(for: URLRequest(url: source))
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw DownloadError.invalidResponse("The model server refused the repair download.")
        }
        guard try Self.fileSize(temporary) == file.bytes,
              try await digest(of: temporary) == file.sha256
        else {
            throw DownloadError.invalidResponse(
                "\(file.displayName) did not pass its security check."
            )
        }
        try Task.checkCancellation()
        let destination = modelDirectory.appendingPathComponent(file.relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Self.adoptDownloadedFile(temporary, at: destination)
    }

    var isComplete: Bool {
        ModelManifest.files.filter(\.required).allSatisfy { file in
            let path = modelDirectory.appendingPathComponent(file.relativePath).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? nil
            return size == file.bytes
        }
    }

    /// Waits for the bounded launch integrity check that owns the transition from
    /// `.verifying` to `.ready`. Hidden benchmark lanes must not bypass this gate merely
    /// because every artifact happens to have the expected byte count.
    func waitForLaunchValidation() async {
        let pending = launchValidationTask
        await pending?.value
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
        launchValidationTask?.cancel()
        launchValidationID = nil
        launchValidationTask = nil
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
        launchValidationTask?.cancel()
        launchValidationID = nil
        launchValidationTask = nil
        task?.cancel()
        task = nil
        activeTaskID = nil
        refreshCachedBytes()
        downloadRateBytesPerSecond = 0
        phase = .idle
    }

    func clear() {
        launchValidationTask?.cancel()
        launchValidationID = nil
        launchValidationTask = nil
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

            let started = ProcessInfo.processInfo.systemUptime
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
        started: TimeInterval,
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
            // Do not publish a stale near-100% count while the replacement begins.
            refreshCachedBytes()
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

    private func requireLaunchValidation(_ validationID: UUID) throws {
        try Task.checkCancellation()
        guard launchValidationID == validationID else { throw CancellationError() }
    }

    private func updateProgress(asset: String, started: TimeInterval, startingBytes: Int64) {
        let elapsed = max(0, ProcessInfo.processInfo.systemUptime - started)
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

    /// Cheap startup validation for the two large artifact formats. This catches
    /// zero-filled, truncated, or mislabeled files without reading their multi-GB payloads.
    nonisolated static func hasValidContainerHeader(
        file: ModelFile,
        at url: URL
    ) async throws -> Bool {
        try await Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            guard file.bytes >= 0,
                  try handle.seekToEnd() == UInt64(file.bytes)
            else { return false }
            try handle.seek(toOffset: 0)

            if file.asset.hasSuffix(".fttsq") {
                guard let prefix = try handle.read(upToCount: 20),
                      prefix.count == 20,
                      prefix.prefix(8) == Data("FTTSQ\0\0\0".utf8)
                else { return false }
                let version = prefix[8..<12].withUnsafeBytes { raw -> UInt32 in
                    raw.loadUnaligned(as: UInt32.self).littleEndian
                }
                let directoryLength = prefix[12..<20].withUnsafeBytes { raw -> UInt64 in
                    raw.loadUnaligned(as: UInt64.self).littleEndian
                }
                guard version == 1,
                      directoryLength > 0,
                      directoryLength <= 64 * 1_024 * 1_024,
                      directoryLength <= UInt64(file.bytes - 20),
                      let directory = try handle.read(upToCount: Int(directoryLength)),
                      directory.count == Int(directoryLength),
                      let json = try? JSONSerialization.jsonObject(with: directory),
                      let object = json as? [String: Any]
                else { return false }
                return validFttsqDirectory(
                    object,
                    version: UInt64(version),
                    directoryEnd: 20 + directoryLength,
                    fileLength: UInt64(file.bytes)
                )
            }
            if file.asset.hasSuffix(".safetensors") {
                guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                    return false
                }
                let headerLength = prefix.withUnsafeBytes { raw -> UInt64 in
                    raw.loadUnaligned(as: UInt64.self).littleEndian
                }
                guard headerLength > 1,
                      headerLength <= 64 * 1024 * 1024,
                      headerLength <= UInt64(max(0, file.bytes - 8))
                else { return false }
                guard let header = try handle.read(upToCount: Int(headerLength)),
                      header.count == Int(headerLength),
                      let json = try? JSONSerialization.jsonObject(with: header),
                      let object = json as? [String: Any]
                else { return false }
                return validSafetensorsDirectory(
                    object,
                    payloadLength: UInt64(file.bytes) - 8 - headerLength
                )
            }
            return false
        }.value
    }

    private nonisolated static func validFttsqDirectory(
        _ object: [String: Any],
        version: UInt64,
        directoryEnd: UInt64,
        fileLength: UInt64
    ) -> Bool {
        guard unsignedInteger(object["format_version"]) == version,
              nonemptyString(object["model_family"]) != nil,
              nonemptyString(object["source_sha256"]) != nil,
              let licenseNotice = nonemptyString(object["license_notice"]),
              !licenseNotice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let sectionObjects = object["sections"] as? [[String: Any]],
              !sectionObjects.isEmpty,
              sectionObjects.count <= 64,
              let tensorObjects = object["tensors"] as? [[String: Any]],
              !tensorObjects.isEmpty,
              tensorObjects.count <= 16_384
        else { return false }

        let accessClasses: Set<String> = [
            "HOT_RECURRENT_MICRODECODER", "HOT_RECURRENT_TALKER",
            "HOT_CODEC_DECODER", "COLD_TEXT_EMBEDDING",
            "ENROLLMENT_SPEAKER_ENCODER", "ENROLLMENT_CODEC_ENCODER", "METADATA",
        ]
        var sectionLengths: [String: UInt64] = [:]
        var sectionRanges: [(UInt64, UInt64)] = []
        for section in sectionObjects {
            guard let name = nonemptyString(section["name"]),
                  sectionLengths[name] == nil,
                  let accessClass = nonemptyString(section["access_class"]),
                  accessClasses.contains(accessClass),
                  let offset = unsignedInteger(section["offset"]),
                  let length = unsignedInteger(section["length"]),
                  offset >= directoryEnd,
                  let end = checkedSum(offset, length),
                  end <= fileLength,
                  isSHA256(section["sha256"])
            else { return false }
            sectionLengths[name] = length
            sectionRanges.append((offset, end))
        }
        sectionRanges.sort { $0.0 < $1.0 }
        for index in 1..<sectionRanges.count
        where sectionRanges[index - 1].1 > sectionRanges[index].0 {
            return false
        }

        let dtypes: Set<String> = ["bf16", "f32", "q8", "q4"]
        var tensorNames = Set<String>()
        var rangesBySection: [String: [(UInt64, UInt64)]] = [:]
        for tensor in tensorObjects {
            guard let name = nonemptyString(tensor["name"]),
                  tensorNames.insert(name).inserted,
                  let section = nonemptyString(tensor["section"]),
                  let sectionLength = sectionLengths[section],
                  let dtype = nonemptyString(tensor["dtype"]),
                  dtypes.contains(dtype),
                  let rawShape = tensor["shape"] as? [Any],
                  rawShape.count <= 8,
                  let shape = integerShape(rawShape),
                  let offset = unsignedInteger(tensor["offset"]),
                  let length = unsignedInteger(tensor["length"]),
                  let end = checkedSum(offset, length),
                  end <= sectionLength,
                  storageByteCount(shape: shape, dtype: dtype) == length
            else { return false }
            rangesBySection[section, default: []].append((offset, end))
        }
        for var ranges in rangesBySection.values {
            ranges.sort { $0.0 < $1.0 }
            for index in 1..<ranges.count where ranges[index - 1].1 > ranges[index].0 {
                return false
            }
        }
        return true
    }

    private nonisolated static func validSafetensorsDirectory(
        _ object: [String: Any],
        payloadLength: UInt64
    ) -> Bool {
        let entries = object.filter { $0.key != "__metadata__" }
        guard !entries.isEmpty, entries.count <= 16_384 else { return false }
        var ranges: [(UInt64, UInt64)] = []
        for (name, value) in entries {
            guard !name.isEmpty,
                  let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  dtype == "BF16" || dtype == "F32",
                  let rawShape = entry["shape"] as? [Any],
                  rawShape.count <= 8,
                  let shape = integerShape(rawShape),
                  let offsets = entry["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let begin = unsignedInteger(offsets[0]),
                  let end = unsignedInteger(offsets[1]),
                  begin <= end,
                  end <= payloadLength,
                  storageByteCount(shape: shape, dtype: dtype) == end - begin
            else { return false }
            ranges.append((begin, end))
        }
        ranges.sort {
            $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0
        }
        var cursor: UInt64 = 0
        for range in ranges {
            // Safetensors requires the payload to be indexed exactly once. Refusing
            // gaps, trailers, and overlaps keeps this bounded launch check aligned
            // with the Rust parser that will consume the same file during warm-up.
            guard range.0 == cursor else { return false }
            cursor = range.1
        }
        return cursor == payloadLength
    }

    private nonisolated static func nonemptyString(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty else { return nil }
        return value
    }

    private nonisolated static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        let signed = number.int64Value
        guard signed >= 0, number.doubleValue == Double(signed) else { return nil }
        return UInt64(signed)
    }

    private nonisolated static func integerShape(_ values: [Any]) -> [UInt64]? {
        var shape: [UInt64] = []
        shape.reserveCapacity(values.count)
        for value in values {
            guard let dimension = unsignedInteger(value), dimension <= 1 << 32 else {
                return nil
            }
            shape.append(dimension)
        }
        return shape
    }

    private nonisolated static func checkedSum(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? nil : result.partialValue
    }

    private nonisolated static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? nil : result.partialValue
    }

    private nonisolated static func storageByteCount(
        shape: [UInt64],
        dtype: String
    ) -> UInt64? {
        var elements: UInt64 = 1
        for dimension in shape {
            let product = elements.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { return nil }
            elements = product.partialValue
        }
        switch dtype {
        case "BF16", "bf16": return checkedProduct(elements, 2)
        case "F32", "f32": return checkedProduct(elements, 4)
        case "q8": return elements
        case "q4": return checkedSum(elements, 1).map { $0 / 2 }
        default: return nil
        }
    }

    private nonisolated static func isSHA256(_ value: Any?) -> Bool {
        guard let text = value as? String, text.utf8.count == 64 else { return false }
        return text.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
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
