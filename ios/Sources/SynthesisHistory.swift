import Foundation
import Observation

struct SynthesisHistoryEntry: Codable, Equatable, Identifiable {
    let id: UUID
    let createdAt: Date
    let voiceLabel: String
    let durationSeconds: Double
    let byteCount: Int
    let fileName: String

    private enum CodingKeys: String, CodingKey {
        case id
        case createdAtMilliseconds
        case voiceLabel
        case durationSeconds
        case byteCount
        case fileName
    }

    init(
        id: UUID,
        createdAt: Date,
        voiceLabel: String,
        durationSeconds: Double,
        byteCount: Int,
        fileName: String
    ) {
        self.id = id
        self.createdAt = createdAt
        self.voiceLabel = voiceLabel
        self.durationSeconds = durationSeconds
        self.byteCount = byteCount
        self.fileName = fileName
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        let milliseconds = try container.decode(Int64.self, forKey: .createdAtMilliseconds)
        createdAt = Date(timeIntervalSince1970: Double(milliseconds) / 1_000)
        voiceLabel = try container.decode(String.self, forKey: .voiceLabel)
        durationSeconds = try container.decode(Double.self, forKey: .durationSeconds)
        byteCount = try container.decode(Int.self, forKey: .byteCount)
        fileName = try container.decode(String.self, forKey: .fileName)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(Self.milliseconds(createdAt), forKey: .createdAtMilliseconds)
        try container.encode(voiceLabel, forKey: .voiceLabel)
        try container.encode(durationSeconds, forKey: .durationSeconds)
        try container.encode(byteCount, forKey: .byteCount)
        try container.encode(fileName, forKey: .fileName)
    }

    static func normalized(_ date: Date) -> Date? {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else { return nil }
        return Date(timeIntervalSince1970: Double(value) / 1_000)
    }

    private static func milliseconds(_ date: Date) throws -> Int64 {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else {
            throw SynthesisHistoryError.invalidResult
        }
        return value
    }
}

@Observable
final class SynthesisHistoryStore {
    static let maximumEntries = 12
    static let maximumAge: TimeInterval = 7 * 24 * 60 * 60
    static let maximumStoredBytes = 64 * 1_024 * 1_024

    private static let manifestSchema = "frankentts.synthesis-history.v1"
    private static let manifestName = "history.json"

    private(set) var entries: [SynthesisHistoryEntry] = []
    private(set) var storageBytes = 0

    private let directory: URL
    private let fileManager: FileManager
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(
        directory requestedDirectory: URL? = nil,
        now: Date = .now,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager
        directory = requestedDirectory ?? Self.defaultDirectory(fileManager: fileManager)
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        prepareDirectory()
        reload(now: now)
    }

    @discardableResult
    func record(
        wavData: Data,
        voiceLabel: String,
        durationSeconds: Double,
        createdAt: Date = .now
    ) throws -> SynthesisHistoryEntry {
        guard let stableCreatedAt = SynthesisHistoryEntry.normalized(createdAt),
              Self.isRIFFWave(wavData),
              wavData.count <= Self.maximumStoredBytes,
              durationSeconds.isFinite,
              durationSeconds > 0 else {
            throw SynthesisHistoryError.invalidResult
        }
        let id = UUID()
        let fileName = "\(id.uuidString.lowercased()).wav"
        let url = directory.appendingPathComponent(fileName, isDirectory: false)
        try wavData.write(to: url, options: .atomic)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try? mutableURL.setResourceValues(values)

        let boundedVoice = String(
            voiceLabel.trimmingCharacters(in: .whitespacesAndNewlines).prefix(120)
        )
        let entry = SynthesisHistoryEntry(
            id: id,
            createdAt: stableCreatedAt,
            voiceLabel: boundedVoice.isEmpty ? "Voice" : boundedVoice,
            durationSeconds: durationSeconds,
            byteCount: wavData.count,
            fileName: fileName
        )
        let previousEntries = entries
        entries.insert(entry, at: 0)
        let removed = prune(now: stableCreatedAt, deleteRemoved: false)
        do {
            try persistManifest()
            for removedEntry in removed { removeAudio(for: removedEntry) }
        } catch {
            entries = previousEntries
            try? fileManager.removeItem(at: url)
            recalculateStorage()
            throw error
        }
        return entry
    }

    func fileURL(for entry: SynthesisHistoryEntry) -> URL? {
        guard entries.contains(where: { $0.id == entry.id && $0.fileName == entry.fileName }),
              Self.isOwnedFileName(entry.fileName, id: entry.id) else { return nil }
        let url = directory.appendingPathComponent(entry.fileName, isDirectory: false)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        return url
    }

    func delete(_ entry: SynthesisHistoryEntry) {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let removed = entries.remove(at: index)
        recalculateStorage()
        do {
            try persistManifest()
            removeAudio(for: removed)
        } catch {
            entries.insert(removed, at: index)
            recalculateStorage()
        }
    }

    func deleteAll() {
        let removed = entries
        entries.removeAll(keepingCapacity: false)
        recalculateStorage()
        do {
            try persistManifest()
            for entry in removed { removeAudio(for: entry) }
        } catch {
            entries = removed
            recalculateStorage()
        }
    }

    private func prepareDirectory() {
        try? fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableDirectory = directory
        try? mutableDirectory.setResourceValues(values)
    }

    private func reload(now: Date) {
        let manifestURL = directory.appendingPathComponent(Self.manifestName)
        guard let data = try? Data(contentsOf: manifestURL), data.count <= 512_000,
              let manifest = try? decoder.decode(Manifest.self, from: data),
              manifest.schema == Self.manifestSchema else {
            entries = []
            storageBytes = 0
            return
        }
        var seenIDs = Set<UUID>()
        entries = manifest.entries.filter { entry in
            guard seenIDs.insert(entry.id).inserted else { return false }
            guard Self.isValidMetadata(entry),
                  Self.isOwnedFileName(entry.fileName, id: entry.id) else { return false }
            let url = directory.appendingPathComponent(entry.fileName)
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true,
                  values.fileSize == entry.byteCount else { return false }
            return true
        }
        // Commit the repaired manifest before removing any expired, future-dated,
        // over-count, or over-budget owned audio. A failed manifest write therefore
        // never leaves a still-referenced clip missing on the next launch.
        let removed = prune(now: now, deleteRemoved: false)
        do {
            try persistManifest()
            for entry in removed { removeAudio(for: entry) }
        } catch {
            // Keep the safe in-memory view. The old manifest and every referenced
            // file remain intact, so a later launch can retry the repair.
        }
    }

    @discardableResult
    private func prune(now: Date, deleteRemoved: Bool = true) -> [SynthesisHistoryEntry] {
        entries.sort {
            if $0.createdAt == $1.createdAt { return $0.id.uuidString < $1.id.uuidString }
            return $0.createdAt > $1.createdAt
        }
        var kept: [SynthesisHistoryEntry] = []
        var removed: [SynthesisHistoryEntry] = []
        var bytes = 0
        for entry in entries {
            let fits = kept.count < Self.maximumEntries
                && now.timeIntervalSince(entry.createdAt) <= Self.maximumAge
                && entry.createdAt.timeIntervalSince(now) <= 60
                && bytes <= Self.maximumStoredBytes - entry.byteCount
            if fits {
                kept.append(entry)
                bytes += entry.byteCount
            } else {
                removed.append(entry)
            }
        }
        entries = kept
        storageBytes = bytes
        if deleteRemoved {
            for entry in removed { removeAudio(for: entry) }
        }
        return removed
    }

    private func removeAudio(for entry: SynthesisHistoryEntry) {
        guard Self.isOwnedFileName(entry.fileName, id: entry.id) else { return }
        try? fileManager.removeItem(
            at: directory.appendingPathComponent(entry.fileName, isDirectory: false)
        )
    }

    private func recalculateStorage() {
        storageBytes = entries.reduce(0) { $0 + $1.byteCount }
    }

    private func persistManifest() throws {
        let data = try encoder.encode(Manifest(schema: Self.manifestSchema, entries: entries))
        try data.write(
            to: directory.appendingPathComponent(Self.manifestName),
            options: .atomic
        )
    }

    private static func isValidMetadata(_ entry: SynthesisHistoryEntry) -> Bool {
        !entry.voiceLabel.isEmpty && entry.voiceLabel.count <= 120
            && entry.durationSeconds.isFinite && entry.durationSeconds > 0
            && entry.byteCount > 0 && entry.byteCount <= maximumStoredBytes
    }

    private static func isRIFFWave(_ data: Data) -> Bool {
        data.count >= 44
            && data.starts(with: Data("RIFF".utf8))
            && data.dropFirst(8).starts(with: Data("WAVE".utf8))
    }

    private static func isOwnedFileName(_ fileName: String, id: UUID) -> Bool {
        fileName == "\(id.uuidString.lowercased()).wav"
            && URL(fileURLWithPath: fileName).lastPathComponent == fileName
    }

    private static func defaultDirectory(fileManager: FileManager) -> URL {
        let root = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        return root
            .appendingPathComponent("FrankenTTS", isDirectory: true)
            .appendingPathComponent("Synthesis History", isDirectory: true)
    }

    private struct Manifest: Codable {
        let schema: String
        let entries: [SynthesisHistoryEntry]
    }
}

enum SynthesisHistoryError: LocalizedError {
    case invalidResult

    var errorDescription: String? {
        "The generated audio is not valid for local history."
    }
}
