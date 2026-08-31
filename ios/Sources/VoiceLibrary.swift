// Persistent enrolled voices: one JSON file per voice under Application Support.
//
// A voice is 4 KB of vector plus its name, so plain files beat a database: trivially
// inspectable, trivially deletable, and backed up with the app (unlike the 2 GB model,
// voices are user-created and worth backing up).

import Foundation

struct EnrolledVoice: Identifiable, Codable, Equatable, Sendable {
    let id: UUID
    var name: String
    var vector: [Float]
}

@MainActor
@Observable
final class VoiceLibrary {
    private(set) var voices: [EnrolledVoice] = []

    private let directory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("franken_tts/voices", isDirectory: true)
    }()

    init() {
        reload()
    }

    #if DEBUG
        /// Non-persistent layout fixture for UI tests. Long names are especially
        /// important here because a wrapped label changes the height of an entire
        /// adaptive grid row on compact phones.
        func installDebugVoices(_ fixtures: [EnrolledVoice]) {
            voices = fixtures.sorted {
                $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
            }
        }
    #endif

    func reload() {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        voices =
            contents
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let voice = try? JSONDecoder().decode(EnrolledVoice.self, from: data),
                      url.deletingPathExtension().lastPathComponent
                        .caseInsensitiveCompare(voice.id.uuidString) == .orderedSame
                else { return nil }
                return voice
            }
            // A truncated or hand-edited file must not reach the engine or the card
            // renderer; both require exactly the speaker width, all finite.
            .filter { voice in
                voice.vector.count == Engine.speakerWidth
                    && voice.vector.allSatisfy(\.isFinite)
                    && !voice.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && !voice.name.utf8.contains(0)
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    func voice(id: UUID) -> EnrolledVoice? {
        voices.first { $0.id == id }
    }

    /// Return a friendly, unused label so a new enrollment can begin immediately.
    /// Voice names are presentation labels rather than storage keys, but avoiding
    /// duplicates makes the library and the synthesis picker unambiguous.
    func suggestedEnrollmentName(base: String = "My Voice") -> String {
        let trimmedBase = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let root = trimmedBase.isEmpty ? "My Voice" : trimmedBase

        func isAvailable(_ candidate: String) -> Bool {
            !voices.contains { voice in
                voice.name.trimmingCharacters(in: .whitespacesAndNewlines)
                    .localizedCaseInsensitiveCompare(candidate) == .orderedSame
            }
        }

        if isAvailable(root) { return root }

        var suffix = 2
        while !isAvailable("\(root) \(suffix)") {
            suffix += 1
        }
        return "\(root) \(suffix)"
    }

    @discardableResult
    func add(name: String, vector: [Float]) throws -> EnrolledVoice {
        let voice = EnrolledVoice(id: UUID(), name: name, vector: vector)
        try persist(voice)
        reload()
        return voice
    }

    func rename(id: UUID, to name: String) throws {
        guard var voice = voice(id: id) else {
            throw EngineError.native("that saved voice no longer exists")
        }
        voice.name = name
        try persist(voice)
        reload()
    }

    func replaceVector(id: UUID, with vector: [Float]) throws {
        guard var voice = voice(id: id) else {
            throw EngineError.native("that saved voice no longer exists")
        }
        voice.vector = vector
        try persist(voice)
        reload()
    }

    /// Replace an enrollment and its label with one atomic file write. Keeping these
    /// together prevents a failed rename from leaving the new fingerprint stored under
    /// the old identity while the UI reports that enrollment failed.
    func update(id: UUID, name: String, vector: [Float]) throws {
        guard var voice = voice(id: id) else {
            throw EngineError.native("that saved voice no longer exists")
        }
        voice.name = name
        voice.vector = vector
        try persist(voice)
        reload()
    }

    func delete(id: UUID) {
        try? FileManager.default.removeItem(at: url(for: id))
        reload()
    }

    private func url(for id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json")
    }

    private func persist(_ voice: EnrolledVoice) throws {
        guard voice.vector.count == Engine.speakerWidth,
              voice.vector.allSatisfy(\.isFinite)
        else {
            throw EngineError.native("voice fingerprint is damaged or has the wrong size")
        }
        guard !voice.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !voice.name.utf8.contains(0)
        else {
            throw EngineError.native("voice name must contain visible text without NUL characters")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(voice).write(to: url(for: voice.id), options: .atomic)
    }
}
