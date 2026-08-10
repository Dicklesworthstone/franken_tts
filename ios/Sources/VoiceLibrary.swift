// Persistent enrolled voices: one JSON file per voice under Application Support.
//
// A voice is 4 KB of vector plus its name, so plain files beat a database: trivially
// inspectable, trivially deletable, and backed up with the app (unlike the 2 GB model,
// voices are user-created and worth backing up).

import Foundation

struct EnrolledVoice: Identifiable, Codable, Equatable {
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

    func reload() {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        voices =
            contents
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                (try? Data(contentsOf: url)).flatMap {
                    try? JSONDecoder().decode(EnrolledVoice.self, from: $0)
                }
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    func voice(id: UUID) -> EnrolledVoice? {
        voices.first { $0.id == id }
    }

    @discardableResult
    func add(name: String, vector: [Float]) throws -> EnrolledVoice {
        let voice = EnrolledVoice(id: UUID(), name: name, vector: vector)
        try persist(voice)
        reload()
        return voice
    }

    func rename(id: UUID, to name: String) throws {
        guard var voice = voice(id: id) else { return }
        voice.name = name
        try persist(voice)
        reload()
    }

    func replaceVector(id: UUID, with vector: [Float]) throws {
        guard var voice = voice(id: id) else { return }
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
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(voice).write(to: url(for: voice.id), options: .atomic)
    }
}
