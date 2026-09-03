import Foundation
import XCTest
@testable import FrankenTTS

final class SynthesisHistoryTests: XCTestCase {
    func testHistoryRejectsBytesThatAreNotWaveAudio() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = SynthesisHistoryStore(directory: directory)

        XCTAssertThrowsError(
            try store.record(
                wavData: Data(repeating: 0x2A, count: 256),
                voiceLabel: "Matt",
                durationSeconds: 1
            )
        )
        XCTAssertTrue(store.entries.isEmpty)
    }

    func testHistoryPersistsOnlyPrivacyRedactedMetadataAndAudio() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = SynthesisHistoryStore(directory: directory)
        let wav = wavData(sample: 0.25, count: 106)

        let entry = try store.record(
            wavData: wav,
            voiceLabel: "  Matt  ",
            durationSeconds: 3.25
        )

        XCTAssertEqual(entry.voiceLabel, "Matt")
        XCTAssertEqual(store.storageBytes, wav.count)
        XCTAssertEqual(try Data(contentsOf: try XCTUnwrap(store.fileURL(for: entry))), wav)

        let manifest = try String(
            contentsOf: directory.appendingPathComponent("history.json"),
            encoding: .utf8
        )
        XCTAssertTrue(manifest.contains("frankentts.synthesis-history.v1"))
        for forbidden in ["utterance", "text", "seed", "speaker", "voiceprint", "vector"] {
            XCTAssertFalse(manifest.localizedCaseInsensitiveContains(forbidden), forbidden)
        }

        let restored = SynthesisHistoryStore(directory: directory)
        XCTAssertEqual(restored.entries, [entry])
        XCTAssertEqual(restored.storageBytes, wav.count)
    }

    func testHistoryPrunesByCountAndAge() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let store = SynthesisHistoryStore(directory: directory, now: now)

        let wav = wavData(sample: 0.1, count: 10)
        for index in 0..<(SynthesisHistoryStore.maximumEntries + 3) {
            try store.record(
                wavData: wav,
                voiceLabel: "Voice \(index)",
                durationSeconds: 1,
                createdAt: now
            )
        }
        XCTAssertEqual(store.entries.count, SynthesisHistoryStore.maximumEntries)
        XCTAssertEqual(store.storageBytes, SynthesisHistoryStore.maximumEntries * wav.count)
        let retainedURLs = try store.entries.map { try XCTUnwrap(store.fileURL(for: $0)) }

        let expired = SynthesisHistoryStore(
            directory: directory,
            now: now.addingTimeInterval(SynthesisHistoryStore.maximumAge + 1)
        )
        XCTAssertTrue(expired.entries.isEmpty)
        XCTAssertEqual(expired.storageBytes, 0)
        XCTAssertTrue(retainedURLs.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
    }

    func testMalformedManifestIsIgnoredWithoutDeletingUnclaimedAudio() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = SynthesisHistoryStore(directory: directory)
        let entry = try store.record(
            wavData: wavData(sample: 0.2, count: 18),
            voiceLabel: "Aria",
            durationSeconds: 2
        )
        let audioURL = try XCTUnwrap(store.fileURL(for: entry))
        try Data("{not-json".utf8).write(
            to: directory.appendingPathComponent("history.json"),
            options: .atomic
        )

        let recovered = SynthesisHistoryStore(directory: directory)

        XCTAssertTrue(recovered.entries.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: audioURL.path))
    }

    func testDeleteAndClearRemoveOnlyOwnedHistoryFiles() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let unrelated = directory.appendingPathComponent("keep-me.txt")
        try Data("owner data".utf8).write(to: unrelated)
        let store = SynthesisHistoryStore(directory: directory)
        let first = try store.record(
            wavData: wavData(sample: 0.1, count: 2),
            voiceLabel: "Matt",
            durationSeconds: 1
        )
        let second = try store.record(
            wavData: wavData(sample: 0.2, count: 26),
            voiceLabel: "Judy",
            durationSeconds: 2
        )
        let firstURL = try XCTUnwrap(store.fileURL(for: first))
        let secondURL = try XCTUnwrap(store.fileURL(for: second))

        store.delete(first)
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: secondURL.path))
        XCTAssertEqual(store.entries, [second])

        store.deleteAll()
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unrelated.path))
        XCTAssertTrue(store.entries.isEmpty)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FrankenTTSHistoryTests-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func wavData(sample: Float, count: Int) -> Data {
        WavWriter.data(from: Array(repeating: sample, count: count))
    }
}
