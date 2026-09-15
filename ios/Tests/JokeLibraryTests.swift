import XCTest
import UIKit
@testable import FrankenTTS

final class JokeLibraryTests: XCTestCase {
    func testKeyboardAppearanceMatchesSelectedTheme() {
        XCTAssertEqual(LabAppearance.dark.keyboardAppearance, .dark)
        XCTAssertEqual(LabAppearance.light.keyboardAppearance, .light)
    }

    func testInterfaceTextScaleStepsClampAndMapToDynamicType() {
        XCTAssertEqual(Lab.steppedTextScale(0.1), Lab.minimumTextScale)
        XCTAssertEqual(Lab.steppedTextScale(1.04), 1.0, accuracy: 0.001)
        XCTAssertEqual(Lab.steppedTextScale(1.46), Lab.maximumTextScale)
        XCTAssertEqual(Lab.steppedTextScale(9), Lab.maximumTextScale)
        XCTAssertEqual(Lab.dynamicTypeSize(from: .large, for: 1.0), .large)
        XCTAssertEqual(
            Lab.dynamicTypeSize(from: .accessibility3, for: Lab.maximumTextScale),
            .accessibility5
        )
        XCTAssertEqual(
            Lab.dynamicTypeSize(from: .accessibility3, for: Lab.minimumTextScale),
            .accessibility1
        )
    }

    func testBundledJokeFileLoadsEveryBlankLineSeparatedEntry() {
        XCTAssertEqual(JokeLibrary.entries.count, 55)
        XCTAssertTrue(JokeLibrary.entries.allSatisfy { !$0.isEmpty })
        XCTAssertTrue(
            JokeLibrary.entries.allSatisfy {
                $0.count <= JokeLibrary.maximumUtteranceLength
            }
        )
        XCTAssertTrue(JokeLibrary.entries.contains { $0.contains("Albert Fish") })
    }

    func testRandomJokeAvoidsImmediateRepeat() {
        let current = JokeLibrary.entries[0]
        XCTAssertNotEqual(JokeLibrary.random(excluding: current), current)
    }

    func testEveryBuiltInVoiceHasTheOriginalModelFreePreview() throws {
        let presets = Engine.presets()

        XCTAssertEqual(presets.count, 18)
        XCTAssertEqual(
            PresetPreviewLibrary.sentence,
            "Now is the time for all good men to come to the aid of the agents."
        )
        for preset in presets {
            let url = try XCTUnwrap(
                PresetPreviewLibrary.url(for: preset.name),
                "Missing model-free preview for \(preset.name)"
            )
            let bytes = try Data(contentsOf: url).count
            XCTAssertGreaterThan(bytes, 0, "Empty preview for \(preset.name)")
            XCTAssertLessThan(bytes, 60 * 1_024, "Preview for \(preset.name) is no longer tiny")
        }
    }

    func testLongUtterancesSplitAtReadableBoundariesWithinNativeBudget() {
        let paragraph = String(repeating: "A substantial spoken sentence. ", count: 180)
        let source = [paragraph, paragraph, paragraph].joined(separator: "\n\n")

        let chunks = UtteranceChunker.split(source)

        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { !$0.text.isEmpty })
        XCTAssertTrue(chunks.allSatisfy { $0.text.count <= UtteranceChunker.maximumChunkCharacters })
        XCTAssertEqual(chunks.last?.trailingPauseSeconds, 0)
        XCTAssertTrue(chunks.dropLast().allSatisfy { $0.trailingPauseSeconds > 0 })
        XCTAssertEqual(
            chunks.map(\.text).joined(separator: " ").split(whereSeparator: \.isWhitespace),
            source.split(whereSeparator: \.isWhitespace)
        )
    }

    func testShortUtteranceRemainsOneChunkWithoutTailSilence() {
        XCTAssertEqual(
            UtteranceChunker.split("A short laboratory sentence."),
            [UtteranceChunk(text: "A short laboratory sentence.", trailingPauseSeconds: 0)]
        )
    }

    func testUnbrokenTextUsesHardBoundariesWithoutDroppingCharacters() {
        let source = String(repeating: "x", count: 50_000)
        let chunks = UtteranceChunker.split(source)

        XCTAssertTrue(chunks.allSatisfy { $0.text.count <= UtteranceChunker.maximumChunkCharacters })
        XCTAssertEqual(chunks.map(\.text).joined(), source)
    }
}
