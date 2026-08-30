import Foundation
import XCTest
@testable import FrankenTTS

final class SynthesisProfileTests: XCTestCase {
    func testLegacyNativeProfileDefaultsNewDiagnosticsWithoutLosingCoreTimings() throws {
        let legacyJSON = Data(
            """
            {
              "total_ms": 1000.0,
              "generation_ms": 800.0,
              "prefill_ms": 100.0,
              "microdecoder_ms": 200.0,
              "feedback_ms": 50.0,
              "talker_ms": 150.0,
              "codec_active_ms": 350.0,
              "frames": 42,
              "team_partitions": 5
            }
            """.utf8)

        let profile = try JSONDecoder().decode(SynthesisProfile.self, from: legacyJSON)

        XCTAssertEqual(profile.totalMs, 1_000)
        XCTAssertEqual(profile.otherGenerationMs, 300)
        XCTAssertEqual(profile.generatorGlueMs, 300)
        XCTAssertEqual(profile.codecBackpressureMs, 0)
        XCTAssertEqual(profile.codecTailMs, 0)
        XCTAssertFalse(profile.codecUserInitiatedQos)
        XCTAssertEqual(profile.frames, 42)
        XCTAssertEqual(profile.teamPartitions, 5)
    }

    func testCurrentNativeProfileDecodesNewDiagnostics() throws {
        let currentJSON = Data(
            """
            {
              "total_ms": 1000.0,
              "generation_ms": 800.0,
              "prefill_ms": 100.0,
              "microdecoder_ms": 200.0,
              "feedback_ms": 50.0,
              "talker_ms": 150.0,
              "codec_active_ms": 350.0,
              "codec_backpressure_ms": 125.0,
              "codec_tail_ms": 25.0,
              "codec_user_initiated_qos": true,
              "frames": 42,
              "team_partitions": 5
            }
            """.utf8)

        let profile = try JSONDecoder().decode(SynthesisProfile.self, from: currentJSON)

        XCTAssertEqual(profile.codecBackpressureMs, 125)
        XCTAssertEqual(profile.codecTailMs, 25)
        XCTAssertTrue(profile.codecUserInitiatedQos)
        XCTAssertEqual(profile.generatorGlueMs, 175)
    }
}
