import XCTest
@testable import FrankenTTS

final class VoiceCardTests: XCTestCase {
    func testCardNameTruncationPreservesUTF8Boundaries() {
        let name = String(repeating: "🧟‍♀️é", count: 20)
        let bytes = VoiceCode.cardNameBytes(name)

        XCTAssertLessThanOrEqual(bytes.count, 64)
        let decoded = String(decoding: bytes, as: UTF8.self)
        XCTAssertFalse(decoded.contains("�"))
        XCTAssertTrue(name.hasPrefix(decoded))
    }

    func testCardRendererRefusesMalformedVoiceBeforePixelWork() async {
        do {
            _ = try await VoicePrintCard.pngData(name: "Broken", vector: [0.5])
            XCTFail("A one-float fingerprint must not reach the mosaic precondition")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("wrong size"), "\(error)")
        }
    }

    func testSignalAnalysisIncludesTheFinalSourceSample() {
        var samples = [Float](repeating: 0, count: 1_001)
        samples[samples.count - 1] = 0.75

        let analysis = SignalAnalysis(samples: samples)

        XCTAssertEqual(analysis.waveHighs.last, 0.75)
    }
}
