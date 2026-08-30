import XCTest
@testable import FrankenTTS

final class SpeechMasteringTests: XCTestCase {
    func testMasteringConvergesQuietAndLoudSpeechWithoutClipping() {
        let quiet = fixture(amplitude: 0.012)
        let loud = fixture(amplitude: 0.62)

        let masteredQuiet = SpeechMastering.process(quiet)
        let masteredLoud = SpeechMastering.process(loud)

        XCTAssertEqual(masteredQuiet.count, quiet.count)
        XCTAssertEqual(masteredLoud.count, loud.count)
        XCTAssertTrue(masteredQuiet.allSatisfy(\.isFinite))
        XCTAssertTrue(masteredLoud.allSatisfy(\.isFinite))
        XCTAssertLessThanOrEqual(masteredQuiet.map(abs).max() ?? 0, 0.951)
        XCTAssertLessThanOrEqual(masteredLoud.map(abs).max() ?? 0, 0.951)

        let quietRMS = activeFixtureRMS(masteredQuiet)
        let loudRMS = activeFixtureRMS(masteredLoud)
        XCTAssertEqual(quietRMS, loudRMS, accuracy: 0.012)
        XCTAssertGreaterThan(quietRMS, 0.08)
        XCTAssertLessThan(quietRMS, 0.15)
    }

    func testMasteringSanitizesInvalidSilence() {
        let mastered = SpeechMastering.process([0, .nan, .infinity, -.infinity, 0])
        XCTAssertEqual(mastered, [0, 0, 0, 0, 0])
    }

    func testWavWriterSerializesNonFiniteSamplesAsSilence() {
        let wav = WavWriter.data(from: [.nan, .infinity, -.infinity, 0])

        XCTAssertEqual(wav.count, 44 + 8)
        XCTAssertEqual(Array(wav.suffix(8)), Array(repeating: UInt8(0), count: 8))
    }

    private func fixture(amplitude: Float) -> [Float] {
        let sampleRate = 24_000
        let activeCount = sampleRate
        let padding = sampleRate / 10
        let active = (0..<activeCount).map { index -> Float in
            let time = Float(index) / Float(sampleRate)
            return amplitude * (
                0.72 * sin(2 * .pi * 170 * time)
                    + 0.20 * sin(2 * .pi * 850 * time)
                    + 0.08 * sin(2 * .pi * 4_400 * time)
            )
        }
        return Array(repeating: 0, count: padding) + active + Array(repeating: 0, count: padding)
    }

    private func activeFixtureRMS(_ samples: [Float]) -> Float {
        let padding = 2_400
        let active = samples[padding..<(samples.count - padding)]
        let power = active.reduce(0.0) { $0 + Double($1 * $1) } / Double(active.count)
        return Float(sqrt(power))
    }
}
