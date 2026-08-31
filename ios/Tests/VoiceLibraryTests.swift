import XCTest
@testable import FrankenTTS

@MainActor
final class VoiceLibraryTests: XCTestCase {
    func testSuggestedEnrollmentNameSkipsExistingNamesCaseInsensitively() {
        let library = VoiceLibrary()
        library.installDebugVoices([
            fixture(name: "My Voice"),
            fixture(name: "my voice 2"),
            fixture(name: "Workshop Narrator"),
        ])

        XCTAssertEqual(library.suggestedEnrollmentName(), "My Voice 3")
        XCTAssertEqual(
            library.suggestedEnrollmentName(base: "Workshop Narrator"),
            "Workshop Narrator 2"
        )
        XCTAssertEqual(library.suggestedEnrollmentName(base: "   "), "My Voice 3")
    }

    private func fixture(name: String) -> EnrolledVoice {
        EnrolledVoice(
            id: UUID(),
            name: name,
            vector: Array(repeating: 0, count: Engine.speakerWidth)
        )
    }
}
