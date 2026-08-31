import XCTest

final class EnrollmentUITests: XCTestCase {
    func testNewVoiceCanStartWithoutTypingANameFirst() throws {
        let app = XCUIApplication()
        app.launchEnvironment["FTTS_DEBUG_ENROLLMENT"] = "1"
        app.launch()

        let name = app.textFields["enrollment-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 8))
        XCTAssertEqual(name.value as? String, "My Voice")

        let start = app.buttons["enrollment-start-recording"]
        XCTAssertTrue(start.waitForExistence(timeout: 3))
        XCTAssertTrue(start.isEnabled, "Start recording must never be a silent no-op")

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "enrollment-ready-to-record"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
