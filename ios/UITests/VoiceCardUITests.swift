import XCTest

final class VoiceCardUITests: XCTestCase {
    func testVoiceCardRendersAndSharesFromTheOwningLibraryCover() throws {
        let app = XCUIApplication()
        app.launchEnvironment["FTTS_DEBUG_CARD"] = "1"
        app.launch()

        let preview = app.descendants(matching: .any)["voice-card-preview"]
        XCTAssertTrue(
            preview.waitForExistence(timeout: 8),
            "The nested Voice Library presentation never produced a voice-card preview."
        )
        XCTAssertTrue(app.buttons["Share the card"].exists)
        XCTAssertTrue(app.buttons["Save to Photos"].exists)
        keepScreenshot(app, named: "custom-voice-card-rendered")

        app.buttons["Share the card"].tap()
        XCTAssertTrue(
            app.otherElements["ActivityListView"].waitForExistence(timeout: 5),
            "The rendered voice-card file did not reach the system share sheet."
        )
    }

    private func keepScreenshot(_ app: XCUIApplication, named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
