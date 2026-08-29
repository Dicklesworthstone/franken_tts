import XCTest

final class VoiceLabUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchEnvironment["FTTS_OPEN_VOICE_COMPARISON"] = "1"
        app.launchEnvironment["FTTS_VOICE_LAB_FIXTURE"] = "1"
        app.launch()
    }

    func testMixedCardStatesStayStableAndLeavingRequiresConfirmation() throws {
        XCTAssertTrue(app.navigationBars["Voice Lab"].waitForExistence(timeout: 8))
        keepScreenshot(named: "voice-lab-mixed-states-top")

        let activeOrb = app.descendants(matching: .any)["voice-lab-active-orb"]
        XCTAssertTrue(activeOrb.waitForExistence(timeout: 3))
        let initialFrame = activeOrb.frame
        let stabilityWindow = expectation(description: "active orb stays in place")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            stabilityWindow.fulfill()
        }
        wait(for: [stabilityWindow], timeout: 1.2)
        XCTAssertEqual(activeOrb.frame, initialFrame)

        app.swipeUp()
        let play = app.buttons["Play"].firstMatch
        XCTAssertTrue(play.waitForExistence(timeout: 3))
        let signal = app.descendants(matching: .any)
            .matching(NSPredicate(format: "identifier BEGINSWITH 'voice-lab-signal-'"))
            .firstMatch
        XCTAssertTrue(signal.waitForExistence(timeout: 3))
        keepScreenshot(named: "voice-lab-complete-spectrogram")

        play.tap()
        XCTAssertTrue(app.buttons["Pause"].firstMatch.waitForExistence(timeout: 2))
        signal.coordinate(withNormalizedOffset: CGVector(dx: 0.72, dy: 0.58)).tap()
        XCTAssertTrue(
            app.buttons["Pause"].firstMatch.waitForExistence(timeout: 2),
            "Touch-seeking the Voice Lab spectrogram must resume that preview"
        )
        keepScreenshot(named: "voice-lab-scrubbed-playback")

        app.navigationBars["Voice Lab"].buttons["Done"].tap()
        let alert = app.alerts["Voice Lab is still generating"]
        XCTAssertTrue(alert.waitForExistence(timeout: 2))
        XCTAssertTrue(alert.buttons["Keep generating"].exists)
        XCTAssertTrue(alert.buttons["Stop and leave"].exists)
        alert.buttons["Keep generating"].tap()
        XCTAssertTrue(app.navigationBars["Voice Lab"].exists)
    }

    private func keepScreenshot(named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
