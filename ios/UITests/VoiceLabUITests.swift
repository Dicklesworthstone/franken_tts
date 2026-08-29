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
        XCTAssertTrue(
            app.staticTexts["0:00 / 0:05"].waitForExistence(timeout: 2),
            "A completed preview must show its known duration before first playback"
        )
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

final class VoiceBrowserUITests: XCTestCase {
    func testLongPersonalVoiceNameStaysOnOneLineOnCompactPhone() throws {
        let app = XCUIApplication()
        app.launchEnvironment["FTTS_DEBUG_LONG_VOICE_NAME"] = "1"
        app.launch()

        XCTAssertTrue(app.navigationBars["Voice Library"].waitForExistence(timeout: 8))
        let name = app.staticTexts[
            "voice-library-name-A11CE000-0000-4000-8000-000000000001"
        ]
        XCTAssertTrue(name.waitForExistence(timeout: 3))
        XCTAssertLessThan(
            name.frame.height,
            32,
            "Long voice names must scale within a stable single-line tile instead of wrapping"
        )

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "voice-browser-long-name"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
