import XCTest

final class VoiceLabUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments += ["-frankentts.appearance", "light"]
        app.launchEnvironment["FTTS_OPEN_VOICE_COMPARISON"] = "1"
        app.launchEnvironment["FTTS_VOICE_LAB_FIXTURE"] = "1"
        app.launch()
    }

    func testComparisonWorkspaceIsDiscoverableWithoutPlayingAudio() throws {
        XCTAssertTrue(app.navigationBars["Voice Lab"].waitForExistence(timeout: 8))
        XCTAssertTrue(app.descendants(matching: .any)["voice-lab-active-orb"].waitForExistence(timeout: 3))
        keepScreenshot(named: "voice-lab-safe-overview")

        app.swipeUp()
        XCTAssertTrue(app.buttons["Play"].firstMatch.waitForExistence(timeout: 3))
        XCTAssertTrue(
            app.descendants(matching: .any)
                .matching(NSPredicate(format: "identifier BEGINSWITH 'voice-lab-signal-'"))
                .firstMatch
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(app.staticTexts["0:00 / 0:05"].waitForExistence(timeout: 2))
        keepScreenshot(named: "voice-lab-safe-results")

        app.navigationBars["Voice Lab"].buttons["Done"].tap()
        let alert = app.alerts["Voice Lab is still generating"]
        XCTAssertTrue(alert.waitForExistence(timeout: 2))
        XCTAssertTrue(alert.buttons["Keep generating"].exists)
        XCTAssertTrue(alert.buttons["Stop and leave"].exists)
        keepScreenshot(named: "voice-lab-safe-leave-confirmation")
        alert.buttons["Keep generating"].tap()
        XCTAssertTrue(app.navigationBars["Voice Lab"].exists)
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
        let names = app.staticTexts.matching(
            NSPredicate(format: "label == %@", "Alexandria-Cassandra Nightingale")
        )
        XCTAssertTrue(names.firstMatch.waitForExistence(timeout: 3))
        let appFrame = app.frame
        for identifier in [
            "voice-filter-all",
            "voice-filter-feminine",
            "voice-filter-masculine",
            "voice-filter-my-voices",
        ] {
            let filter = app.buttons[identifier]
            XCTAssertTrue(filter.exists, "Missing discoverable filter: \(identifier)")
            XCTAssertGreaterThanOrEqual(
                filter.frame.minX,
                appFrame.minX,
                "\(identifier) escaped the leading edge of the compact phone"
            )
            XCTAssertLessThanOrEqual(
                filter.frame.maxX,
                appFrame.maxX,
                "\(identifier) escaped the trailing edge of the compact phone"
            )
        }
        let visibleNames = names.allElementsBoundByIndex.filter {
            $0.exists && !$0.frame.isEmpty
        }
        guard let name = visibleNames.max(by: { $0.frame.minY < $1.frame.minY }) else {
            XCTFail("The long personal voice tile was not visible")
            return
        }
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
