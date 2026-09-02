import XCTest

final class WarmRetentionUITests: XCTestCase {
    func testWarmCoreSurvivesBriefPhysicalDeviceBackgroundInterval() throws {
        guard ProcessInfo.processInfo.environment["FTTS_RUN_WARM_RETENTION_UI_TEST"] == "1"
        else {
            throw XCTSkip("opt-in physical-device lifecycle check")
        }

        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(
            app.staticTexts["Voice core warm"].waitForExistence(timeout: 30),
            "the downloaded engine never reached its warm state"
        )

        XCUIDevice.shared.press(.home)
        sleep(3)
        app.activate()

        XCTAssertTrue(
            app.staticTexts["Voice core warm"].waitForExistence(timeout: 5),
            "a brief background interval discarded the warm engine"
        )
    }
}
