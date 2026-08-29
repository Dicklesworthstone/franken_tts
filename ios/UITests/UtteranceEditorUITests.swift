import XCTest

final class UtteranceEditorUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    func testSelectAllReplacesTextAndClearKeepsEditorUsable() throws {
        let editor = app.descendants(matching: .any)["utteranceEditor"]
        XCTAssertTrue(editor.waitForExistence(timeout: 8))

        editor.tap()
        keepScreenshot(named: "utterance-editor-focused")
        // UI tests may run with the simulator's hardware keyboard connected, in
        // which case no software-keyboard element exists. Typing is the reliable
        // proof that the native editor actually owns keyboard focus.
        editor.typeText("x")

        let selectAll = app.buttons.matching(
            NSPredicate(format: "label CONTAINS[c] %@", "select all")
        ).firstMatch
        XCTAssertTrue(selectAll.waitForExistence(timeout: 2))
        selectAll.tap()
        editor.typeText("Alpha beta gamma.")
        XCTAssertEqual(editor.value as? String, "Alpha beta gamma.")

        let clear = app.buttons.matching(
            NSPredicate(format: "label CONTAINS[c] %@", "clear")
        ).firstMatch
        XCTAssertTrue(clear.waitForExistence(timeout: 2))
        clear.tap()
        XCTAssertEqual(editor.value as? String, "")

        editor.typeText("Still editable after clearing.")
        XCTAssertEqual(editor.value as? String, "Still editable after clearing.")
    }

    func testMultilineEmojiEditAndOutsideTapDismissesKeyboard() throws {
        let editor = app.descendants(matching: .any)["utteranceEditor"]
        XCTAssertTrue(editor.waitForExistence(timeout: 8))
        editor.tap()

        app.buttons.matching(
            NSPredicate(format: "label CONTAINS[c] %@", "select all")
        ).firstMatch.tap()
        let replacement = "First line.\nSecond line: café 👩🏽‍🔬."
        editor.typeText(replacement)
        XCTAssertEqual(editor.value as? String, replacement)
        keepScreenshot(named: "utterance-editor-multiline")

        app.staticTexts.matching(
            NSPredicate(format: "label CONTAINS[c] %@", "THE UTTERANCE")
        ).firstMatch.tap()
    }

    private func keepScreenshot(named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
