import UIKit
import XCTest
@testable import FrankenTTS

final class TextImportTests: XCTestCase {
    func testLargeTextFileReadsOnlyTheUtteranceBudget() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-import-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try String(repeating: "A", count: JokeLibrary.maximumUtteranceLength + 200)
            .write(to: url, atomically: true, encoding: .utf8)

        let imported = try TextImportLoader.readTextFile(from: url)

        XCTAssertEqual(imported.text.count, JokeLibrary.maximumUtteranceLength)
        XCTAssertTrue(imported.wasTruncated)
        XCTAssertTrue(imported.text.allSatisfy { $0 == "A" })
    }

    func testExtendedGraphemesAreNotCutByAFourByteAssumption() throws {
        let family = "👨‍👩‍👧‍👦"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-unicode-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try String(repeating: family, count: JokeLibrary.maximumUtteranceLength + 20)
            .write(to: url, atomically: true, encoding: .utf8)

        let imported = try TextImportLoader.readTextFile(from: url)

        XCTAssertEqual(imported.text.count, JokeLibrary.maximumUtteranceLength)
        XCTAssertTrue(imported.wasTruncated)
        XCTAssertTrue(imported.text.allSatisfy { String($0) == family })
    }

    func testNulHeavyBinaryFileIsRejectedAsText() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-binary-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0, count: 8_192).write(to: url)

        XCTAssertThrowsError(try TextImportLoader.readTextFile(from: url))
    }

    func testNulAfterTheReadabilitySampleIsStillRejected() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-late-nul-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try (String(repeating: "A", count: 5_000) + "\0hidden tail")
            .write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try TextImportLoader.readTextFile(from: url))
    }

    func testPDFKitExtractsTextInPageOrder() throws {
        let renderer = UIGraphicsPDFRenderer(
            bounds: CGRect(x: 0, y: 0, width: 612, height: 792)
        )
        let data = renderer.pdfData { context in
            context.beginPage()
            ("First laboratory page" as NSString).draw(at: CGPoint(x: 48, y: 64))
            context.beginPage()
            ("Second laboratory page" as NSString).draw(at: CGPoint(x: 48, y: 64))
        }

        let imported = try TextImportLoader.extractPDF(from: data)

        XCTAssertTrue(imported.text.contains("First laboratory page"))
        XCTAssertTrue(imported.text.contains("Second laboratory page"))
        XCTAssertLessThan(
            imported.text.range(of: "First")!.lowerBound,
            imported.text.range(of: "Second")!.lowerBound
        )
        XCTAssertFalse(imported.wasTruncated)
    }

    @MainActor
    func testClearRevokesAnInFlightPDFImport() async throws {
        let renderer = UIGraphicsPDFRenderer(
            bounds: CGRect(x: 0, y: 0, width: 612, height: 792)
        )
        let data = renderer.pdfData { context in
            for page in 0..<80 {
                context.beginPage()
                (("Laboratory page \(page) " + String(repeating: "specimen ", count: 80)) as NSString)
                    .draw(in: CGRect(x: 48, y: 48, width: 516, height: 690))
            }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-cancelled-pdf-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let model = LabModel()
        model.text = "Text that should be replaced only if the import still owns the editor."
        model.importDesktopFile(url)
        await Task.yield()

        model.clearUtterance()
        XCTAssertEqual(model.text, "")
        XCTAssertFalse(model.isImportingText)

        // PDFKit does not cooperatively cancel page extraction. Give the detached worker
        // time to finish and prove its stale result cannot resurrect the cleared document.
        try await Task.sleep(for: .milliseconds(500))
        XCTAssertEqual(model.text, "")
        XCTAssertFalse(model.isImportingText)
    }

    @MainActor
    func testReadableHTMLPrefersArticleAndDropsNavigation() async throws {
        let html = """
        <!doctype html><html><head><title>Test page</title></head><body>
          <nav>HOME PRODUCTS PRICING ACCOUNT ACCOUNT ACCOUNT</nav>
          <main><article>
            <h1>The useful story</h1>
            <p>This is the first substantial paragraph of the text that a person actually wants to hear.</p>
            <p>This second paragraph makes the continuous reading group stronger and preserves its order.</p>
          </article></main>
          <footer>Privacy Terms Contact Careers</footer>
        </body></html>
        """

        let text = try await ReadableTextExtractor.extract(from: html)

        XCTAssertTrue(text.contains("The useful story"))
        XCTAssertTrue(text.contains("first substantial paragraph"))
        XCTAssertFalse(text.contains("PRICING ACCOUNT"))
        XCTAssertFalse(text.contains("Privacy Terms"))
    }

    @MainActor
    func testReadableHTMLExtractionHonorsCancellation() async {
        let html = "<html><body><article>" + String(repeating: "laboratory ", count: 20_000)
            + "</article></body></html>"
        let task = Task { @MainActor in
            try await ReadableTextExtractor.extract(from: html)
        }
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("a cancelled extraction must not publish text")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
    }

    @MainActor
    func testReadableHTMLExtractionTimesOutInsteadOfHanging() async {
        let html = "<html><body><article>" + String(repeating: "laboratory ", count: 20_000)
            + "</article></body></html>"

        do {
            _ = try await ReadableTextExtractor.extract(
                from: html,
                timeout: .milliseconds(1)
            )
            XCTFail("the deliberately tiny deadline should expire")
        } catch TextImportLoader.ImportError.extractionTimedOut {
            // Expected.
        } catch {
            XCTFail("expected extractionTimedOut, got \(error)")
        }
    }
}
