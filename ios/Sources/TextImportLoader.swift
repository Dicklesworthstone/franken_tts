import Foundation
import PDFKit

struct ImportedText: Sendable {
    let text: String
    let wasTruncated: Bool
}

enum TextImportLoader {
    // A Swift Character can contain several Unicode scalars, so four bytes per
    // character is not a sound upper bound. Keep URL/file reads bounded while
    // leaving ample room to recover 50,000 extended grapheme clusters.
    private static let maximumTextBytes = 4 * 1_024 * 1_024
    private static let maximumHTMLBytes = 8 * 1_024 * 1_024
    private static let maximumPDFBytes = 32 * 1_024 * 1_024
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        return URLSession(configuration: configuration)
    }()

    static func download(from url: URL) async throws -> ImportedText {
        guard url.scheme?.lowercased() == "https" else {
            throw ImportError.invalidURL
        }

        let (bytes, response) = try await session.bytes(from: url)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              http.url?.scheme?.lowercased() == "https"
        else {
            throw ImportError.badResponse
        }

        let mime = (http.mimeType ?? "").lowercased()
        let responseURL = http.url ?? url
        let responseExtension = responseURL.pathExtension.lowercased()
        let isPDF = mime == "application/pdf" || responseExtension == "pdf"
        let isHTML = mime == "text/html" || mime == "application/xhtml+xml"
        let textLikeMIME = mime.isEmpty
            || mime.hasPrefix("text/")
            || mime == "application/json"
            || mime == "application/xml"
            || mime == "application/octet-stream"
        guard isPDF || isHTML || textLikeMIME else {
            throw ImportError.notText
        }
        let byteLimit = isPDF ? maximumPDFBytes : (isHTML ? maximumHTMLBytes : maximumTextBytes)
        if http.expectedContentLength > Int64(byteLimit), isPDF {
            throw ImportError.pdfTooLarge
        }

        var data = Data()
        data.reserveCapacity(min(byteLimit, 512 * 1_024))
        var exceededByteLimit = false
        for try await byte in bytes {
            try Task.checkCancellation()
            if data.count == byteLimit {
                exceededByteLimit = true
                break
            }
            data.append(byte)
        }

        if isPDF {
            guard !exceededByteLimit else { throw ImportError.pdfTooLarge }
            return try extractPDF(from: data)
        }

        let decoded = try decodeText(data)
        if isHTML || looksLikeHTML(decoded) {
            let readable = try await ReadableTextExtractor.extract(from: decoded)
            return capped(readable, alreadyTruncated: exceededByteLimit)
        }
        return capped(decoded, alreadyTruncated: exceededByteLimit)
    }

    static func extractPDF(from data: Data) throws -> ImportedText {
        guard let document = PDFDocument(data: data), document.pageCount > 0 else {
            throw ImportError.unreadablePDF
        }
        return try extractPDF(document)
    }

    static func extractPDF(from url: URL) throws -> ImportedText {
        guard let document = PDFDocument(url: url), document.pageCount > 0 else {
            throw ImportError.unreadablePDF
        }
        return try extractPDF(document)
    }

    static func readTextFile(from url: URL) throws -> ImportedText {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: maximumTextBytes) ?? Data()
        let hasMore = !(try handle.read(upToCount: 1) ?? Data()).isEmpty
        return capped(try decodeText(data), alreadyTruncated: hasMore)
    }

    private static func extractPDF(_ document: PDFDocument) throws -> ImportedText {

        var result = ""
        var wasTruncated = false
        for index in 0..<document.pageCount {
            guard let pageText = document.page(at: index)?.string, !pageText.isEmpty else { continue }
            if !result.isEmpty { result += "\n\n" }
            let remaining = JokeLibrary.maximumUtteranceLength + 1 - result.count
            guard remaining > 0 else {
                wasTruncated = true
                break
            }
            result += String(pageText.prefix(remaining))
            if result.count > JokeLibrary.maximumUtteranceLength {
                wasTruncated = true
                break
            }
        }
        guard !result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImportError.pdfHasNoText
        }
        return capped(result, alreadyTruncated: wasTruncated)
    }

    private static func capped(_ source: String, alreadyTruncated: Bool) -> ImportedText {
        ImportedText(
            text: String(source.prefix(JokeLibrary.maximumUtteranceLength)),
            wasTruncated: alreadyTruncated || source.count > JokeLibrary.maximumUtteranceLength
        )
    }

    private static func decodeText(_ data: Data) throws -> String {
        var candidate = data
        var decoded = String(data: candidate, encoding: .utf8)
        for _ in 0..<4 where decoded == nil && !candidate.isEmpty {
            candidate.removeLast()
            decoded = String(data: candidate, encoding: .utf8)
        }
        if decoded == nil {
            decoded = String(data: data, encoding: .isoLatin1)
        }
        guard let decoded, isReadableText(decoded) else { throw ImportError.notText }
        return decoded
    }

    private static func isReadableText(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let sample = text.unicodeScalars.prefix(4_096)
        if sample.contains(where: { $0.value == 0 }) { return false }
        let disallowed = sample.reduce(into: 0) { count, scalar in
            let isC0Control = scalar.value < 0x20
                && scalar != "\n" && scalar != "\r" && scalar != "\t"
            let isC1Control = (0x7F...0x9F).contains(scalar.value)
            if isC0Control || isC1Control {
                count += 1
            }
        }
        return disallowed * 100 <= max(1, sample.count)
    }

    private static func looksLikeHTML(_ text: String) -> Bool {
        let prefix = text.prefix(2_048).lowercased()
        return prefix.contains("<!doctype html") || prefix.contains("<html") || prefix.contains("<article")
    }

    enum ImportError: LocalizedError {
        case invalidURL
        case badResponse
        case notText
        case unreadablePDF
        case pdfHasNoText
        case pdfTooLarge

        var errorDescription: String? {
            switch self {
            case .invalidURL:
                "Enter a complete HTTPS URL for a web page, text file, or PDF."
            case .badResponse:
                "That URL did not return a readable document."
            case .notText:
                "That document is not readable text."
            case .unreadablePDF:
                "That PDF could not be opened."
            case .pdfHasNoText:
                "That PDF has no embedded text to extract."
            case .pdfTooLarge:
                "That PDF is larger than the 32 MB URL-import safety limit. Download it to Files and import it there instead."
            }
        }
    }
}
