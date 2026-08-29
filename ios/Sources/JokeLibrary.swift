import Foundation

enum JokeLibrary {
    static let maximumUtteranceLength = 50_000

    static let entries: [String] = {
        guard let url = Bundle.main.url(forResource: "norm_jokes", withExtension: "txt"),
              let source = try? String(contentsOf: url, encoding: .utf8)
        else {
            return [fallback]
        }

        let normalized = source.replacingOccurrences(of: "\r\n", with: "\n")
        return normalized
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && $0.count <= maximumUtteranceLength }
    }()

    static func random(excluding current: String? = nil) -> String {
        let alternatives = entries.filter { $0 != current }
        return alternatives.randomElement() ?? entries.randomElement() ?? fallback
    }

    private static let fallback =
        "I signed up for my company's 401k, but I don't think I can run that far."
}

struct UtteranceChunk: Equatable, Sendable {
    let text: String
    let trailingPauseSeconds: Double
}

/// The native model has a finite per-call generation ceiling. Long imported
/// documents therefore need explicit, language-aware segmentation rather than
/// a UI limit that accepts text the engine can silently stop before finishing.
enum UtteranceChunker {
    // 1,500 is conservative even for text that verbalizes nearly every
    // character (digits, acronyms, or CJK), keeping each call below the native
    // 8,192-frame ceiling with substantial headroom.
    static let maximumChunkCharacters = 1_500

    static func split(_ source: String) -> [UtteranceChunk] {
        let source = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty else { return [] }

        var result: [UtteranceChunk] = []
        var start = source.startIndex
        while start < source.endIndex {
            let hardEnd = source.index(
                start,
                offsetBy: maximumChunkCharacters,
                limitedBy: source.endIndex
            ) ?? source.endIndex
            if hardEnd == source.endIndex {
                append(source[start..<source.endIndex], pause: 0, to: &result)
                break
            }

            let searchFloor = source.index(
                start,
                offsetBy: maximumChunkCharacters / 2,
                limitedBy: hardEnd
            ) ?? start
            var cursor = hardEnd
            var paragraphBreak: String.Index?
            var sentenceBreak: String.Index?
            var wordBreak: String.Index?

            while cursor > searchFloor {
                let index = source.index(before: cursor)
                let character = source[index]
                if character == "\n" {
                    paragraphBreak = cursor
                    break
                }
                if sentenceBreak == nil, ".!?…".contains(character) {
                    sentenceBreak = cursor
                }
                if wordBreak == nil, character.isWhitespace {
                    wordBreak = index
                }
                cursor = index
            }

            let end = paragraphBreak ?? sentenceBreak ?? wordBreak ?? hardEnd
            let pause: Double = paragraphBreak != nil ? 0.18 : (sentenceBreak != nil ? 0.10 : 0.05)
            append(source[start..<end], pause: pause, to: &result)

            start = end
            while start < source.endIndex, source[start].isWhitespace {
                start = source.index(after: start)
            }
        }
        return result
    }

    private static func append(
        _ slice: Substring,
        pause: Double,
        to result: inout [UtteranceChunk]
    ) {
        let text = slice.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        result.append(UtteranceChunk(text: text, trailingPauseSeconds: pause))
    }
}
