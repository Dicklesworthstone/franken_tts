import Foundation
import WebKit

@MainActor
final class ReadableTextExtractor: NSObject, WKNavigationDelegate {
    private var continuation: CheckedContinuation<String, Error>?
    private let webView: WKWebView

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
    }

    static func extract(from html: String) async throws -> String {
        let extractor = ReadableTextExtractor()
        return try await extractor.extract(html)
    }

    private func extract(_ html: String) async throws -> String {
        let policy = "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        let inertHTML = html.replacingOccurrences(
            of: #"(?is)<meta\b[^>]*>"#,
            with: "",
            options: .regularExpression
        )
        let sealedHTML: String
        if let head = inertHTML.range(of: "<head", options: .caseInsensitive),
           let close = inertHTML[head.lowerBound...].firstIndex(of: ">") {
            sealedHTML = String(inertHTML[...close]) + policy
                + String(inertHTML[inertHTML.index(after: close)...])
        } else {
            sealedHTML = "<html><head>\(policy)</head><body>\(inertHTML)</body></html>"
        }

        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            webView.loadHTMLString(sealedHTML, baseURL: nil)
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        webView.evaluateJavaScript(Self.readerScript) { [weak self] value, error in
            guard let self, let continuation = self.continuation else { return }
            self.continuation = nil
            if let error {
                continuation.resume(throwing: error)
            } else if let text = value as? String,
                      !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continuation.resume(returning: text)
            } else {
                continuation.resume(throwing: TextImportLoader.ImportError.notText)
            }
        }
    }

    func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation!,
        withError error: Error
    ) {
        finish(with: error)
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        finish(with: error)
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        finish(with: TextImportLoader.ImportError.notText)
    }

    private func finish(with error: Error) {
        guard let continuation else { return }
        self.continuation = nil
        continuation.resume(throwing: error)
    }

    private static let readerScript = #"""
    (() => {
      document.querySelectorAll(
        'script,style,noscript,nav,footer,body > header,aside,form,button,iframe,svg,canvas,' +
        '[aria-hidden="true"],[class*="cookie"],[class*="advert"],[class*="newsletter"],' +
        '[class*="social"],[class*="share"],[class*="related"],[class*="comment"]'
      ).forEach(node => node.remove());

      const clean = value => (value || '')
        .replace(/[ \t]+\n/g, '\n')
        .replace(/\n[ \t]+/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      const text = node => clean(node.innerText || node.textContent || '');
      const linkText = node => Array.from(node.querySelectorAll('a'))
        .reduce((sum, link) => sum + text(link).length, 0);
      const boilerplate = /(nav|menu|footer|header|sidebar|comment|share|social|promo|advert|cookie|related)/i;
      const candidates = Array.from(document.querySelectorAll(
        'article,main,[role="main"],section,div[class*="article"],div[class*="post"],div[class*="content"],body'
      ));

      let best = document.body;
      let bestScore = -Infinity;
      for (const node of candidates) {
        const body = text(node);
        if (body.length < 120) continue;
        const identity = `${node.id || ''} ${node.className || ''}`;
        const paragraphs = node.querySelectorAll('p').length;
        const headings = node.querySelectorAll('h1,h2,h3').length;
        const links = linkText(node);
        const punctuation = (body.match(/[.!?](?:\s|$)/g) || []).length;
        const density = body.length / Math.max(1, node.querySelectorAll('*').length);
        let score = body.length + paragraphs * 180 + headings * 60 + punctuation * 18 + density * 3 - links * 2.4;
        if (boilerplate.test(identity)) score *= 0.35;
        if (node.matches('article,main,[role="main"]')) score *= 1.35;
        if (node === document.body) score *= 0.55;
        if (score > bestScore) { bestScore = score; best = node; }
      }

      const title = clean(document.querySelector('h1')?.innerText || document.title || '');
      const body = text(best);
      return clean(title && !body.startsWith(title) ? `${title}\n\n${body}` : body);
    })();
    """#
}
