import UIKit
import UniformTypeIdentifiers

final class ShareViewController: UIViewController {
    private let statusLabel = UILabel()
    private let openButton = UIButton(type: .system)

    override func viewDidLoad() {
        super.viewDidLoad()
        configureView()
        loadSharedText()
    }

    private func configureView() {
        view.backgroundColor = UIColor(red: 0.005, green: 0.035, blue: 0.022, alpha: 1)

        let mark = UIImageView(image: UIImage(systemName: "waveform.badge.plus"))
        mark.tintColor = UIColor(red: 0.20, green: 0.83, blue: 0.60, alpha: 1)
        mark.preferredSymbolConfiguration = UIImage.SymbolConfiguration(pointSize: 30, weight: .bold)

        let title = UILabel()
        title.text = "VOICE FORGE"
        title.textColor = .white
        title.font = .monospacedSystemFont(ofSize: 18, weight: .black)

        statusLabel.text = "Reading the selected text…"
        statusLabel.textColor = UIColor.white.withAlphaComponent(0.66)
        statusLabel.font = .preferredFont(forTextStyle: .subheadline)
        statusLabel.numberOfLines = 0
        statusLabel.textAlignment = .center

        var configuration = UIButton.Configuration.filled()
        configuration.title = "Open FrankenTTS"
        configuration.image = UIImage(systemName: "bolt.fill")
        configuration.imagePadding = 8
        configuration.baseBackgroundColor = UIColor(red: 0.04, green: 0.55, blue: 0.34, alpha: 1)
        configuration.cornerStyle = .capsule
        openButton.configuration = configuration
        openButton.isEnabled = false
        openButton.addTarget(self, action: #selector(openForge), for: .touchUpInside)

        let cancel = UIButton(type: .system)
        cancel.setTitle("Cancel", for: .normal)
        cancel.tintColor = UIColor.white.withAlphaComponent(0.62)
        cancel.addTarget(self, action: #selector(cancelShare), for: .touchUpInside)

        let stack = UIStackView(arrangedSubviews: [mark, title, statusLabel, openButton, cancel])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
            stack.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 360),
            openButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
        ])
    }

    private func loadSharedText() {
        let providers = (extensionContext?.inputItems as? [NSExtensionItem])?
            .compactMap(\.attachments)
            .flatMap { $0 } ?? []
        guard let provider = providers.first(where: {
            $0.hasItemConformingToTypeIdentifier(UTType.plainText.identifier)
        }) else {
            showFailure("Select text before opening FrankenTTS from Share.")
            return
        }

        provider.loadItem(forTypeIdentifier: UTType.plainText.identifier, options: nil) { [weak self] item, _ in
            let text: String?
            switch item {
            case let value as String:
                text = value
            case let value as Data:
                text = String(data: value, encoding: .utf8)
            case let value as URL:
                text = try? Self.readTextPrefix(from: value)
            default:
                text = nil
            }
            Task { @MainActor in
                guard let self else { return }
                let cleaned = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !cleaned.isEmpty else {
                    self.showFailure("That share did not contain readable text.")
                    return
                }
                FrankenTTSSharedStore.stage(text: cleaned)
                self.statusLabel.text = "Text secured locally. The Voice Forge is charged and ready."
                self.openButton.isEnabled = true
            }
        }
    }

    /// A provider-supplied URL can name a very large document. The app accepts
    /// at most 600 characters, so never read an unbounded file into the share
    /// extension's much smaller memory budget just to discard almost all of it.
    private static func readTextPrefix(from url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: 32 * 1024) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    private func showFailure(_ message: String) {
        statusLabel.text = message
        statusLabel.textColor = UIColor(red: 0.97, green: 0.44, blue: 0.44, alpha: 1)
    }

    @objc private func openForge() {
        guard let url = URL(string: "frankentts://forge") else { return }
        extensionContext?.open(url) { [weak self] _ in
            self?.extensionContext?.completeRequest(returningItems: nil)
        }
    }

    @objc private func cancelShare() {
        extensionContext?.cancelRequest(withError: CocoaError(.userCancelled))
    }
}
