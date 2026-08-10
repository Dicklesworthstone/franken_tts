// The monster's palette: the site's dark-emerald laboratory, in native clothes.

import SwiftUI

enum Lab {
    static let background = Color(red: 0.008, green: 0.039, blue: 0.024) // #020a06 family
    static let panel = Color.black.opacity(0.4)
    static let stroke = Color.white.opacity(0.06)
    static let emerald = Color(red: 0.204, green: 0.827, blue: 0.6) // #34d399
    static let emeraldDeep = Color(red: 0.0, green: 0.259, blue: 0.145) // #004225
    static let textPrimary = Color(red: 0.886, green: 0.91, blue: 0.941)
    static let textSecondary = Color(red: 0.58, green: 0.639, blue: 0.722)
    static let danger = Color(red: 0.973, green: 0.443, blue: 0.443)
}

/// Section label in the site's uppercase-tracked style.
struct LabLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 11, weight: .black, design: .monospaced))
            .kerning(2.5)
            .foregroundStyle(Lab.emerald)
    }
}

/// One stitched bolt stud, the theme's signature.
struct Bolt: View {
    var body: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(white: 0.35), Color(white: 0.05)],
                        center: .topLeading, startRadius: 1, endRadius: 8))
            Rectangle().fill(Color(white: 0.15)).frame(width: 1.2, height: 7).rotationEffect(.degrees(45))
            Rectangle().fill(Color(white: 0.15)).frame(width: 1.2, height: 7).rotationEffect(.degrees(-45))
        }
        .frame(width: 13, height: 13)
        .overlay(Circle().stroke(Color.white.opacity(0.15), lineWidth: 0.8))
        .shadow(color: Lab.emerald.opacity(0.35), radius: 4)
    }
}

/// The laboratory panel: dark card, hairline border, bolts on two corners.
struct LabPanel<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Lab.panel, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(Lab.stroke, lineWidth: 1))
            .overlay(alignment: .topLeading) { Bolt().offset(x: -5, y: -5) }
            .overlay(alignment: .bottomTrailing) { Bolt().offset(x: 5, y: 5) }
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 11)
            .background(
                LinearGradient(
                    colors: [Lab.emeraldDeep, Lab.emerald.opacity(0.8)],
                    startPoint: .topLeading, endPoint: .bottomTrailing),
                in: Capsule())
            .opacity(configuration.isPressed ? 0.75 : 1)
    }
}

struct GhostButtonStyle: ButtonStyle {
    var tint: Color = Lab.textSecondary
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(tint)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.02), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.1), lineWidth: 1))
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

/// Waveform of the finished audio: min/max envelope per column, site-style emerald.
struct WaveformView: View {
    let samples: [Float]
    var body: some View {
        Canvas { context, size in
            guard !samples.isEmpty else { return }
            let columns = max(1, Int(size.width))
            let perColumn = max(1, samples.count / columns)
            let mid = size.height / 2
            for column in 0..<columns {
                let start = column * perColumn
                guard start < samples.count else { break }
                var low: Float = 0
                var high: Float = 0
                for index in start..<min(start + perColumn, samples.count) {
                    low = min(low, samples[index])
                    high = max(high, samples[index])
                }
                let top = mid - CGFloat(high) * (mid - 2)
                let bottom = mid - CGFloat(low) * (mid - 2)
                let rect = CGRect(
                    x: CGFloat(column), y: top, width: 1, height: max(1, bottom - top))
                context.fill(Path(rect), with: .color(Lab.emerald.opacity(0.75)))
            }
        }
        .frame(height: 64)
        .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10).stroke(Lab.emerald.opacity(0.15), lineWidth: 1))
        .accessibilityLabel("Waveform of the generated audio")
    }
}
