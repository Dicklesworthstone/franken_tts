// The monster's palette: the site's dark-emerald laboratory, in native clothes.

import SwiftUI

enum Lab {
    static let background = Color(red: 0.008, green: 0.039, blue: 0.024) // #020a06 family
    static let backgroundDeep = Color(red: 0.002, green: 0.012, blue: 0.009)
    static let panel = Color.black.opacity(0.52)
    static let panelStrong = Color.black.opacity(0.72)
    static let stroke = Color.white.opacity(0.06)
    static let emerald = Color(red: 0.204, green: 0.827, blue: 0.6) // #34d399
    static let emeraldDeep = Color(red: 0.0, green: 0.259, blue: 0.145) // #004225
    static let cyan = Color(red: 0.25, green: 0.82, blue: 0.96)
    static let violet = Color(red: 0.66, green: 0.55, blue: 0.98)
    static let amber = Color(red: 0.98, green: 0.75, blue: 0.14)
    static let textPrimary = Color(red: 0.886, green: 0.91, blue: 0.941)
    static let textSecondary = Color(red: 0.58, green: 0.639, blue: 0.722)
    static let danger = Color(red: 0.973, green: 0.443, blue: 0.443)
}

/// A deep, spatial laboratory wash. The grid is architectural, not progress
/// telemetry; it stays quiet until a processing instrument lights above it.
struct LaboratoryBackground: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        ZStack {
            Lab.backgroundDeep
            RadialGradient(
                colors: [Lab.emerald.opacity(reduceTransparency ? 0.08 : 0.19), .clear],
                center: UnitPoint(x: 0.08, y: -0.02),
                startRadius: 0,
                endRadius: 520
            )
            RadialGradient(
                colors: [Lab.violet.opacity(reduceTransparency ? 0.035 : 0.08), .clear],
                center: UnitPoint(x: 1.02, y: 0.24),
                startRadius: 0,
                endRadius: 460
            )
            Canvas { context, size in
                var grid = Path()
                let spacing: CGFloat = 44
                var x: CGFloat = 0
                while x <= size.width {
                    grid.move(to: CGPoint(x: x, y: 0))
                    grid.addLine(to: CGPoint(x: x, y: size.height))
                    x += spacing
                }
                var y: CGFloat = 0
                while y <= size.height {
                    grid.move(to: CGPoint(x: 0, y: y))
                    grid.addLine(to: CGPoint(x: size.width, y: y))
                    y += spacing
                }
                context.stroke(grid, with: .color(Color.white.opacity(0.018)), lineWidth: 0.5)
            }
        }
        .ignoresSafeArea()
    }
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
            .background {
                ZStack {
                    Lab.panelStrong
                    LinearGradient(
                        colors: [Lab.emerald.opacity(0.035), Color.clear, Lab.violet.opacity(0.022)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                }
                .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            }
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [Lab.emerald.opacity(0.25), Color.white.opacity(0.055), Color.white.opacity(0.025)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
            .overlay(alignment: .topLeading) { Bolt().offset(x: -5, y: -5) }
            .overlay(alignment: .bottomTrailing) { Bolt().offset(x: 5, y: 5) }
            .shadow(color: .black.opacity(0.42), radius: 22, y: 12)
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 11)
            .frame(minHeight: 48)
            .background(
                LinearGradient(
                    colors: [Lab.emeraldDeep, Lab.emerald.opacity(0.8)],
                    startPoint: .topLeading, endPoint: .bottomTrailing),
                in: Capsule())
            .opacity(isEnabled ? (configuration.isPressed ? 0.76 : 1) : 0.34)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .shadow(color: isEnabled ? Lab.emerald.opacity(0.22) : .clear, radius: 12, y: 5)
            .animation(.easeOut(duration: 0.14), value: configuration.isPressed)
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
            .frame(minHeight: 44)
            .background(Color.white.opacity(0.02), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.1), lineWidth: 1))
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

struct StatusCapsule: View {
    let title: String
    let detail: String
    let systemImage: String
    var tint: Color = Lab.emerald

    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 24, height: 24)
                .background(tint.opacity(0.12), in: Circle())
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Lab.textPrimary)
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(Lab.textSecondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            Image(systemName: "chevron.right")
                .font(.caption2.weight(.bold))
                .foregroundStyle(Lab.textSecondary.opacity(0.65))
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(tint.opacity(0.18), lineWidth: 1))
        .contentShape(Capsule())
        .accessibilityElement(children: .combine)
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
