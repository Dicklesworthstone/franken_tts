// The monster's palette: the site's dark-emerald laboratory, in native clothes.

import AVFoundation
import SwiftUI
import UIKit

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

    static func typeSize(_ base: CGFloat) -> CGFloat {
#if targetEnvironment(macCatalyst)
        base * 1.38
#else
        UIFontMetrics(forTextStyle: .body).scaledValue(for: base)
#endif
    }
}

/// A shared FrankenSuite wordmark: full-size product initials with the
/// connective uppercase letters set as small caps. This preserves the playful
/// laboratory voice without flattening the app name into one visual block.
struct FrankenWordmark: View {
    let productInitial: String
    let productRemainder: String
    let fullName: String
    var size: CGFloat = 22
    var accent: Color = Lab.emerald

    var body: some View {
        (
            Text("F")
                .font(.system(size: Lab.typeSize(size), weight: .black, design: .monospaced))
                .foregroundColor(Lab.textPrimary.opacity(0.88))
            + Text("RANKEN")
                .font(.system(size: Lab.typeSize(size * 0.66), weight: .black, design: .monospaced))
                .foregroundColor(Lab.textPrimary.opacity(0.88))
            + Text(productInitial)
                .font(.system(size: Lab.typeSize(size), weight: .black, design: .monospaced))
                .foregroundColor(accent)
            + Text(productRemainder)
                .font(.system(size: Lab.typeSize(size * 0.66), weight: .black, design: .monospaced))
                .foregroundColor(accent)
        )
        .kerning(0.8)
        .lineLimit(1)
        .minimumScaleFactor(0.72)
        .allowsTightening(true)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(fullName)
    }
}

private struct CatalystReadableType: ViewModifier {
    func body(content: Content) -> some View {
#if targetEnvironment(macCatalyst)
        content.dynamicTypeSize(.xLarge)
#else
        content
#endif
    }
}

extension View {
    func catalystReadableType() -> some View {
        modifier(CatalystReadableType())
    }
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
            .font(.system(size: Lab.typeSize(11), weight: .black, design: .monospaced))
            .kerning(2.5)
            .foregroundStyle(Lab.emerald)
    }
}

/// The laboratory panel: dark card with a restrained hairline border.
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
            .shadow(color: .black.opacity(0.42), radius: 22, y: 12)
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: Lab.typeSize(13), weight: .black, design: .monospaced))
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
            .hoverEffect(.highlight)
    }
}

struct GhostButtonStyle: ButtonStyle {
    var tint: Color = Lab.textSecondary
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: Lab.typeSize(12), weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(tint)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .frame(minHeight: 44)
            .background(Color.white.opacity(0.02), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.1), lineWidth: 1))
            .opacity(configuration.isPressed ? 0.7 : 1)
            .hoverEffect(.highlight)
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
                .font(.system(size: Lab.typeSize(13), weight: .bold))
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

/// A real, sample-derived time/frequency instrument for finished audio. The
/// cells are short-time DFT magnitudes; the waveform and playhead share the
/// player's clock so this doubles as a precise scrubber rather than decoration.
struct PlaybackSignalView: View {
    let samples: [Float]
    let player: AVAudioPlayer?

    @State private var analysis = SignalAnalysis.empty
    @State private var draggedProgress: Double?

    var body: some View {
        GeometryReader { geometry in
            TimelineView(
                .animation(
                    minimumInterval: 1.0 / 30.0,
                    paused: player?.isPlaying != true && draggedProgress == nil
                )
            ) { _ in
                let progress = draggedProgress ?? playbackProgress
                ZStack(alignment: .topLeading) {
                    Canvas(opaque: false, colorMode: .linear, rendersAsynchronously: true) {
                        context, size in
                        drawSpectrum(context: &context, size: size, progress: progress)
                        drawWaveform(context: &context, size: size, progress: progress)
                        drawPlayhead(context: &context, size: size, progress: progress)
                    }
                    .contentShape(Rectangle())
                    .gesture(scrubGesture(width: geometry.size.width))

                    HStack {
                        Label("SIGNAL SPECTRUM", systemImage: "waveform.path.ecg")
                            .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                            .kerning(1.3)
                            .foregroundStyle(Lab.emerald.opacity(0.9))
                        Spacer()
                        Text("\(Self.clock((player?.duration ?? 0) * progress)) / \(Self.clock(player?.duration ?? 0))")
                            .font(.system(size: Lab.typeSize(9), weight: .semibold, design: .monospaced))
                            .foregroundStyle(Lab.textSecondary)
                            .monospacedDigit()
                    }
                    .padding(.horizontal, 10)
                    .padding(.top, 7)
                    .allowsHitTesting(false)
                }
            }
        }
        .background(Color.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(
                    LinearGradient(
                        colors: [Lab.emerald.opacity(0.38), Lab.cyan.opacity(0.12)],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    lineWidth: 1
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .task(id: samples.count) {
            let captured = samples
            analysis = await Task.detached(priority: .userInitiated) {
                SignalAnalysis(samples: captured)
            }.value
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Generated audio waveform and frequency spectrum")
        .accessibilityValue("Playback at \(Int(playbackProgress * 100)) percent")
        .accessibilityAdjustableAction { direction in
            guard let player, player.duration > 0 else { return }
            let delta = max(1, player.duration * 0.05)
            switch direction {
            case .increment: player.currentTime = min(player.duration, player.currentTime + delta)
            case .decrement: player.currentTime = max(0, player.currentTime - delta)
            @unknown default: break
            }
        }
    }

    private var playbackProgress: Double {
        guard let player, player.duration > 0 else { return 0 }
        return min(1, max(0, player.currentTime / player.duration))
    }

    private func scrubGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let progress = min(1, max(0, value.location.x / max(1, width)))
                draggedProgress = progress
                if let player, player.duration > 0 {
                    player.currentTime = player.duration * progress
                }
            }
            .onEnded { _ in draggedProgress = nil }
    }

    private func drawSpectrum(
        context: inout GraphicsContext,
        size: CGSize,
        progress: Double
    ) {
        guard analysis.timeBins > 0, analysis.bandCount > 0 else { return }
        let top: CGFloat = 25
        let usableHeight = max(1, size.height - top - 7)
        let cellWidth = size.width / CGFloat(analysis.timeBins)
        let cellHeight = usableHeight / CGFloat(analysis.bandCount)
        let playedX = size.width * progress

        for time in 0..<analysis.timeBins {
            for band in 0..<analysis.bandCount {
                let magnitude = analysis.magnitudes[time * analysis.bandCount + band]
                guard magnitude > 0.015 else { continue }
                let x = CGFloat(time) * cellWidth
                let y = top + usableHeight - CGFloat(band + 1) * cellHeight
                let isPlayed = x <= playedX
                let tint: Color = band < analysis.bandCount / 3
                    ? Lab.amber : (band < analysis.bandCount * 2 / 3 ? Lab.emerald : Lab.cyan)
                let opacity = (isPlayed ? 0.16 : 0.045) + Double(magnitude) * (isPlayed ? 0.72 : 0.18)
                let rect = CGRect(
                    x: x + 0.55,
                    y: y + 0.45,
                    width: max(0.6, cellWidth - 1.1),
                    height: max(0.6, cellHeight - 0.9)
                )
                context.fill(Path(roundedRect: rect, cornerRadius: 1.3), with: .color(tint.opacity(opacity)))
            }
        }
    }

    private func drawWaveform(
        context: inout GraphicsContext,
        size: CGSize,
        progress: Double
    ) {
        guard !analysis.waveLows.isEmpty else { return }
        let center = size.height * 0.67
        let amplitude = size.height * 0.22
        let width = size.width / CGFloat(analysis.waveLows.count)
        for index in analysis.waveLows.indices {
            let x = CGFloat(index) * width
            let rect = CGRect(
                x: x,
                y: center - CGFloat(analysis.waveHighs[index]) * amplitude,
                width: max(1, width * 0.72),
                height: max(
                    1,
                    CGFloat(analysis.waveHighs[index] - analysis.waveLows[index]) * amplitude
                )
            )
            let played = Double(index) / Double(analysis.waveLows.count) <= progress
            context.fill(
                Path(rect),
                with: .color((played ? Lab.emerald : Lab.textSecondary).opacity(played ? 0.92 : 0.24))
            )
        }
    }

    private func drawPlayhead(
        context: inout GraphicsContext,
        size: CGSize,
        progress: Double
    ) {
        let x = size.width * progress
        let glow = CGRect(x: x - 5, y: 23, width: 10, height: size.height - 23)
        context.fill(Path(glow), with: .linearGradient(
            Gradient(colors: [.clear, Lab.emerald.opacity(0.13), .clear]),
            startPoint: CGPoint(x: x - 5, y: 0),
            endPoint: CGPoint(x: x + 5, y: 0)
        ))
        var line = Path()
        line.move(to: CGPoint(x: x, y: 23))
        line.addLine(to: CGPoint(x: x, y: size.height))
        context.stroke(line, with: .color(Lab.emerald.opacity(0.95)), lineWidth: 1.4)
    }

    private static func clock(_ time: TimeInterval) -> String {
        guard time.isFinite, time >= 0 else { return "0:00" }
        let seconds = Int(time.rounded(.down))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

private struct SignalAnalysis: Sendable {
    static let empty = SignalAnalysis(
        timeBins: 0,
        bandCount: 0,
        magnitudes: [],
        waveLows: [],
        waveHighs: []
    )

    let timeBins: Int
    let bandCount: Int
    let magnitudes: [Float]
    let waveLows: [Float]
    let waveHighs: [Float]

    init(
        timeBins: Int,
        bandCount: Int,
        magnitudes: [Float],
        waveLows: [Float],
        waveHighs: [Float]
    ) {
        self.timeBins = timeBins
        self.bandCount = bandCount
        self.magnitudes = magnitudes
        self.waveLows = waveLows
        self.waveHighs = waveHighs
    }

    init(samples: [Float]) {
        guard !samples.isEmpty else {
            self = .empty
            return
        }

        let timeBins = 84
        let bandCount = 16
        let windowSize = 256
        let sampleRate = 24_000.0
        var magnitudes = Array(repeating: Float.zero, count: timeBins * bandCount)
        var peak: Float = 0

        for time in 0..<timeBins {
            let center = Int(Double(time) / Double(max(1, timeBins - 1)) * Double(samples.count - 1))
            let start = max(0, min(samples.count - windowSize, center - windowSize / 2))
            for band in 0..<bandCount {
                let fraction = Double(band) / Double(max(1, bandCount - 1))
                let frequency = 90.0 * pow(7_200.0 / 90.0, fraction)
                let angularStep = 2.0 * Double.pi * frequency / sampleRate
                var real = 0.0
                var imaginary = 0.0
                for sampleIndex in 0..<windowSize {
                    let source = min(samples.count - 1, start + sampleIndex)
                    let hann = 0.5 - 0.5 * cos(
                        2.0 * Double.pi * Double(sampleIndex) / Double(windowSize - 1)
                    )
                    let value = Double(samples[source]) * hann
                    let angle = angularStep * Double(sampleIndex)
                    real += value * cos(angle)
                    imaginary -= value * sin(angle)
                }
                let magnitude = Float(log1p(sqrt(real * real + imaginary * imaginary)))
                magnitudes[time * bandCount + band] = magnitude
                peak = max(peak, magnitude)
            }
        }
        if peak > 0 {
            for index in magnitudes.indices {
                magnitudes[index] = min(1, magnitudes[index] / peak)
            }
        }

        let waveColumns = 180
        var lows = Array(repeating: Float.zero, count: waveColumns)
        var highs = Array(repeating: Float.zero, count: waveColumns)
        let samplesPerColumn = max(1, samples.count / waveColumns)
        for column in 0..<waveColumns {
            let start = column * samplesPerColumn
            guard start < samples.count else { break }
            for index in start..<min(samples.count, start + samplesPerColumn) {
                lows[column] = min(lows[column], samples[index])
                highs[column] = max(highs[column], samples[index])
            }
        }

        self.init(
            timeBins: timeBins,
            bandCount: bandCount,
            magnitudes: magnitudes,
            waveLows: lows,
            waveHighs: highs
        )
    }
}
