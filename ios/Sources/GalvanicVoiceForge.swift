import SwiftUI

enum VoiceForgePhase: String, Sendable {
    case idle
    case readingBundle
    case hydratingWeights
    case chargingRuntime
    case checkingMemory
    case bindingText
    case forgingFrames
    case decodingAudio
    case denoising
    case cancelling
    case cancelled
    case complete
    case failed

    var title: String {
        switch self {
        case .idle: "Forge standing by"
        case .readingBundle: "Mapping the specimen"
        case .hydratingWeights: "Hydrating neural tissue"
        case .chargingRuntime: "Charging the runtime"
        case .checkingMemory: "Checking memory headroom"
        case .bindingText: "Binding words to the voice"
        case .forgingFrames: "Growing semantic voice frames"
        case .decodingAudio: "Turning frames into sound"
        case .denoising: "Polishing the final signal"
        case .cancelling: "Draining the forge"
        case .cancelled: "Forge safely stopped"
        case .complete: "Voice alive"
        case .failed: "The signal broke"
        }
    }

    var systemImage: String {
        switch self {
        case .idle: "bolt.horizontal.circle"
        case .readingBundle: "map"
        case .hydratingWeights: "brain.head.profile"
        case .chargingRuntime: "bolt.circle"
        case .checkingMemory: "memorychip"
        case .bindingText: "textformat.abc"
        case .forgingFrames: "waveform.path.ecg"
        case .decodingAudio: "waveform"
        case .denoising: "sparkles"
        case .cancelling: "bolt.slash"
        case .cancelled: "stop.circle"
        case .complete: "checkmark.seal.fill"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    var isActive: Bool {
        switch self {
        case .readingBundle, .hydratingWeights, .chargingRuntime, .checkingMemory,
             .bindingText, .forgingFrames, .decodingAudio, .denoising, .cancelling:
            true
        case .idle, .cancelled, .complete, .failed:
            false
        }
    }
}

struct VoiceForgeTelemetry: Sendable, Equatable {
    var phase: VoiceForgePhase = .idle
    var preparedTokens: UInt64 = 0
    var generatedFrames: UInt64 = 0
    var predictedMaximumFrames: UInt64 = 0
    var decodedFrames: UInt64 = 0
    var decodedSamples: UInt64 = 0
    var predictedPeakBytes: UInt64 = 0
    var memoryBudgetBytes: UInt64 = 0
    var synthesisMilliseconds: Double = 0
    var eventCount = 0

    mutating func reset(for phase: VoiceForgePhase) {
        self = VoiceForgeTelemetry(phase: phase)
    }

    mutating func apply(_ event: EngineProgress) {
        eventCount += 1
        switch (event.kind, event.stage) {
        case (.stageStarted, .modelBundle):
            phase = .readingBundle
        case (.stageStarted, .modelWeights):
            phase = .hydratingWeights
        case (.stageStarted, .runtime):
            phase = .chargingRuntime
        case (.admission, .resourceAdmission):
            phase = .checkingMemory
            predictedPeakBytes = event.current
            memoryBudgetBytes = event.total
            predictedMaximumFrames = event.detail
        case (.unit, .text):
            phase = .bindingText
            preparedTokens = event.current
        case (.unit, .frames):
            phase = .forgingFrames
            generatedFrames = event.current
            predictedMaximumFrames = event.total
        case (.unit, .codec):
            phase = .decodingAudio
            decodedFrames = event.current
            decodedSamples += event.detail
            if predictedMaximumFrames == 0 { predictedMaximumFrames = event.total }
        case (.stageFinished, .synthesis):
            synthesisMilliseconds = event.elapsedMilliseconds
        case (.health, .health) where event.outputInvalid:
            phase = .failed
        default:
            break
        }
    }

    var factualDetail: String {
        switch phase {
        case .readingBundle:
            "Resolving the verified model files"
        case .hydratingWeights:
            "Loading the local neural weights into memory"
        case .chargingRuntime:
            "Starting the on-device execution teams"
        case .checkingMemory:
            predictedMaximumFrames > 0
                ? "Admitted for up to \(predictedMaximumFrames) frames"
                : "Measuring this utterance against the memory budget"
        case .bindingText:
            "\(preparedTokens) prepared tokens entered the model"
        case .forgingFrames:
            predictedMaximumFrames > 0
                ? "\(generatedFrames) real frames emitted · ceiling \(predictedMaximumFrames)"
                : "\(generatedFrames) real frames emitted"
        case .decodingAudio:
            "\(decodedFrames) frames decoded · \(decodedSamples) samples produced"
        case .denoising:
            "The local neural denoiser is cleaning the generated waveform"
        case .cancelling:
            "Stopping cooperatively at the next safe frame boundary"
        case .cancelled:
            "No partial audio was published"
        case .complete:
            generatedFrames > 0 ? "\(generatedFrames) frames forged on this device" : "Finished on this device"
        case .failed:
            "Your text and voice remain available to try again"
        case .idle:
            "Choose a voice, write an utterance, and charge the forge"
        }
    }

    var accessibilitySummary: String {
        "\(phase.title). \(factualDetail)."
    }
}

/// The hero processing instrument. Its counts and topology come only from native
/// progress events; time drives shimmer and electrical travel, never completion.
struct GalvanicVoiceForge: View {
    let telemetry: VoiceForgeTelemetry
    let elapsed: TimeInterval
    var compact = false
    var cancel: (() -> Void)?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 12) {
                ZStack {
                    Circle()
                        .fill(phaseColor.opacity(0.13))
                        .frame(width: 42, height: 42)
                    Image(systemName: telemetry.phase.systemImage)
                        .font(.system(size: Lab.typeSize(17), weight: .bold))
                        .foregroundStyle(phaseColor)
                        .symbolEffect(
                            .pulse,
                            options: reduceMotion || !telemetry.phase.isActive
                                ? .nonRepeating : .repeating
                        )
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(telemetry.phase.title)
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(Lab.textPrimary)
                    Text(telemetry.factualDetail)
                        .font(.subheadline)
                        .foregroundStyle(Lab.textSecondary)
                        .contentTransition(.numericText())
                }
                Spacer(minLength: 8)
                if telemetry.phase.isActive {
                    Text(Self.elapsed(elapsed))
                        .font(.system(.caption, design: .monospaced, weight: .semibold))
                        .foregroundStyle(Lab.textSecondary)
                        .monospacedDigit()
                }
            }

            forgeCanvas
                .frame(height: compact ? 126 : 190)

            HStack(spacing: 10) {
                ForgeMetric(label: "TOKENS", value: telemetry.preparedTokens.formatted())
                ForgeMetric(label: "FRAMES", value: telemetry.generatedFrames.formatted())
                ForgeMetric(label: "DECODED", value: telemetry.decodedFrames.formatted())
                Spacer(minLength: 0)
                if let cancel, telemetry.phase.isActive, telemetry.phase != .cancelling {
                    Button(role: .cancel, action: cancel) {
                        Label("Stop", systemImage: "stop.fill")
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    .layoutPriority(2)
                }
            }
        }
        .padding(compact ? 14 : 18)
        .background {
            panelBackground
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [phaseColor.opacity(0.55), Color.white.opacity(0.04), phaseColor.opacity(0.18)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        }
        .shadow(color: phaseColor.opacity(telemetry.phase.isActive ? 0.22 : 0.08), radius: 30, y: 12)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Galvanic Voice Forge")
        .accessibilityValue(telemetry.accessibilitySummary)
    }

    private var forgeCanvas: some View {
        TimelineView(.animation(minimumInterval: cadence, paused: !telemetry.phase.isActive || reduceMotion)) { timeline in
            Canvas(opaque: false, colorMode: .linear, rendersAsynchronously: true) { context, size in
                drawForge(
                    context: &context,
                    size: size,
                    time: timeline.date.timeIntervalSinceReferenceDate
                )
            }
        }
        .background(Color.black.opacity(0.34), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.white.opacity(0.055), lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .accessibilityHidden(true)
    }

    private func drawForge(
        context: inout GraphicsContext,
        size: CGSize,
        time: TimeInterval
    ) {
        let phase = reduceMotion ? 0 : time * 1.45
        let midY = size.height * 0.5
        let leftX = size.width * 0.15
        let coilX = size.width * 0.47
        let rightX = size.width * 0.83
        let activeAlpha = telemetry.phase.isActive ? 1.0 : 0.58

        var rail = Path()
        rail.move(to: CGPoint(x: 24, y: midY))
        rail.addLine(to: CGPoint(x: size.width - 24, y: midY))
        context.stroke(
            rail,
            with: .linearGradient(
                Gradient(colors: [Lab.emerald.opacity(0.08), phaseColor.opacity(0.75), Lab.emerald.opacity(0.12)]),
                startPoint: CGPoint(x: 0, y: midY),
                endPoint: CGPoint(x: size.width, y: midY)
            ),
            style: StrokeStyle(lineWidth: 1.4, dash: [3, 7])
        )

        let tokenCount = min(8, Int(telemetry.preparedTokens))
        for index in 0..<tokenCount {
            let angle = (Double(index) / Double(tokenCount)) * Double.pi * 2 + phase * 0.18
            let radius = 23.0 + Double(index % 3) * 4
            let point = CGPoint(
                x: leftX + CGFloat(cos(angle) * radius),
                y: midY + CGFloat(sin(angle) * radius)
            )
            let rect = CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)
            context.fill(Path(ellipseIn: rect), with: .color(Lab.emerald.opacity(0.28 + activeAlpha * 0.45)))
        }

        for ring in 0..<4 {
            let radius = CGFloat(20 + ring * 10)
            let pulse = reduceMotion ? 0 : CGFloat(sin(phase * 1.4 + Double(ring)) * 2.5)
            let rect = CGRect(
                x: coilX - radius - pulse,
                y: midY - radius - pulse,
                width: (radius + pulse) * 2,
                height: (radius + pulse) * 2
            )
            context.stroke(
                Path(ellipseIn: rect),
                with: .color(phaseColor.opacity(0.16 + Double(ring) * 0.08)),
                style: StrokeStyle(lineWidth: ring == 1 ? 2.2 : 1, dash: ring.isMultiple(of: 2) ? [] : [4, 5])
            )
        }

        if telemetry.phase.isActive {
            context.addFilter(.shadow(color: phaseColor.opacity(0.75), radius: 7))
            for arcIndex in 0..<3 {
                var arc = Path()
                arc.move(to: CGPoint(x: leftX + 30, y: midY + CGFloat(arcIndex - 1) * 9))
                let segments = 9
                for segment in 1...segments {
                    let fraction = CGFloat(segment) / CGFloat(segments)
                    let x = leftX + 30 + (coilX - leftX - 64) * fraction
                    let wave = sin(phase * 4 + Double(segment * (arcIndex + 2))) * 7
                    arc.addLine(to: CGPoint(x: x, y: midY + CGFloat(wave) + CGFloat(arcIndex - 1) * 7))
                }
                context.stroke(arc, with: .color(phaseColor.opacity(0.30 + Double(arcIndex) * 0.17)), lineWidth: 1.2)
            }
        }

        let frameCount = min(10, Int(telemetry.generatedFrames))
        for index in 0..<frameCount {
            let spacing = min(17.0, Double((rightX - coilX - 52) / CGFloat(max(frameCount, 1))))
            let x = coilX + 42 + CGFloat(index) * CGFloat(spacing)
            let height = 10 + CGFloat((index * 7) % 17)
            let frameRect = CGRect(x: x, y: midY - height / 2, width: 8, height: height)
            context.fill(
                Path(roundedRect: frameRect, cornerRadius: 3),
                with: .color(phaseColor.opacity(0.30 + activeAlpha * 0.42))
            )
        }

        if telemetry.decodedFrames > 0 {
            var waveform = Path()
            let amplitude = 8 + min(20, CGFloat(telemetry.decodedFrames) * 0.7)
            let startX = max(coilX + 48, rightX - 54)
            waveform.move(to: CGPoint(x: startX, y: midY))
            let width = max(1, size.width - startX - 16)
            for sample in 1...36 {
                let f = CGFloat(sample) / 36
                let x = startX + width * f
                let envelope = sin(f * .pi)
                let y = midY + sin(Double(sample) * 0.88 + phase * 2.4) * amplitude * envelope
                waveform.addLine(to: CGPoint(x: x, y: y))
            }
            context.stroke(
                waveform,
                with: .color(Lab.emerald.opacity(0.82 * activeAlpha)),
                lineWidth: 1.7
            )
        }

        let labels: [(String, CGPoint)] = [
            ("UTTERANCE", CGPoint(x: leftX, y: size.height - 20)),
            ("VOICE CORE", CGPoint(x: coilX, y: size.height - 20)),
            ("AUDIO", CGPoint(x: rightX, y: size.height - 20)),
        ]
        for (label, point) in labels {
            context.draw(
                Text(label)
                    .font(.system(size: Lab.typeSize(8), weight: .bold, design: .monospaced))
                    .foregroundColor(Lab.textSecondary.opacity(0.7)),
                at: point
            )
        }
    }

    private var cadence: TimeInterval {
        let constrained = ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue
        return constrained ? 1.0 / 15.0 : 1.0 / 30.0
    }

    private var phaseColor: Color {
        switch telemetry.phase {
        case .failed: Lab.danger
        case .cancelling, .cancelled: Lab.textSecondary
        case .complete: Lab.emerald
        default: Lab.emerald
        }
    }

    @ViewBuilder
    private var panelBackground: some View {
        if reduceTransparency {
            Color.black.opacity(0.94)
        } else {
            ZStack {
                Color.black.opacity(0.62)
                LinearGradient(
                    colors: [phaseColor.opacity(0.12), Color.clear, Lab.emeraldDeep.opacity(0.12)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
        }
    }

    private static func elapsed(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}

private struct ForgeMetric: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: Lab.typeSize(8), weight: .bold, design: .monospaced))
                .kerning(1.2)
                .foregroundStyle(Lab.textSecondary)
            Text(value)
                .font(.system(size: Lab.typeSize(13), weight: .semibold, design: .monospaced))
                .foregroundStyle(Lab.textPrimary)
                .contentTransition(.numericText())
        }
        .accessibilityElement(children: .combine)
    }
}
