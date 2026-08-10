// The Voice Constellation: every voice on one dark canvas.
//
// Positions come from MDS over cosine distances (similar voices cluster), the glyphs
// come from the vectors themselves (similar voices look alike), and threads connect
// the closest pairs. Tap a voice for its name and nearest neighbors.

import SwiftUI

private struct GalaxyEntry: Identifiable {
    let id: String
    let name: String
    let vector: [Float]
    let enrolled: Bool
    var position: CGPoint = .zero
}

struct VoiceGalaxyView: View {
    let presets: [Preset]
    let enrolled: [EnrolledVoice]
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var entries: [GalaxyEntry] = []
    @State private var similarity: [[Double]] = []
    @State private var selected: String?
    @State private var appeared = false

    var body: some View {
        ZStack {
            Lab.background.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        LabLabel(text: "The Voice Constellation")
                        Text("Nearby and alike means similar. Every shape is drawn from the voice's own 1,024 numbers.")
                            .font(.system(size: 12))
                            .foregroundStyle(Lab.textSecondary)
                    }
                    Spacer()
                    Button("Done") { dismiss() }
                        .buttonStyle(GhostButtonStyle())
                }
                .padding([.top, .horizontal], 18)

                GeometryReader { proxy in
                    let laidOut = layout(in: proxy.size)
                    ZStack {
                        // Threads between the most similar pairs, opacity by closeness.
                        Canvas { context, _ in
                            for i in 0..<laidOut.count {
                                for j in (i + 1)..<laidOut.count {
                                    let closeness = similarity.indices.contains(i)
                                        ? similarity[i][j] : 0
                                    guard closeness > 0.55 else { continue }
                                    var path = Path()
                                    path.move(to: laidOut[i].position)
                                    path.addLine(to: laidOut[j].position)
                                    context.stroke(
                                        path,
                                        with: .color(
                                            Lab.emerald.opacity((closeness - 0.55) * 0.9)),
                                        lineWidth: 1)
                                }
                            }
                        }
                        ForEach(laidOut) { entry in
                            VStack(spacing: 4) {
                                VoicePrintGlyph(vector: entry.vector)
                                    .frame(width: 74, height: 74)
                                Text(entry.name)
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundStyle(
                                        entry.enrolled ? Lab.emerald : Lab.textPrimary)
                                    .lineLimit(1)
                            }
                            .scaleEffect(selected == entry.id ? 1.18 : 1)
                            .opacity(appeared ? 1 : 0)
                            .position(entry.position)
                            .onTapGesture {
                                withAnimation(.snappy) {
                                    selected = selected == entry.id ? nil : entry.id
                                }
                            }
                            .accessibilityLabel(
                                "\(entry.name)\(entry.enrolled ? ", enrolled voice" : "")")
                            .accessibilityAddTraits(.isButton)
                        }
                    }
                    .animation(reduceMotion ? nil : .spring(duration: 0.7), value: appeared)
                }

                if let selected, let index = entries.firstIndex(where: { $0.id == selected }) {
                    neighborCard(for: index)
                        .padding(.horizontal, 18)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .padding(.bottom, 14)
        }
        .presentationDetents([.large])
        .onAppear {
            build()
            withAnimation { appeared = true }
        }
    }

    private func build() {
        var built: [GalaxyEntry] = presets.compactMap { preset in
            guard let vector = try? Engine.presetVector(named: preset.name) else { return nil }
            return GalaxyEntry(
                id: "preset:\(preset.name)", name: preset.name, vector: vector, enrolled: false)
        }
        built += enrolled.map { voice in
            GalaxyEntry(
                id: "voice:\(voice.id)", name: voice.name, vector: voice.vector, enrolled: true)
        }
        let n = built.count
        var matrix = [[Double]](repeating: [Double](repeating: 1, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                matrix[i][j] = VoiceMath.cosineSimilarity(built[i].vector, built[j].vector)
            }
        }
        similarity = matrix
        let distances = matrix.map { row in row.map { max(0, 1 - $0) } }
        let raw = VoiceMath.multidimensionalScaling(distances: distances)
        for index in 0..<n {
            built[index].position = raw[index]
        }
        entries = built
    }

    /// Fit MDS coordinates into the canvas with padding, then relax overlaps.
    private func layout(in size: CGSize) -> [GalaxyEntry] {
        guard !entries.isEmpty, size.width > 10 else { return [] }
        let xs = entries.map(\.position.x)
        let ys = entries.map(\.position.y)
        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 1
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 1
        let spanX = max(maxX - minX, 1e-6)
        let spanY = max(maxY - minY, 1e-6)
        let inset: CGFloat = 64
        var placed = entries.map { entry in
            var out = entry
            out.position = CGPoint(
                x: inset + (entry.position.x - minX) / spanX * (size.width - 2 * inset),
                y: inset + (entry.position.y - minY) / spanY * (size.height - 2 * inset))
            return out
        }
        // A few repulsion passes so labels never sit on top of each other; MDS decides
        // the neighborhood, this only resolves collisions inside it.
        let minimumDistance: CGFloat = 86
        for _ in 0..<24 {
            for i in 0..<placed.count {
                for j in (i + 1)..<placed.count {
                    let dx = placed[j].position.x - placed[i].position.x
                    let dy = placed[j].position.y - placed[i].position.y
                    let distance = max((dx * dx + dy * dy).squareRoot(), 1e-3)
                    guard distance < minimumDistance else { continue }
                    let push = (minimumDistance - distance) / 2
                    let ux = dx / distance
                    let uy = dy / distance
                    placed[i].position.x -= ux * push
                    placed[i].position.y -= uy * push
                    placed[j].position.x += ux * push
                    placed[j].position.y += uy * push
                }
            }
            for index in 0..<placed.count {
                placed[index].position.x = min(max(placed[index].position.x, inset), size.width - inset)
                placed[index].position.y = min(max(placed[index].position.y, inset), size.height - inset)
            }
        }
        return placed
    }

    private func neighborCard(for index: Int) -> some View {
        let neighbors = entries.indices
            .filter { $0 != index }
            .sorted { similarity[index][$0] > similarity[index][$1] }
            .prefix(3)
        return LabPanel {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    VoicePrintGlyph(vector: entries[index].vector)
                        .frame(width: 40, height: 40)
                    Text(entries[index].name)
                        .font(.system(size: 16, weight: .black))
                        .foregroundStyle(Lab.textPrimary)
                }
                ForEach(Array(neighbors), id: \.self) { other in
                    HStack {
                        Text(entries[other].name)
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundStyle(Lab.textSecondary)
                        Spacer()
                        Text("\(Int((similarity[index][other] * 100).rounded()))% similar")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Lab.emerald)
                    }
                }
            }
        }
    }
}
