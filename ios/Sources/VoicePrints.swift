// Voiceprints: a deterministic visual identity for each 1,024-float x-vector.
//
// Two rules make similarity VISIBLE rather than decorative:
//
// 1. Every voice is drawn by the same fixed function of its vector — same pooling, same
//    projections, same palette mapping — so voices whose vectors are close produce
//    glyphs that genuinely look alike. No randomness, no per-voice seeds.
// 2. The constellation positions glyphs by classical multidimensional scaling over
//    pairwise cosine distances, so similar voices also sit near each other. Similarity
//    is double-encoded: in the shape and in the map.

import SwiftUI

enum VoiceMath {
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in 0..<min(a.count, b.count) {
            let x = Double(a[index])
            let y = Double(b[index])
            dot += x * y
            normA += x * x
            normB += y * y
        }
        let denominator = normA.squareRoot() * normB.squareRoot()
        return denominator > 0 ? dot / denominator : 0
    }

    /// Mean-pool the vector into `count` bins, then standardize the profile.
    /// Standardizing per voice keeps the SHAPE comparable while the palette (below)
    /// carries the magnitude information the normalization removes.
    static func pooledProfile(_ vector: [Float], bins: Int) -> [Double] {
        guard !vector.isEmpty else { return Array(repeating: 0, count: bins) }
        var pooled = [Double](repeating: 0, count: bins)
        for bin in 0..<bins {
            let start = bin * vector.count / bins
            let end = (bin + 1) * vector.count / bins
            var sum = 0.0
            for index in start..<max(end, start + 1) {
                sum += Double(vector[min(index, vector.count - 1)])
            }
            pooled[bin] = sum / Double(max(end - start, 1))
        }
        let mean = pooled.reduce(0, +) / Double(bins)
        let variance = pooled.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(bins)
        let deviation = max(variance.squareRoot(), 1e-9)
        return pooled.map { ($0 - mean) / deviation }
    }

    /// Deterministic pseudo-random unit vector for palette projections (fixed seed).
    static func projectionVector(dimension: Int, seed: UInt64) -> [Double] {
        var state = seed | 1
        var vector = [Double](repeating: 0, count: dimension)
        for index in 0..<dimension {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            vector[index] = Double(state >> 40) / Double(1 << 24) - 0.5
        }
        let norm = max(vector.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-9)
        return vector.map { $0 / norm }
    }

    static func project(_ vector: [Float], onto direction: [Double]) -> Double {
        var sum = 0.0
        for index in 0..<min(vector.count, direction.count) {
            sum += Double(vector[index]) * direction[index]
        }
        return sum
    }

    /// Classical MDS to 2-D over a small distance matrix (n is at most a few dozen).
    static func multidimensionalScaling(distances: [[Double]]) -> [CGPoint] {
        let n = distances.count
        guard n > 2 else {
            return (0..<n).map { CGPoint(x: Double($0) * 2 - 1, y: 0) }
        }
        // Double-centered Gram matrix from squared distances.
        var gram = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var rowMeans = [Double](repeating: 0, count: n)
        var grandMean = 0.0
        for i in 0..<n {
            for j in 0..<n {
                let d2 = distances[i][j] * distances[i][j]
                rowMeans[i] += d2 / Double(n)
                grandMean += d2 / Double(n * n)
            }
        }
        for i in 0..<n {
            for j in 0..<n {
                let d2 = distances[i][j] * distances[i][j]
                gram[i][j] = -0.5 * (d2 - rowMeans[i] - rowMeans[j] + grandMean)
            }
        }
        // Top-2 eigenvectors by power iteration with deflation.
        func powerIteration(_ matrix: [[Double]]) -> ([Double], Double) {
            var vector = (0..<n).map { Double(($0 * 37 + 11) % 17) - 8.0 }
            var value = 0.0
            for _ in 0..<64 {
                var next = [Double](repeating: 0, count: n)
                for i in 0..<n {
                    for j in 0..<n {
                        next[i] += matrix[i][j] * vector[j]
                    }
                }
                value = max(next.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-12)
                vector = next.map { $0 / value }
            }
            return (vector, value)
        }
        let (first, firstValue) = powerIteration(gram)
        var deflated = gram
        for i in 0..<n {
            for j in 0..<n {
                deflated[i][j] -= firstValue * first[i] * first[j]
            }
        }
        let (second, secondValue) = powerIteration(deflated)
        let scaleX = firstValue.squareRoot()
        let scaleY = max(secondValue, 0).squareRoot()
        return (0..<n).map { index in
            CGPoint(x: first[index] * scaleX, y: second[index] * scaleY)
        }
    }
}

/// The glyph: a smoothed radial profile of the vector, with a palette from fixed
/// projections. Pure function of the vector — identical input, identical picture.
struct VoicePrintGlyph: View {
    let vector: [Float]
    var lineWidth: CGFloat = 1.6

    private static let outerProjection = VoiceMath.projectionVector(dimension: 1024, seed: 0xF0A1)
    private static let innerProjection = VoiceMath.projectionVector(dimension: 1024, seed: 0xB007)

    var body: some View {
        let profile = VoiceMath.pooledProfile(vector, bins: 72)
        let hue = (atan2(
            VoiceMath.project(vector, onto: Self.outerProjection),
            VoiceMath.project(vector, onto: Self.innerProjection)
        ) / (2 * .pi) + 0.5)
        // The lab is emerald; voices claim the green-to-cyan-to-gold band around it so
        // the constellation stays on-theme while distinct voices stay distinguishable.
        let themedHue = 0.22 + hue * 0.28
        let color = Color(hue: themedHue, saturation: 0.75, brightness: 0.95)

        return Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let base = min(size.width, size.height) / 2

            func ring(scale: Double, amplitude: Double, offset: Int) -> Path {
                var path = Path()
                let bins = profile.count
                var points: [CGPoint] = []
                for index in 0...bins {
                    let value = profile[(index + offset) % bins]
                    let radius = base * scale * (1 + amplitude * tanh(value * 0.8))
                    let angle = Double(index) / Double(bins) * 2 * .pi - .pi / 2
                    points.append(
                        CGPoint(
                            x: center.x + radius * cos(angle),
                            y: center.y + radius * sin(angle)))
                }
                // Midpoint-smoothed closed curve.
                guard points.count > 2 else { return path }
                path.move(to: CGPoint(
                    x: (points[0].x + points[1].x) / 2, y: (points[0].y + points[1].y) / 2))
                for index in 1..<points.count {
                    let current = points[index]
                    let next = points[(index + 1) % points.count]
                    path.addQuadCurve(
                        to: CGPoint(x: (current.x + next.x) / 2, y: (current.y + next.y) / 2),
                        control: current)
                }
                path.closeSubpath()
                return path
            }

            let outer = ring(scale: 0.62, amplitude: 0.34, offset: 0)
            let inner = ring(scale: 0.32, amplitude: 0.30, offset: profile.count / 2)

            context.addFilter(.shadow(color: color.opacity(0.65), radius: 6))
            context.stroke(outer, with: .color(color), lineWidth: lineWidth)
            context.stroke(
                inner, with: .color(color.opacity(0.7)), lineWidth: lineWidth * 0.8)
            context.fill(
                Path(ellipseIn: CGRect(
                    x: center.x - 1.5, y: center.y - 1.5, width: 3, height: 3)),
                with: .color(color))
        }
        .accessibilityHidden(true)
    }
}
