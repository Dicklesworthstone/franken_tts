// The voice mosaic: 4 KB of x-vector as a self-locating field of emerald cells.
//
// Design, and why:
// - 144×144 cells, 2 bits per cell as one of FOUR brightness levels on a constant-hue
//   emerald ramp. JPEG keeps luminance at full resolution and halves chroma, so the
//   information rides in the channel compression treats best; the constant hue is also
//   what makes the code look like the lab instead of a barcode. Four levels per cell is
//   what makes this denser than a QR code's one bit per module.
// - Three QR-style finder patterns (bright/dark/bright concentric squares) in the
//   corners. The decoder scans the WHOLE image for their 1:1:3:1:1 run signature, so a
//   screenshot from any iPhone — any size, any aspect ratio, status bar and all —
//   registers automatically. Shared images stay axis-aligned (the channel is a file,
//   not a camera), so scale and offset per axis is the full geometry.
// - A known calibration row teaches the decoder the four brightness levels as they
//   survived compression, and refines the grid registration by a small local search.
// - Interleaved Reed-Solomon (255, 223): compression artifacts are local, the
//   interleave spreads any damaged patch across all 19 blocks, and each block corrects
//   up to 16 wrong bytes. A CRC over the plaintext refuses silent miscorrection.
//
// The core is pure byte arrays in and out, so the exact codec is testable off-device.

import Foundation

enum VoiceCode {
    static let gridN = 144
    static let cardSize = 1024
    /// The mosaic square inside the rendered card: integer 6-px cells, 80-px quiet margin.
    static let regionOrigin = 80
    static let cellPx = 6

    /// Emerald ramp, darkest to brightest. Constant hue family, widely separated luma.
    static let levels: [(r: UInt8, g: UInt8, b: UInt8)] = [
        (6, 22, 15), (26, 92, 61), (56, 168, 110), (112, 248, 170),
    ]

    private static let magic = Array("FV02".utf8)
    private static let dataBytesPerBlock = 223
    private static let parityBytesPerBlock = 32
    private static let blocks = 19
    static var plaintextCapacity: Int { dataBytesPerBlock * blocks } // 4237

    private static let finderSpan = 7 // finder pattern cells, plus a 1-cell separator
    private static let calibrationRow = 8
    private static let calibrationColumn = 8

    // ------------------------------------------------------------------ layout

    /// Finder + separator zones: top-left, top-right, bottom-left (8×8 cells each).
    private static func inFinderZone(row: Int, column: Int) -> Bool {
        let zone = finderSpan + 1
        if row < zone && column < zone { return true }
        if row < zone && column >= gridN - zone { return true }
        if row >= gridN - zone && column < zone { return true }
        return false
    }

    /// The calibration column pins the VERTICAL scale the way the row pins the
    /// horizontal one; it runs between the top-left and bottom-left zones.
    private static func inCalibrationColumn(row: Int, column: Int) -> Bool {
        column == calibrationColumn && row > calibrationRow
            && row < gridN - finderSpan - 1
    }

    private static func isReserved(row: Int, column: Int) -> Bool {
        row == calibrationRow || inCalibrationColumn(row: row, column: column)
            || inFinderZone(row: row, column: column)
    }

    /// Level for a reserved cell: finder rings or the known calibration patterns.
    private static func reservedLevel(row: Int, column: Int) -> Int {
        if row == calibrationRow { return column % 4 }
        if inCalibrationColumn(row: row, column: column) { return row % 4 }
        // Local coordinates within whichever finder square this is.
        var r = row
        var c = column
        if column >= gridN - finderSpan - 1 { c = column - (gridN - finderSpan) }
        if row >= gridN - finderSpan - 1 { r = row - (gridN - finderSpan) }
        guard (0..<finderSpan).contains(r), (0..<finderSpan).contains(c) else {
            return 0 // separator ring: darkest
        }
        let ring = min(r, c, finderSpan - 1 - r, finderSpan - 1 - c)
        return ring == 1 ? 0 : 3 // bright border, dark ring, bright 3×3 core
    }

    // ------------------------------------------------------------------ encode

    /// Encode name + vector into an RGB24 mosaic image (cardSize × cardSize).
    static func renderMosaicPixels(name: String, vector: [Float]) -> [UInt8] {
        var plaintext = magic
        let nameBytes = Array(name.utf8.prefix(64))
        plaintext += [UInt8(nameBytes.count >> 8), UInt8(nameBytes.count & 0xFF)]
        plaintext += nameBytes
        for value in vector {
            plaintext += withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
        }
        let crc = crc32(plaintext)
        plaintext += [
            UInt8(crc >> 24 & 0xFF), UInt8(crc >> 16 & 0xFF),
            UInt8(crc >> 8 & 0xFF), UInt8(crc & 0xFF),
        ]
        precondition(plaintext.count <= plaintextCapacity, "payload exceeds the mosaic")
        plaintext += [UInt8](repeating: 0, count: plaintextCapacity - plaintext.count)

        // Reed-Solomon per block, then byte-interleave across blocks.
        var blocksOut = [[UInt8]]()
        for block in 0..<blocks {
            let start = block * dataBytesPerBlock
            let data = Array(plaintext[start..<start + dataBytesPerBlock])
            blocksOut.append(data + ReedSolomon.parity(for: data))
        }
        var coded = [UInt8]()
        for position in 0..<(dataBytesPerBlock + parityBytesPerBlock) {
            for block in 0..<blocks {
                coded.append(blocksOut[block][position])
            }
        }
        whiten(&coded)

        var pixels = [UInt8](repeating: 0, count: cardSize * cardSize * 3)
        let background = levels[0]
        for index in stride(from: 0, to: pixels.count, by: 3) {
            pixels[index] = background.r / 2
            pixels[index + 1] = background.g / 2
            pixels[index + 2] = background.b / 2
        }
        var bitCursor = 0
        let totalBits = coded.count * 8
        for row in 0..<gridN {
            for column in 0..<gridN {
                let level: Int
                if isReserved(row: row, column: column) {
                    level = reservedLevel(row: row, column: column)
                } else if bitCursor + 2 <= totalBits {
                    let byte = Int(coded[bitCursor >> 3])
                    let shift = 6 - (bitCursor & 7)
                    level = (byte >> shift) & 0b11
                    bitCursor += 2
                } else {
                    // Deterministic filler, hashed so it blends with the data field.
                    var hash = UInt64(row * gridN + column) &* 0x9E37_79B9_7F4A_7C15
                    hash ^= hash >> 29
                    hash = hash &* 0xBF58_476D_1CE4_E5B9
                    level = Int(hash >> 62)
                }
                paintCell(&pixels, row: row, column: column, level: level)
            }
        }
        return pixels
    }

    /// XOR the coded stream with a fixed pseudo-random mask. The floats in the
    /// payload repeat byte patterns that would otherwise show as visible stripes,
    /// and a pathological payload could produce large flat regions that starve the
    /// decoder's threshold estimate — masking keeps the field uniformly mixed.
    private static func whiten(_ bytes: inout [UInt8]) {
        var state: UInt64 = 0x5DEE_CE66_D1CE_F001
        for index in bytes.indices {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            bytes[index] ^= UInt8(truncatingIfNeeded: state >> 32)
        }
    }

    private static func paintCell(_ pixels: inout [UInt8], row: Int, column: Int, level: Int) {
        let color = levels[level]
        let x0 = regionOrigin + column * cellPx
        let y0 = regionOrigin + row * cellPx
        for y in y0..<(y0 + cellPx) {
            var at = (y * cardSize + x0) * 3
            for _ in 0..<cellPx {
                pixels[at] = color.r
                pixels[at + 1] = color.g
                pixels[at + 2] = color.b
                at += 3
            }
        }
    }

    // ------------------------------------------------------------------ decode

    /// Decode a voice from RGB24 pixels of any image containing the mosaic:
    /// original PNG, recompressed JPEG, or a screenshot from any size of phone.
    static func decode(pixels: [UInt8], width: Int, height: Int) -> (String, [Float])? {
        guard width >= gridN, height >= gridN, pixels.count >= width * height * 3 else {
            return nil
        }
        var luma = [Float](repeating: 0, count: width * height)
        var minLuma = 255.0
        var maxLuma = 0.0
        for index in 0..<(width * height) {
            let at = index * 3
            let value = 0.299 * Double(pixels[at]) + 0.587 * Double(pixels[at + 1])
                + 0.114 * Double(pixels[at + 2])
            luma[index] = Float(value)
            minLuma = min(minLuma, value)
            maxLuma = max(maxLuma, value)
        }
        guard maxLuma - minLuma > 30 else { return nil }
        let threshold = (minLuma + maxLuma) / 2

        let finders = findFinderPatterns(
            luma: luma, width: width, height: height, threshold: threshold)
        // Data cells can imitate a finder by chance, so several plausible triples may
        // exist; try them best-first — the calibration fit and the CRC reject impostor
        // grids, so the first one that decodes is the real one.
        for triple in rankFinderTriples(finders) {
            if let decoded = decodeGrid(
                triple: triple, luma: luma, width: width, height: height,
                threshold: threshold) {
                return decoded
            }
        }
        return nil
    }

    private static func decodeGrid(
        triple: (FinderCandidate, FinderCandidate, FinderCandidate),
        luma: [Float], width: Int, height: Int, threshold: Double
    ) -> (String, [Float])? {
        func bilinear(_ x: Double, _ y: Double) -> Double {
            let cx = min(max(x, 0), Double(width - 1))
            let cy = min(max(y, 0), Double(height - 1))
            let x0 = min(Int(cx), width - 2)
            let y0 = min(Int(cy), height - 2)
            let fx = cx - Double(x0)
            let fy = cy - Double(y0)
            let a = Double(luma[y0 * width + x0])
            let b = Double(luma[y0 * width + x0 + 1])
            let c = Double(luma[(y0 + 1) * width + x0])
            let d = Double(luma[(y0 + 1) * width + x0 + 1])
            return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
                + c * (1 - fx) * fy + d * fx * fy
        }

        /// Sub-pixel refinement of a finder center: brightness-weighted centroid over
        /// a symmetric window that covers the finder but stops inside its separator.
        func refined(_ finder: FinderCandidate) -> (x: Double, y: Double) {
            let reach = finder.module * 3.4
            let x0 = max(Int((finder.x - reach).rounded()), 0)
            let x1 = min(Int((finder.x + reach).rounded()), width - 1)
            let y0 = max(Int((finder.y - reach).rounded()), 0)
            let y1 = min(Int((finder.y + reach).rounded()), height - 1)
            guard x1 > x0, y1 > y0 else { return (finder.x, finder.y) }
            var weightSum = 0.0
            var sumX = 0.0
            var sumY = 0.0
            for y in y0...y1 {
                for x in x0...x1 {
                    let weight = max(Double(luma[y * width + x]) - threshold, 0)
                    weightSum += weight
                    sumX += weight * Double(x)
                    sumY += weight * Double(y)
                }
            }
            guard weightSum > 0 else { return (finder.x, finder.y) }
            return (sumX / weightSum, sumY / weightSum)
        }

        let topLeft = refined(triple.0)
        let topRight = refined(triple.1)
        let bottomLeft = refined(triple.2)

        // Grid geometry: cell k's center sits at (k + 0.5) cell-units from the region
        // edge; the finder centers sit at cell coordinate 3.5 and gridN − 3.5.
        let centerSpan = Double(gridN - finderSpan)
        let cellW = (topRight.x - topLeft.x) / centerSpan
        let cellH = (bottomLeft.y - topLeft.y) / centerSpan
        guard cellW > 1.2, cellH > 1.2 else { return nil } // near 1 px/cell is gone

        // Cells much bigger than the resampling blur average well over a small
        // footprint; cells near 3 px are only trustworthy at their very center. Try
        // the footprint suited to this scale first, the other as a fallback.
        let footprints = cellW < 4.5 ? [0.0, 0.22] : [0.22, 0.0]
        for footprint in footprints {
            if let decoded = attempt(footprint: footprint) {
                return decoded
            }
        }
        return nil

        /// One full fit + sample + error-correct pass with a fixed sample footprint.
        func attempt(footprint: Double) -> (String, [Float])? {

        /// Matched box sample of one cell, anchored at the top-left finder so a scale
        /// tweak grows outward from a fixed point.
        func sample(
            row: Double, column: Double,
            scaleX: Double, scaleY: Double, dx: Double, dy: Double
        ) -> Double {
            let x = topLeft.x + (column - 3.0) * cellW * scaleX + dx
            let y = topLeft.y + (row - 3.0) * cellH * scaleY + dy
            guard footprint > 0 else { return bilinear(x, y) }
            let rx = cellW * footprint
            let ry = cellH * footprint
            return (bilinear(x, y)
                + bilinear(x - rx, y - ry) + bilinear(x + rx, y - ry)
                + bilinear(x - rx, y + ry) + bilinear(x + rx, y + ry)) / 5
        }

        /// Fit one axis against its known calibration strip: search a small offset and
        /// scale range. The score is separation between adjacent levels MINUS the
        /// spread within each level — a misaligned grid still averages to clean means
        /// (the strip repeats every four flat cells) but bleeds neighboring cells into
        /// individual samples, and the variance term is what catches that.
        func fitAxis(
            horizontal: Bool
        ) -> (score: Double, scale: Double, offset: Double, means: [Double])? {
            var best: (score: Double, scale: Double, offset: Double, means: [Double])?
            for scaleStep in -3...3 {
                let scale = 1.0 + Double(scaleStep) * 0.0018
                for offsetStep in -4...4 {
                    let offset = Double(offsetStep) * (horizontal ? cellW : cellH) * 0.1
                    var sums = [Double](repeating: 0, count: 4)
                    var squares = [Double](repeating: 0, count: 4)
                    var counts = [Double](repeating: 0, count: 4)
                    if horizontal {
                        for column in 0..<gridN {
                            let value = sample(
                                row: Double(calibrationRow), column: Double(column),
                                scaleX: scale, scaleY: 1, dx: offset, dy: 0)
                            sums[column % 4] += value
                            squares[column % 4] += value * value
                            counts[column % 4] += 1
                        }
                    } else {
                        for row in (calibrationRow + 1)..<(gridN - finderSpan - 1) {
                            let value = sample(
                                row: Double(row), column: Double(calibrationColumn),
                                scaleX: 1, scaleY: scale, dx: 0, dy: offset)
                            sums[row % 4] += value
                            squares[row % 4] += value * value
                            counts[row % 4] += 1
                        }
                    }
                    let means = zip(sums, counts).map { $0 / max($1, 1) }
                    var spread = 0.0
                    for level in 0..<4 {
                        let count = max(counts[level], 1)
                        let variance = max(
                            squares[level] / count - means[level] * means[level], 0)
                        spread += variance.squareRoot() / 4
                    }
                    let gaps = [
                        means[1] - means[0], means[2] - means[1], means[3] - means[2],
                    ]
                    let score = (gaps.min() ?? -1) - 2 * spread
                    if score > (best?.score ?? -.infinity) {
                        best = (score, scale, offset, means)
                    }
                }
            }
            return best
        }

        guard let fitX = fitAxis(horizontal: true), fitX.score > 4,
            let fitY = fitAxis(horizontal: false), fitY.score > 4
        else { return nil }
        let means = zip(fitX.means, fitY.means).map { ($0 + $1) / 2 }
        let thresholds = [
            (means[0] + means[1]) / 2,
            (means[1] + means[2]) / 2,
            (means[2] + means[3]) / 2,
        ]

        // Sample every data cell in layout order.
        let codedCount = (dataBytesPerBlock + parityBytesPerBlock) * blocks
        var coded = [UInt8](repeating: 0, count: codedCount)
        var bitCursor = 0
        let totalBits = codedCount * 8
        outer: for row in 0..<gridN {
            for column in 0..<gridN {
                if isReserved(row: row, column: column) { continue }
                if bitCursor + 2 > totalBits { break outer }
                let value = sample(
                    row: Double(row), column: Double(column),
                    scaleX: fitX.scale, scaleY: fitY.scale, dx: fitX.offset, dy: fitY.offset)
                var level = 0
                for threshold in thresholds where value > threshold {
                    level += 1
                }
                let shift = 6 - (bitCursor & 7)
                coded[bitCursor >> 3] |= UInt8(level << shift)
                bitCursor += 2
            }
        }

        // Unmask, then de-interleave and correct each block.
        whiten(&coded)
        var plaintext = [UInt8]()
        for block in 0..<blocks {
            var received = [UInt8]()
            for position in 0..<(dataBytesPerBlock + parityBytesPerBlock) {
                received.append(coded[position * blocks + block])
            }
            guard let corrected = ReedSolomon.correct(received) else { return nil }
            plaintext += corrected.prefix(dataBytesPerBlock)
        }
        return parse(plaintext)
        }
    }

    // ------------------------------------------------------------------ finder search

    struct FinderCandidate {
        var x: Double
        var y: Double
        var module: Double
        var votes: Int
    }

    /// Scan rows for the bright-dark-bright(3)-dark-bright 1:1:3:1:1 run signature,
    /// verify each hit vertically, and cluster the survivors.
    private static func findFinderPatterns(
        luma: [Float], width: Int, height: Int, threshold: Double
    ) -> [FinderCandidate] {
        var candidates = [FinderCandidate]()
        let rowStep = max(1, height / 700)

        func matchesRatio(_ runs: [Double], module: Double) -> Bool {
            guard module > 0.6 else { return false }
            return abs(runs[0] - module) < module * 0.65
                && abs(runs[1] - module) < module * 0.65
                && abs(runs[2] - 3 * module) < module * 1.2
                && abs(runs[3] - module) < module * 0.65
                && abs(runs[4] - module) < module * 0.65
        }

        /// Vertical confirmation at a candidate x: bright core of ~3 modules with a
        /// dark ring then a bright ring above and below. Returns center y and module.
        func verticalCheck(x: Int, y: Int, module: Double) -> (Double, Double)? {
            func bright(_ yy: Int) -> Bool { Double(luma[yy * width + x]) > threshold }
            guard bright(y) else { return nil }
            let limit = Int(module * 6) + 2
            var top = y
            while top > 0 && bright(top - 1) && y - top < limit { top -= 1 }
            var bottom = y
            while bottom < height - 1 && bright(bottom + 1) && bottom - y < limit {
                bottom += 1
            }
            let core = Double(bottom - top + 1)
            guard abs(core - 3 * module) < module * 1.6 else { return nil }
            func ringSpan(from start: Int, step: Int) -> (dark: Int, brightRun: Int)? {
                var yy = start + step
                var dark = 0
                while yy >= 0 && yy < height && !bright(yy) && dark < limit {
                    dark += 1
                    yy += step
                }
                var brightRun = 0
                while yy >= 0 && yy < height && bright(yy) && brightRun < limit {
                    brightRun += 1
                    yy += step
                }
                return dark > 0 && brightRun > 0 ? (dark, brightRun) : nil
            }
            guard let above = ringSpan(from: top, step: -1),
                let below = ringSpan(from: bottom, step: 1),
                abs(Double(above.dark) - module) < module * 0.9,
                abs(Double(below.dark) - module) < module * 0.9,
                abs(Double(above.brightRun) - module) < module * 0.9,
                abs(Double(below.brightRun) - module) < module * 0.9
            else { return nil }
            return ((Double(top) + Double(bottom)) / 2, core / 3)
        }

        for y in stride(from: 0, to: height, by: rowStep) {
            let rowBase = y * width
            var runs = [Double]()
            var runIsBright = [Bool]()
            var current = Double(luma[rowBase]) > threshold
            var length = 1.0
            for x in 1..<width {
                let bright = Double(luma[rowBase + x]) > threshold
                if bright == current {
                    length += 1
                } else {
                    runs.append(length)
                    runIsBright.append(current)
                    current = bright
                    length = 1
                }
            }
            runs.append(length)
            runIsBright.append(current)

            var position = 0.0
            for index in 0..<runs.count {
                defer { position += runs[index] }
                guard index + 4 < runs.count, runIsBright[index] else { continue }
                let window = Array(runs[index...index + 4])
                let module = window.reduce(0, +) / 7
                guard matchesRatio(window, module: module) else { continue }
                let centerX = position + window.reduce(0, +) / 2
                let xi = min(max(Int(centerX.rounded()), 0), width - 1)
                guard let (centerY, moduleV) = verticalCheck(x: xi, y: y, module: module),
                    moduleV / module > 0.5, moduleV / module < 2
                else { continue }
                var merged = false
                for at in 0..<candidates.count {
                    let candidate = candidates[at]
                    if abs(candidate.x - centerX) < module * 4,
                        abs(candidate.y - centerY) < module * 4 {
                        let weight = Double(candidate.votes)
                        candidates[at].x = (candidate.x * weight + centerX) / (weight + 1)
                        candidates[at].y = (candidate.y * weight + centerY) / (weight + 1)
                        candidates[at].module =
                            (candidate.module * weight + module) / (weight + 1)
                        candidates[at].votes += 1
                        merged = true
                        break
                    }
                }
                if !merged {
                    candidates.append(
                        FinderCandidate(x: centerX, y: centerY, module: module, votes: 1))
                }
            }
        }
        return candidates.filter { $0.votes >= 2 }
    }

    /// Rank plausible (topLeft, topRight, bottomLeft) triples, best first. The layout
    /// itself is the constraint: arms are axis-aligned, near-equal (shared images keep
    /// their aspect), and exactly gridN − finderSpan cells long, so each arm must be
    /// about 137× the finder's own module size.
    private static func rankFinderTriples(
        _ candidates: [FinderCandidate]
    ) -> [(FinderCandidate, FinderCandidate, FinderCandidate)] {
        guard candidates.count >= 3 else { return [] }
        // The search below is cubic; a busy photo can spawn hundreds of accidental
        // candidates, so rank only the most-voted few dozen.
        let candidates = Array(
            candidates.sorted { $0.votes > $1.votes }.prefix(48))
        let armCells = Double(gridN - finderSpan)
        var ranked =
            [(score: Double, triple: (FinderCandidate, FinderCandidate, FinderCandidate))]()
        for i in 0..<candidates.count {
            for j in 0..<candidates.count where j != i {
                for k in 0..<candidates.count where k != i && k != j {
                    let corner = candidates[i]
                    let right = candidates[j]
                    let down = candidates[k]
                    let armX = (right.x - corner.x, right.y - corner.y)
                    let armY = (down.x - corner.x, down.y - corner.y)
                    guard armX.0 > 0, armY.1 > 0 else { continue } // orientation
                    let lengthX = (armX.0 * armX.0 + armX.1 * armX.1).squareRoot()
                    let lengthY = (armY.0 * armY.0 + armY.1 * armY.1).squareRoot()
                    guard lengthX > 10, lengthY > 10 else { continue }
                    guard abs(armX.1) < lengthX * 0.1, abs(armY.0) < lengthY * 0.1,
                        lengthX / lengthY > 0.9, lengthX / lengthY < 1.11
                    else { continue }
                    let modules = [corner.module, right.module, down.module]
                    let moduleSpread = (modules.max() ?? 1) / max(modules.min() ?? 1, 0.1)
                    guard moduleSpread < 1.5 else { continue }
                    let meanModule = modules.reduce(0, +) / 3
                    let armError = max(
                        abs(lengthX / (meanModule * armCells) - 1),
                        abs(lengthY / (meanModule * armCells) - 1))
                    guard armError < 0.25 else { continue }
                    let score = Double(corner.votes + right.votes + down.votes)
                        - moduleSpread - armError * 10
                    ranked.append((score, (corner, right, down)))
                }
            }
        }
        return ranked.sorted { $0.score > $1.score }.prefix(8).map(\.triple)
    }

    // ------------------------------------------------------------------ payload

    private static func parse(_ plaintext: [UInt8]) -> (String, [Float])? {
        guard plaintext.count >= magic.count + 2,
            Array(plaintext.prefix(magic.count)) == magic
        else { return nil }
        var at = magic.count
        let nameLength = Int(plaintext[at]) << 8 | Int(plaintext[at + 1])
        at += 2
        let vectorBytes = 1024 * 4
        guard nameLength <= 64, plaintext.count >= at + nameLength + vectorBytes + 4 else {
            return nil
        }
        let name = String(decoding: plaintext[at..<at + nameLength], as: UTF8.self)
        at += nameLength
        var vector = [Float](repeating: 0, count: 1024)
        for index in 0..<1024 {
            let base = at + index * 4
            let bits = UInt32(plaintext[base]) | UInt32(plaintext[base + 1]) << 8
                | UInt32(plaintext[base + 2]) << 16 | UInt32(plaintext[base + 3]) << 24
            vector[index] = Float(bitPattern: bits)
        }
        at += vectorBytes
        let stored = UInt32(plaintext[at]) << 24 | UInt32(plaintext[at + 1]) << 16
            | UInt32(plaintext[at + 2]) << 8 | UInt32(plaintext[at + 3])
        guard crc32(Array(plaintext[0..<at])) == stored,
            vector.allSatisfy(\.isFinite)
        else { return nil }
        return (name.isEmpty ? "shared voice" : name, vector)
    }

    static func crc32(_ bytes: [UInt8]) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in bytes {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}

/// Reed-Solomon (255, 223) over GF(2^8), generator polynomial roots α^0..α^31.
enum ReedSolomon {
    static let parityCount = 32

    private static let field: (exp: [UInt8], log: [UInt8]) = {
        var exp = [UInt8](repeating: 0, count: 512)
        var log = [UInt8](repeating: 0, count: 256)
        var x = 1
        for power in 0..<255 {
            exp[power] = UInt8(x)
            log[x] = UInt8(power)
            x <<= 1
            if x & 0x100 != 0 { x ^= 0x11D }
        }
        for power in 255..<512 {
            exp[power] = exp[power - 255]
        }
        return (exp, log)
    }()

    private static func multiply(_ a: UInt8, _ b: UInt8) -> UInt8 {
        guard a != 0, b != 0 else { return 0 }
        return field.exp[Int(field.log[Int(a)]) + Int(field.log[Int(b)])]
    }

    private static let generator: [UInt8] = {
        var poly: [UInt8] = [1]
        for root in 0..<parityCount {
            let alpha = field.exp[root]
            var next = [UInt8](repeating: 0, count: poly.count + 1)
            for (index, coefficient) in poly.enumerated() {
                next[index] ^= multiply(coefficient, alpha)
                next[index + 1] ^= coefficient
            }
            poly = next
        }
        return poly.reversed() // highest degree first
    }()

    /// Parity bytes for a data block (systematic encoding).
    static func parity(for data: [UInt8]) -> [UInt8] {
        var remainder = [UInt8](repeating: 0, count: parityCount)
        for byte in data {
            let factor = byte ^ remainder[0]
            remainder.removeFirst()
            remainder.append(0)
            if factor != 0 {
                for index in 0..<parityCount {
                    remainder[index] ^= multiply(generator[index + 1], factor)
                }
            }
        }
        return remainder
    }

    /// Correct up to 16 byte errors in a 255-byte codeword. Nil when unrecoverable.
    static func correct(_ received: [UInt8]) -> [UInt8]? {
        let n = received.count
        var syndromes = [UInt8](repeating: 0, count: parityCount)
        var clean = true
        for index in 0..<parityCount {
            var value: UInt8 = 0
            for byte in received {
                value = multiply(value, field.exp[index]) ^ byte
            }
            syndromes[index] = value
            if value != 0 { clean = false }
        }
        if clean { return received }

        // Berlekamp-Massey for the error locator polynomial.
        var sigma: [UInt8] = [1]
        var previous: [UInt8] = [1]
        var discrepancyLast: UInt8 = 1
        var m = 1
        for step in 0..<parityCount {
            var discrepancy = syndromes[step]
            for index in 1..<sigma.count where step >= index {
                discrepancy ^= multiply(sigma[index], syndromes[step - index])
            }
            if discrepancy == 0 {
                m += 1
                continue
            }
            let scale = multiply(discrepancy, inverse(discrepancyLast))
            var shifted = [UInt8](repeating: 0, count: m) + previous
            for index in 0..<shifted.count {
                shifted[index] = multiply(shifted[index], scale)
            }
            if 2 * (sigma.count - 1) <= step {
                let old = sigma
                sigma = xorPolynomials(sigma, shifted)
                previous = old
                discrepancyLast = discrepancy
                m = 1
            } else {
                sigma = xorPolynomials(sigma, shifted)
                m += 1
            }
        }
        let errorCount = sigma.count - 1
        guard errorCount > 0, errorCount <= parityCount / 2 else { return nil }

        // Chien search for error positions.
        var positions = [Int]()
        for position in 0..<n {
            let xInverse = field.exp[(255 - (n - 1 - position)) % 255]
            var value: UInt8 = 0
            for (index, coefficient) in sigma.enumerated() {
                value ^= multiply(coefficient, power(xInverse, index))
            }
            if value == 0 {
                positions.append(position)
            }
        }
        guard positions.count == errorCount else { return nil }

        // Forney magnitudes: omega = (syndromes · sigma) mod x^parityCount.
        var omega = [UInt8](repeating: 0, count: parityCount)
        for index in 0..<parityCount {
            var value: UInt8 = 0
            for j in 0..<sigma.count where index >= j {
                value ^= multiply(sigma[j], syndromes[index - j])
            }
            omega[index] = value
        }
        var corrected = received
        for position in positions {
            let xInverse = field.exp[(255 - (n - 1 - position)) % 255]
            var numerator: UInt8 = 0
            for (index, coefficient) in omega.enumerated() {
                numerator ^= multiply(coefficient, power(xInverse, index))
            }
            var denominator: UInt8 = 0
            var index = 1
            while index < sigma.count {
                denominator ^= multiply(sigma[index], power(xInverse, index - 1))
                index += 2
            }
            guard denominator != 0 else { return nil }
            corrected[position] ^= multiply(numerator, inverse(denominator))
        }
        // Verify: syndromes of the corrected word must vanish.
        for index in 0..<parityCount {
            var value: UInt8 = 0
            for byte in corrected {
                value = multiply(value, field.exp[index]) ^ byte
            }
            if value != 0 { return nil }
        }
        return corrected
    }

    private static func inverse(_ value: UInt8) -> UInt8 {
        guard value != 0 else { return 0 }
        return field.exp[255 - Int(field.log[Int(value)])]
    }

    private static func power(_ base: UInt8, _ exponent: Int) -> UInt8 {
        guard base != 0 else { return exponent == 0 ? 1 : 0 }
        return field.exp[(Int(field.log[Int(base)]) * exponent) % 255]
    }

    private static func xorPolynomials(_ a: [UInt8], _ b: [UInt8]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: max(a.count, b.count))
        for (index, value) in a.enumerated() { out[index] ^= value }
        for (index, value) in b.enumerated() { out[index] ^= value }
        return out
    }
}
