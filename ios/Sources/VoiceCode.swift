// The dense visual voice code: 4 KB of x-vector as an emerald mosaic.
//
// Why this shape:
// - 144×144 cells, 2 bits per cell as one of FOUR brightness levels on a constant-hue
//   emerald ramp. JPEG stores luminance at full resolution and chroma at half, so the
//   information rides in the channel compression treats best, and the constant hue is
//   also what makes the code look like the lab instead of like a barcode.
// - Interleaved Reed-Solomon (255, 223) over GF(2^8): JPEG artifacts are local, the
//   interleave spreads any damaged patch across every block, and each block corrects
//   up to 16 wrong bytes. A CRC over the plaintext refuses any silent miscorrection.
// - Row zero is a known calibration pattern: the decoder learns the four levels from
//   it (so global brightness shifts do not matter) and uses its fit to refine grid
//   alignment by a small search. Shared images are axis-aligned — the channel is the
//   file itself, not a camera — so scale and offset are the only geometry.
//
// The core is pure byte arrays in and out, so the exact codec is testable off-device.

import Foundation

enum VoiceCode {
    static let gridN = 144
    static let cardSize = 1024
    /// The mosaic square inside the card: integer 6-px cells, 80-px quiet margins.
    static let regionOrigin = 80
    static let cellPx = 6
    static var regionSize: Int { gridN * cellPx } // 864

    /// Emerald ramp, darkest to brightest. Constant hue family, widely separated luma.
    static let levels: [(r: UInt8, g: UInt8, b: UInt8)] = [
        (6, 22, 15), (26, 92, 61), (56, 168, 110), (112, 248, 170),
    ]

    private static let magic = Array("FV02".utf8)
    private static let dataBytesPerBlock = 223
    private static let parityBytesPerBlock = 32
    private static let blocks = 19
    static var plaintextCapacity: Int { dataBytesPerBlock * blocks } // 4237

    // ------------------------------------------------------------------ public API

    /// Encode name + vector into an RGB24 card image (cardSize × cardSize).
    static func renderCardPixels(name: String, vector: [Float]) -> [UInt8] {
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
        var coded = [UInt8]()
        var blocksOut = [[UInt8]]()
        for block in 0..<blocks {
            let start = block * dataBytesPerBlock
            let data = Array(plaintext[start..<start + dataBytesPerBlock])
            blocksOut.append(data + ReedSolomon.parity(for: data))
        }
        for position in 0..<(dataBytesPerBlock + parityBytesPerBlock) {
            for block in 0..<blocks {
                coded.append(blocksOut[block][position])
            }
        }

        // Paint the card.
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
                if row == 0 {
                    level = column % 4 // calibration + alignment row
                } else if bitCursor + 2 <= totalBits {
                    let byte = Int(coded[bitCursor >> 3])
                    let shift = 6 - (bitCursor & 7)
                    level = (byte >> shift) & 0b11
                    bitCursor += 2
                } else {
                    level = (row &+ column) % 4 // deterministic filler
                }
                paintCell(&pixels, row: row, column: column, level: level)
            }
        }
        return pixels
    }

    /// Decode a voice from RGB24 pixels of any uniformly scaled copy of the card.
    static func decode(pixels: [UInt8], width: Int, height: Int) -> (String, [Float])? {
        guard width > gridN, height > gridN else { return nil }
        let scaleX = Double(width) / Double(cardSize)
        let scaleY = Double(height) / Double(cardSize)

        func luma(atX x: Double, y: Double) -> Double {
            let xi = min(max(Int(x.rounded()), 0), width - 1)
            let yi = min(max(Int(y.rounded()), 0), height - 1)
            var sum = 0.0
            var count = 0.0
            for dy in -1...1 {
                for dx in -1...1 {
                    let sx = min(max(xi + dx, 0), width - 1)
                    let sy = min(max(yi + dy, 0), height - 1)
                    let at = (sy * width + sx) * 3
                    sum += 0.299 * Double(pixels[at]) + 0.587 * Double(pixels[at + 1])
                        + 0.114 * Double(pixels[at + 2])
                    count += 1
                }
            }
            return sum / count
        }

        func cellCenter(row: Int, column: Int, dx: Double, dy: Double) -> (Double, Double) {
            let x = (Double(regionOrigin) + (Double(column) + 0.5) * Double(cellPx) + dx) * scaleX
            let y = (Double(regionOrigin) + (Double(row) + 0.5) * Double(cellPx) + dy) * scaleY
            return (x, y)
        }

        // Fit the calibration row under a small offset search; keep the best fit.
        var best: (score: Double, dx: Double, dy: Double, means: [Double])?
        for dyStep in -4...4 {
            for dxStep in -4...4 {
                let dx = Double(dxStep) * 0.75
                let dy = Double(dyStep) * 0.75
                var sums = [Double](repeating: 0, count: 4)
                var counts = [Double](repeating: 0, count: 4)
                for column in 0..<gridN {
                    let (x, y) = cellCenter(row: 0, column: column, dx: dx, dy: dy)
                    sums[column % 4] += luma(atX: x, y: y)
                    counts[column % 4] += 1
                }
                let means = zip(sums, counts).map { $0 / max($1, 1) }
                // Score: monotone separation of the four recovered levels.
                let gaps = [means[1] - means[0], means[2] - means[1], means[3] - means[2]]
                let score = gaps.min() ?? -1
                if score > (best?.score ?? -.infinity) {
                    best = (score, dx, dy, means)
                }
            }
        }
        guard let fit = best, fit.score > 4 else { return nil }
        let thresholds = [
            (fit.means[0] + fit.means[1]) / 2,
            (fit.means[1] + fit.means[2]) / 2,
            (fit.means[2] + fit.means[3]) / 2,
        ]

        // Sample every data cell.
        let codedCount = (dataBytesPerBlock + parityBytesPerBlock) * blocks
        var coded = [UInt8](repeating: 0, count: codedCount)
        var bitCursor = 0
        let totalBits = codedCount * 8
        outer: for row in 1..<gridN {
            for column in 0..<gridN {
                if bitCursor + 2 > totalBits { break outer }
                let (x, y) = cellCenter(row: row, column: column, dx: fit.dx, dy: fit.dy)
                let value = luma(atX: x, y: y)
                var level = 0
                for threshold in thresholds where value > threshold {
                    level += 1
                }
                let shift = 6 - (bitCursor & 7)
                coded[bitCursor >> 3] |= UInt8(level << shift)
                bitCursor += 2
            }
        }

        // De-interleave and correct each block.
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

    // ------------------------------------------------------------------ internals

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
        // Syndromes.
        var syndromes = [UInt8](repeating: 0, count: parityCount)
        var clean = true
        for index in 0..<parityCount {
            var value: UInt8 = 0
            for &byte in received {
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
            for index in 1..<sigma.count {
                if step >= index {
                    discrepancy ^= multiply(sigma[index], syndromes[step - index])
                }
            }
            if discrepancy == 0 {
                m += 1
            } else if 2 * (sigma.count - 1) <= step {
                let old = sigma
                let scale = multiply(discrepancy, inverse(discrepancyLast))
                var shifted = [UInt8](repeating: 0, count: m) + previous
                for index in 0..<shifted.count {
                    shifted[index] = multiply(shifted[index], scale)
                }
                sigma = xorPolynomials(sigma, shifted)
                previous = old
                discrepancyLast = discrepancy
                m = 1
            } else {
                let scale = multiply(discrepancy, inverse(discrepancyLast))
                var shifted = [UInt8](repeating: 0, count: m) + previous
                for index in 0..<shifted.count {
                    shifted[index] = multiply(shifted[index], scale)
                }
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

        // Forney for magnitudes: omega = (syndromes * sigma) mod x^parity.
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
        // Verify: recompute syndromes on the corrected word.
        for index in 0..<parityCount {
            var value: UInt8 = 0
            for &byte in corrected {
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
