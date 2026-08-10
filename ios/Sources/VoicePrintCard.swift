// The voice card: a picture that carries the voice itself.
//
// Two layers, one honest promise:
// - The visible emerald mosaic IS the data — the full 1,024-float voiceprint written
//   as brightness levels, with error correction (VoiceCode). It survives messaging
//   apps that recompress images and screenshots from any size of phone.
// - A private PNG chunk carries the same bytes losslessly as a fast path for AirDrop
//   and Files transfers that keep the file byte-exact.
//
// Import tries the chunk first, then reads the mosaic out of the pixels.

import SwiftUI
import UIKit

enum VoicePrintCard {
    static let cardWidth = VoiceCode.cardSize // 1024
    static let cardHeight = VoiceCode.cardSize + 156

    /// Private ancillary PNG chunk (`ftTS`) with `MAGIC + u16 len + name + 1024 f32 LE`.
    private static let chunkType: [UInt8] = Array("ftTS".utf8)
    private static let magic = Array("FTTSVOICE1".utf8)

    // ---- encode -----------------------------------------------------------------

    /// Render the card and embed the lossless copy into its PNG bytes.
    @MainActor
    static func pngData(name: String, vector: [Float]) throws -> Data {
        let mosaic = mosaicImage(name: name, vector: vector)
        let renderer = ImageRenderer(content: CardView(name: name, mosaic: mosaic))
        renderer.scale = 1
        renderer.proposedSize = .init(
            width: CGFloat(cardWidth), height: CGFloat(cardHeight))
        guard let image = renderer.uiImage, let png = image.pngData() else {
            throw EngineError.native("cannot render the voice card")
        }
        return try embed(name: name, vector: vector, into: png)
    }

    private static func mosaicImage(name: String, vector: [Float]) -> UIImage {
        let pixels = VoiceCode.renderMosaicPixels(name: name, vector: vector)
        let size = VoiceCode.cardSize
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData),
            let cgImage = CGImage(
                width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: size * 3, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent)
        else {
            return UIImage()
        }
        return UIImage(cgImage: cgImage)
    }

    static func embed(name: String, vector: [Float], into png: Data) throws -> Data {
        guard png.count >= 12 else { throw EngineError.native("truncated PNG") }
        var payload = magic
        let nameBytes = Array(name.utf8.prefix(64))
        payload += [UInt8(nameBytes.count >> 8), UInt8(nameBytes.count & 0xFF)]
        payload += nameBytes
        for value in vector {
            payload += withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
        }
        var chunk = [UInt8]()
        chunk += [
            UInt8(payload.count >> 24 & 0xFF), UInt8(payload.count >> 16 & 0xFF),
            UInt8(payload.count >> 8 & 0xFF), UInt8(payload.count & 0xFF),
        ]
        chunk += chunkType
        chunk += payload
        let crc = VoiceCode.crc32(chunkType + payload)
        chunk += [
            UInt8(crc >> 24 & 0xFF), UInt8(crc >> 16 & 0xFF),
            UInt8(crc >> 8 & 0xFF), UInt8(crc & 0xFF),
        ]
        // Insert immediately before IEND (the last 12 bytes of a well-formed PNG).
        var out = png
        out.insert(contentsOf: chunk, at: png.count - 12)
        return out
    }

    // ---- decode -----------------------------------------------------------------

    /// Extract a voice from image bytes: lossless chunk first, then the mosaic.
    static func decode(_ data: Data) -> (name: String, vector: [Float])? {
        if let fromChunk = decodeChunk(data) {
            return fromChunk
        }
        return decodePixels(data)
    }

    private static func decodeChunk(_ data: Data) -> (name: String, vector: [Float])? {
        let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        guard data.count > 16, Array(data.prefix(8)) == signature else { return nil }
        var offset = 8
        let bytes = [UInt8](data)
        while offset + 12 <= bytes.count {
            let length = Int(bytes[offset]) << 24 | Int(bytes[offset + 1]) << 16
                | Int(bytes[offset + 2]) << 8 | Int(bytes[offset + 3])
            let type = Array(bytes[offset + 4..<offset + 8])
            let dataStart = offset + 8
            guard length >= 0, dataStart + length + 4 <= bytes.count else { return nil }
            if type == chunkType {
                let payload = Array(bytes[dataStart..<dataStart + length])
                let stored = UInt32(bytes[dataStart + length]) << 24
                    | UInt32(bytes[dataStart + length + 1]) << 16
                    | UInt32(bytes[dataStart + length + 2]) << 8
                    | UInt32(bytes[dataStart + length + 3])
                guard VoiceCode.crc32(type + payload) == stored else { return nil }
                return parseChunk(payload: payload)
            }
            offset = dataStart + length + 4
        }
        return nil
    }

    private static func parseChunk(payload: [UInt8]) -> (name: String, vector: [Float])? {
        guard payload.count > magic.count + 2,
            Array(payload.prefix(magic.count)) == magic
        else { return nil }
        var at = magic.count
        let nameLength = Int(payload[at]) << 8 | Int(payload[at + 1])
        at += 2
        guard nameLength <= 64,
            payload.count >= at + nameLength + Engine.speakerWidth * 4
        else { return nil }
        let name = String(decoding: payload[at..<at + nameLength], as: UTF8.self)
        at += nameLength
        var vector = [Float](repeating: 0, count: Engine.speakerWidth)
        for index in 0..<Engine.speakerWidth {
            let base = at + index * 4
            let bits = UInt32(payload[base]) | UInt32(payload[base + 1]) << 8
                | UInt32(payload[base + 2]) << 16 | UInt32(payload[base + 3]) << 24
            vector[index] = Float(bitPattern: bits)
        }
        guard vector.allSatisfy(\.isFinite) else { return nil }
        return (name.isEmpty ? "shared voice" : name, vector)
    }

    /// Decode from the rendered pixels: works for recompressed JPEGs and screenshots.
    private static func decodePixels(_ data: Data) -> (name: String, vector: [Float])? {
        guard var image = UIImage(data: data) else { return nil }
        // A photo-library round trip can attach an EXIF orientation; the pixel grid
        // must be upright before sampling, so bake the rotation in first.
        if image.imageOrientation != .up {
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            image = UIGraphicsImageRenderer(size: image.size, format: format).image {
                _ in
                image.draw(in: CGRect(origin: .zero, size: image.size))
            }
        }
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0, width * height <= 40_000_000 else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.interpolationQuality = .none
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            rgb[index * 3] = rgba[index * 4]
            rgb[index * 3 + 1] = rgba[index * 4 + 1]
            rgb[index * 3 + 2] = rgba[index * 4 + 2]
        }
        guard let (name, vector) = VoiceCode.decode(
            pixels: rgb, width: width, height: height)
        else { return nil }
        return (name, vector)
    }

    // ---- card artwork -----------------------------------------------------------

    /// The card: title band, the mosaic (which is the data), name band.
    struct CardView: View {
        let name: String
        let mosaic: UIImage

        var body: some View {
            VStack(spacing: 0) {
                Text("F R A N K E N T T S · V O I C E  C A R D")
                    .font(.system(size: 24, weight: .black, design: .monospaced))
                    .foregroundStyle(Lab.emerald)
                    .frame(height: 72)
                Image(uiImage: mosaic)
                    .interpolation(.none)
                    .resizable()
                    .frame(
                        width: CGFloat(VoiceCode.cardSize),
                        height: CGFloat(VoiceCode.cardSize))
                VStack(spacing: 8) {
                    Text(name)
                        .font(.system(size: 40, weight: .black))
                        .foregroundStyle(Lab.textPrimary)
                        .lineLimit(1)
                    Text("the green mosaic is the voice · add it from a photo in FrankenTTS")
                        .font(.system(size: 19, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
                .frame(height: 84)
            }
            .frame(
                width: CGFloat(VoicePrintCard.cardWidth),
                height: CGFloat(VoicePrintCard.cardHeight))
            .background(Lab.background)
        }
    }
}
