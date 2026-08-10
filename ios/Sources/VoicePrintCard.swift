// A voiceprint image that IS the voice: the glyph card carries the full 1,024-float
// x-vector and its name in a private PNG chunk, so sending someone the picture sends
// them the voice. Import reads the chunk straight back out of the camera-roll bytes.
//
// Robustness is stated honestly rather than oversold: AirDrop, Files, and the photo
// library preserve PNG bytes exactly, so the voice survives; a messaging app that
// recompresses images strips it, and the share copy says to send the file.

import SwiftUI
import UIKit

enum VoicePrintCard {
    /// Private ancillary PNG chunk type (lowercase first letter: ancillary; lowercase
    /// second: private) carrying `MAGIC + u16 name length + name UTF-8 + 1024 f32 LE`.
    private static let chunkType: [UInt8] = Array("ftTS".utf8)
    private static let magic = Array("FTTSVOICE1".utf8)

    // ---- encode -----------------------------------------------------------------

    /// Render the branded card and embed the voice into its PNG bytes.
    @MainActor
    static func pngData(name: String, vector: [Float]) throws -> Data {
        let renderer = ImageRenderer(content: CardView(name: name, vector: vector))
        renderer.scale = 1
        renderer.proposedSize = .init(width: 1024, height: 1024)
        guard let image = renderer.uiImage, let png = image.pngData() else {
            throw EngineError.native("cannot render the voiceprint card")
        }
        return try embed(name: name, vector: vector, into: png)
    }

    static func embed(name: String, vector: [Float], into png: Data) throws -> Data {
        guard png.count > 8 else { throw EngineError.native("not a PNG") }
        var payload = magic
        let nameBytes = Array(name.utf8.prefix(120))
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
        let crc = crc32(over: chunkType + payload)
        chunk += [
            UInt8(crc >> 24 & 0xFF), UInt8(crc >> 16 & 0xFF),
            UInt8(crc >> 8 & 0xFF), UInt8(crc & 0xFF),
        ]
        // Insert immediately before IEND (the last 12 bytes of any well-formed PNG).
        guard png.count >= 12 else { throw EngineError.native("truncated PNG") }
        var out = png
        out.insert(contentsOf: chunk, at: png.count - 12)
        return out
    }

    // ---- decode -----------------------------------------------------------------

    /// Extract a voice from image bytes, if this image carries one.
    static func decode(_ data: Data) -> (name: String, vector: [Float])? {
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
                let stored = Int(bytes[dataStart + length]) << 24
                    | Int(bytes[dataStart + length + 1]) << 16
                    | Int(bytes[dataStart + length + 2]) << 8
                    | Int(bytes[dataStart + length + 3])
                guard crc32(over: type + payload) == UInt32(stored & 0xFFFF_FFFF) else {
                    return nil
                }
                return parse(payload: payload)
            }
            offset = dataStart + length + 4
        }
        return nil
    }

    private static func parse(payload: [UInt8]) -> (name: String, vector: [Float])? {
        guard payload.count > magic.count + 2,
            Array(payload.prefix(magic.count)) == magic
        else { return nil }
        var at = magic.count
        let nameLength = Int(payload[at]) << 8 | Int(payload[at + 1])
        at += 2
        guard payload.count >= at + nameLength + Engine.speakerWidth * 4 else { return nil }
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

    /// PNG CRC-32 (polynomial 0xEDB88320).
    private static func crc32(over bytes: [UInt8]) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in bytes {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1
            }
        }
        return crc ^ 0xFFFF_FFFF
    }

    /// The card artwork: glyph, name, and the promise, on the lab's background.
    struct CardView: View {
        let name: String
        let vector: [Float]

        var body: some View {
            ZStack {
                Lab.background
                RadialGradient(
                    colors: [Lab.emerald.opacity(0.12), .clear],
                    center: .center, startRadius: 60, endRadius: 470)
                VStack(spacing: 26) {
                    Text("F R A N K E N T T S · V O I C E P R I N T")
                        .font(.system(size: 22, weight: .black, design: .monospaced))
                        .foregroundStyle(Lab.emerald)
                    VoicePrintGlyph(vector: vector, lineWidth: 3.4)
                        .frame(width: 620, height: 620)
                    Text(name)
                        .font(.system(size: 52, weight: .black))
                        .foregroundStyle(Lab.textPrimary)
                    Text("this image carries the voice · import it in the FrankenTTS app")
                        .font(.system(size: 20, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
            }
            .frame(width: 1024, height: 1024)
        }
    }
}
