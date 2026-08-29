import Foundation
import XCTest
@testable import FrankenTTS

final class ModelStoreTests: XCTestCase {
    func testFttsqHeaderRejectsAnExactSizeZeroFilledFile() async throws {
        let file = ModelFile(
            asset: "fixture.fttsq",
            relativePath: "fixture.fttsq",
            bytes: 32,
            sha256: "unused"
        )
        let url = temporaryURL(extension: "fttsq")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0, count: Int(file.bytes)).write(to: url)

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)

        XCTAssertFalse(valid)
    }

    func testFttsqHeaderAcceptsCanonicalMagic() async throws {
        let file = ModelFile(
            asset: "fixture.fttsq",
            relativePath: "fixture.fttsq",
            bytes: 32,
            sha256: "unused"
        )
        let url = temporaryURL(extension: "fttsq")
        defer { try? FileManager.default.removeItem(at: url) }
        try (Data("FTTSQ\0\0\0".utf8) + Data(repeating: 0, count: 24)).write(to: url)

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)

        XCTAssertTrue(valid)
    }

    func testSafetensorsHeaderRequiresAJsonTensorDirectory() async throws {
        let validURL = temporaryURL(extension: "safetensors")
        let invalidURL = temporaryURL(extension: "safetensors")
        defer {
            try? FileManager.default.removeItem(at: validURL)
            try? FileManager.default.removeItem(at: invalidURL)
        }
        let validData = safetensorsFixture(header: "{\"voice.weight\":{}}")
        let invalidData = Data(repeating: 0, count: validData.count)
        try validData.write(to: validURL)
        try invalidData.write(to: invalidURL)
        let file = ModelFile(
            asset: "fixture.safetensors",
            relativePath: "fixture.safetensors",
            bytes: Int64(validData.count),
            sha256: "unused"
        )

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: validURL)
        let invalid = try await ModelStore.hasValidContainerHeader(file: file, at: invalidURL)

        XCTAssertTrue(valid)
        XCTAssertFalse(invalid)
    }

    private func safetensorsFixture(header: String) -> Data {
        let headerData = Data(header.utf8)
        var length = UInt64(headerData.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(headerData)
        data.append(Data(repeating: 0, count: 16))
        return data
    }

    private func temporaryURL(extension pathExtension: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("frankentts-model-test-\(UUID().uuidString)")
            .appendingPathExtension(pathExtension)
    }
}
