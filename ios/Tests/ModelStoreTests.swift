import Foundation
import XCTest
@testable import FrankenTTS

final class ModelStoreTests: XCTestCase {
    @MainActor
    func testModelClearIsBlockedWhileVoiceWorkOwnsTheEngine() {
        let model = LabModel()
        XCTAssertTrue(model.canClearModel)

        model.isSynthesizing = true
        XCTAssertFalse(model.canClearModel)
        model.isSynthesizing = false

        model.isComparingVoices = true
        XCTAssertFalse(model.canClearModel)
        model.isComparingVoices = false

        model.isEnrolling = true
        XCTAssertFalse(model.canClearModel)
        model.isEnrolling = false

        model.isClearingModel = true
        XCTAssertFalse(model.canClearModel)
    }

    @MainActor
    func testFinishedAudioKeepsItsProducingVoiceLabelAfterSelectionChanges() {
        let model = LabModel()
        model.selectedVoice = "matt"
        model.lastAudioVoiceLabel = model.currentVoiceLabel

        model.selectedVoice = "james"

        XCTAssertEqual(model.currentVoiceLabel, "James")
        XCTAssertEqual(model.lastAudioExportVoiceLabel, "Matt")
    }

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

    func testFttsqHeaderAcceptsCanonicalDirectory() async throws {
        let contents = try fttsqFixture()
        let file = ModelFile(
            asset: "fixture.fttsq",
            relativePath: "fixture.fttsq",
            bytes: Int64(contents.count),
            sha256: "unused"
        )
        let url = temporaryURL(extension: "fttsq")
        defer { try? FileManager.default.removeItem(at: url) }
        try contents.write(to: url)

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)

        XCTAssertTrue(valid)
    }

    func testFttsqHeaderRejectsMagicFollowedByMalformedDirectoryJSON() async throws {
        var contents = Data("FTTSQ\0\0\0".utf8)
        appendLittleEndian(UInt32(1), to: &contents)
        appendLittleEndian(UInt64(4), to: &contents)
        contents.append(Data("nope".utf8))
        contents.append(Data(repeating: 0, count: 4_096 - contents.count))
        let file = ModelFile(
            asset: "fixture.fttsq",
            relativePath: "fixture.fttsq",
            bytes: Int64(contents.count),
            sha256: "unused"
        )
        let url = temporaryURL(extension: "fttsq")
        defer { try? FileManager.default.removeItem(at: url) }
        try contents.write(to: url)

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)

        XCTAssertFalse(valid)
    }

    func testSafetensorsHeaderRequiresAJsonTensorDirectory() async throws {
        let validURL = temporaryURL(extension: "safetensors")
        let invalidURL = temporaryURL(extension: "safetensors")
        defer {
            try? FileManager.default.removeItem(at: validURL)
            try? FileManager.default.removeItem(at: invalidURL)
        }
        let validData = safetensorsFixture(
            header: "{\"voice.weight\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}"
        )
        let invalidData = safetensorsFixture(header: "{\"voice.weight\":{}}")
        try validData.write(to: validURL)
        try invalidData.write(to: invalidURL)
        let validFile = ModelFile(
            asset: "fixture.safetensors",
            relativePath: "fixture.safetensors",
            bytes: Int64(validData.count),
            sha256: "unused"
        )
        let invalidFile = ModelFile(
            asset: "fixture.safetensors",
            relativePath: "fixture.safetensors",
            bytes: Int64(invalidData.count),
            sha256: "unused"
        )

        let valid = try await ModelStore.hasValidContainerHeader(file: validFile, at: validURL)
        let invalid = try await ModelStore.hasValidContainerHeader(file: invalidFile, at: invalidURL)

        XCTAssertTrue(valid)
        XCTAssertFalse(invalid)
    }

    func testSafetensorsHeaderRejectsGapsOverlapsAndTrailingPayload() async throws {
        let headers = [
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        ]
        for header in headers {
            let data = safetensorsFixture(header: header)
            let url = temporaryURL(extension: "safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try data.write(to: url)
            let file = ModelFile(
                asset: "fixture.safetensors",
                relativePath: "fixture.safetensors",
                bytes: Int64(data.count),
                sha256: "unused"
            )

            let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)
            XCTAssertFalse(valid)
        }
    }

    func testSafetensorsHeaderAcceptsAZeroSizedTensorAtTheNextTensorStart() async throws {
        let data = safetensorsFixture(
            header: "{\"empty\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]},\"weight\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}"
        )
        let url = temporaryURL(extension: "safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let file = ModelFile(
            asset: "fixture.safetensors",
            relativePath: "fixture.safetensors",
            bytes: Int64(data.count),
            sha256: "unused"
        )

        let valid = try await ModelStore.hasValidContainerHeader(file: file, at: url)
        XCTAssertTrue(valid)
    }

    private func fttsqFixture() throws -> Data {
        let payloadOffset = 2_048
        let fileLength = 4_096
        let directory: [String: Any] = [
            "format_version": 1,
            "model_family": "fixture",
            "source_sha256": String(repeating: "a", count: 64),
            "license_notice": "fixture notice",
            "sections": [[
                "name": "weights",
                "access_class": "METADATA",
                "offset": payloadOffset,
                "length": 4,
                "sha256": String(repeating: "0", count: 64),
            ]],
            "tensors": [[
                "name": "voice.weight",
                "section": "weights",
                "dtype": "f32",
                "shape": [1],
                "offset": 0,
                "length": 4,
            ]],
        ]
        let directoryData = try JSONSerialization.data(
            withJSONObject: directory,
            options: [.sortedKeys]
        )
        XCTAssertLessThan(20 + directoryData.count, payloadOffset)
        var data = Data("FTTSQ\0\0\0".utf8)
        appendLittleEndian(UInt32(1), to: &data)
        appendLittleEndian(UInt64(directoryData.count), to: &data)
        data.append(directoryData)
        data.append(Data(repeating: 0, count: payloadOffset - data.count))
        data.append(Data(repeating: 0, count: 4))
        data.append(Data(repeating: 0, count: fileLength - data.count))
        return data
    }

    private func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
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
