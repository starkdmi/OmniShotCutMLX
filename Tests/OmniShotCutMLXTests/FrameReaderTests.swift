import Foundation
import XCTest

@testable import OmniShotCutMLX

/// `FrameReader` against the decode the official code runs,
/// `ffmpeg -s 128x96 -pix_fmt rgb24 -fps_mode passthrough`, frame by frame and
/// byte by byte, on the videos `scripts/fetch-videos.sh` downloads and on
/// rotated copies of one of them. Skips without ffmpeg.
final class FrameReaderTests: XCTestCase {

    func testFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        var compared = 0
        for name in ["sintel_trailer-480p.mp4", "trailer_480p.mov", "tears_of_steel_720p.mov"] {
            let video = ReferenceTests.videoDirectory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: video.path) else { continue }
            try await compare(video, ffmpeg: ffmpeg)
            compared += 1
        }
        try XCTSkipIf(compared == 0, "no videos; run scripts/fetch-videos.sh")
    }

    /// ffmpeg turns a rotated picture before scaling it, so the horizontal
    /// filter runs along the stored picture's columns.
    func testRotatedFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let source = ReferenceTests.videoDirectory.appendingPathComponent(
            "sintel_trailer-480p.mp4")
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw XCTSkip("no videos; run scripts/fetch-videos.sh")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "omnishotcut-rotation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for degrees in [90, 180, 270] {
            let rotated = directory.appendingPathComponent("rotated-\(degrees).mp4")
            try Self.run(
                ffmpeg,
                [
                    "-v", "error", "-display_rotation", "\(degrees)", "-i", source.path,
                    "-t", "12", "-c", "copy", rotated.path,
                ])
            try await compare(rotated, ffmpeg: ffmpeg)
        }
    }

    private func compare(_ video: URL, ffmpeg: URL) async throws {
        let name = video.lastPathComponent
        let theirs = try Self.run(
            ffmpeg,
            [
                "-v", "error", "-i", video.path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "128x96", "-fps_mode", "passthrough", "-",
            ])
        let frameBytes = 128 * 96 * 3
        let reader = try await FrameReader(url: video, width: 128, height: 96)
        var frames = 0
        var different = 0
        var largest = 0
        var offset = 0
        while let frame = try reader.next() {
            guard offset + frameBytes <= theirs.count else { break }
            let expected = theirs[offset..<(offset + frameBytes)]
            if !frame.pixels.elementsEqual(expected) {
                for (a, b) in zip(frame.pixels, expected) where a != b {
                    different += 1
                    largest = max(largest, abs(Int(a) - Int(b)))
                }
            }
            offset += frameBytes
            frames += 1
        }
        print(
            "FrameReader \(name): \(frames) frames, \(different) values differ, largest \(largest)"
        )
        XCTAssertEqual(frames * frameBytes, theirs.count, "\(name): frame count")
        XCTAssertEqual(different, 0, name)
    }

    @discardableResult
    private static func run(_ executable: URL, _ arguments: [String]) throws -> Data {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw XCTSkip("ffmpeg \(arguments.joined(separator: " ")) failed")
        }
        return data
    }

    private static var ffmpeg: URL? {
        let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
        let directories =
            path.split(separator: ":").map(String.init) + [
                "/opt/homebrew/bin", "/usr/local/bin",
            ]
        return directories.map { URL(fileURLWithPath: $0).appendingPathComponent("ffmpeg") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}
