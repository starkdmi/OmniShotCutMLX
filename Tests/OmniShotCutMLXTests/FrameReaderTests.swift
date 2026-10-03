import Foundation
import XCTest

@testable import OmniShotCutMLX

/// `FrameReader` against the decode the official code runs,
/// `ffmpeg -s 128x96 -pix_fmt rgb24 -fps_mode passthrough`, on the two
/// trailers `scripts/fetch-videos.sh` downloads. Skips without ffmpeg.
final class FrameReaderTests: XCTestCase {

    func testFramesMatchFFmpeg() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        var compared = 0
        for name in ["sintel_trailer-480p.mp4", "trailer_480p.mov"] {
            let video = ReferenceTests.videoDirectory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: video.path) else { continue }

            let process = Process()
            process.executableURL = ffmpeg
            process.arguments = [
                "-v", "error", "-i", video.path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "128x96", "-fps_mode", "passthrough", "-",
            ]
            let pipe = Pipe()
            process.standardOutput = pipe
            try process.run()
            let theirs = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0, name)

            let reader = try await FrameReader(url: video, width: 128, height: 96)
            var ours: [UInt8] = []
            while let frame = try reader.next() { ours += frame.pixels }

            XCTAssertEqual(ours.count, theirs.count, "\(name): frame count")
            var identical = 0
            var largest = 0
            for (a, b) in zip(ours, theirs) {
                let difference = abs(Int(a) - Int(b))
                if difference == 0 { identical += 1 }
                largest = max(largest, difference)
            }
            let share = Double(identical) / Double(max(1, min(ours.count, theirs.count)))
            print(
                "FrameReader \(name): \(ours.count / (128 * 96 * 3)) frames, "
                    + String(format: "%.2f%%", share * 100)
                    + " of values identical to ffmpeg, largest difference \(largest)")
            XCTAssertGreaterThan(share, 0.99, name)
            compared += 1
        }
        try XCTSkipIf(compared == 0, "no trailers; run scripts/fetch-videos.sh")
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
