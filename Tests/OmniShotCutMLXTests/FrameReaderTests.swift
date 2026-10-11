import AVFoundation
import Foundation
import XCTest

@testable import OmniShotCutMLX

/// `FrameReader` against the decode the official code runs,
/// `ffmpeg -s 128x96 -pix_fmt rgb24 -fps_mode passthrough`, frame by frame and
/// byte by byte, on the videos `scripts/fetch-videos.sh` downloads and on
/// turned, mirrored and re-tagged copies of one of them. Skips without ffmpeg.
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

    /// ffmpeg turns and mirrors the picture before scaling it, so the
    /// horizontal filter runs along the stored picture's columns after a
    /// quarter turn. With the original these are all eight orientations.
    func testTurnedAndMirroredFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let (source, directory) = try sintelAndScratchDirectory()
        let orientations = [
            ("rotated-90", ["-display_rotation", "90"]),
            ("rotated-180", ["-display_rotation", "180"]),
            ("rotated-270", ["-display_rotation", "270"]),
            ("mirrored", ["-display_hflip"]),
            ("flipped", ["-display_vflip"]),
            ("rotated-90-mirrored", ["-display_rotation", "90", "-display_hflip"]),
            ("rotated-90-flipped", ["-display_rotation", "90", "-display_vflip"]),
        ]
        for (name, options) in orientations {
            let video = directory.appendingPathComponent("\(name).mp4")
            try Self.run(
                ffmpeg,
                ["-v", "error"] + options + [
                    "-i", source.path, "-t", "12", "-c", "copy", video.path,
                ])
            try await compare(video, ffmpeg: ffmpeg)
        }
    }

    /// The transform's signs as ffmpeg's autorotate reads them: a mirror is a
    /// flip, not a half turn.
    func testTransformsAreReadAsFFmpegReadsThem() {
        let cases: [(a: CGFloat, b: CGFloat, c: CGFloat, d: CGFloat, filters: String)] = [
            (1, 0, 0, 1, ""),
            (-1, 0, 0, 1, "hflip"),
            (1, 0, 0, -1, "vflip"),
            (-1, 0, 0, -1, "vflip,hflip"),
            (0, 1, -1, 0, "transpose,hflip"),  // transpose=clock
            (0, -1, 1, 0, "transpose,vflip"),  // transpose=cclock
            (0, 1, 1, 0, "transpose"),  // transpose=cclock_flip
            (0, -1, -1, 0, "transpose,vflip,hflip"),  // transpose=clock_flip
        ]
        for (a, b, c, d, filters) in cases {
            let orientation = Orientation(
                CGAffineTransform(a: a, b: b, c: c, d: d, tx: 0, ty: 0))
            let applied = [
                orientation.transposed ? "transpose" : nil,
                orientation.reversesRows ? "vflip" : nil,
                orientation.reversesColumns ? "hflip" : nil,
            ]
            XCTAssertEqual(
                applied.compactMap { $0 }.joined(separator: ","), filters, "\([a, b, c, d])")
        }
    }

    /// A track joined from clips coded differently has a sample description
    /// per clip. ffmpeg converts each clip with its own range and matrix, here
    /// limited BT.709 and then full BT.601.
    func testJoinedClipsKeepTheirOwnColour() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        guard #available(macOS 15, iOS 18, *) else {
            throw XCTSkip("joining needs AVAssetExportSession.export(to:as:)")
        }
        let (source, directory) = try sintelAndScratchDirectory()
        var clips: [URL] = []
        for (name, tags) in [
            ("limited-709", "video_full_range_flag=0:matrix_coefficients=1"),
            ("full-601", "video_full_range_flag=1:matrix_coefficients=6"),
        ] {
            let clip = directory.appendingPathComponent("\(name).mp4")
            try Self.run(
                ffmpeg,
                [
                    "-v", "error", "-t", "6", "-i", source.path, "-an", "-c", "copy",
                    "-bsf:v", "h264_metadata=\(tags)", clip.path,
                ])
            clips.append(clip)
        }
        let joined = directory.appendingPathComponent("joined.mov")
        try await Self.join(clips, into: joined)
        let tracks = try await AVURLAsset(url: joined).loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let descriptions = try await track.load(.formatDescriptions)
        XCTAssertEqual(descriptions.count, 2, "one sample description per clip")
        try await compare(joined, ffmpeg: ffmpeg)
    }

    /// The last segment ends with the video, not with audio that runs on.
    func testEndIsWhereTheVideoEnds() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let (source, directory) = try sintelAndScratchDirectory()
        let video = directory.appendingPathComponent("long-audio.mp4")
        try Self.run(
            ffmpeg,
            [
                "-v", "error", "-t", "6", "-i", source.path, "-f", "lavfi", "-t", "10", "-i",
                "sine", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", video.path,
            ])
        let reader = try await FrameReader(url: video, width: 128, height: 96)
        var last: TimeInterval = 0
        while let frame = try reader.next() { last = frame.time }
        XCTAssertEqual(reader.end, last + 1.0 / 24, accuracy: 1e-6, "one frame past the last")
        let duration = try await AVURLAsset(url: video).load(.duration).seconds
        XCTAssertGreaterThan(duration, reader.end + 3, "the audio runs on")
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

    /// The Sintel trailer, and a directory removed after the test.
    private func sintelAndScratchDirectory() throws -> (source: URL, directory: URL) {
        let source = ReferenceTests.videoDirectory.appendingPathComponent(
            "sintel_trailer-480p.mp4")
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw XCTSkip("no videos; run scripts/fetch-videos.sh")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "omnishotcut-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return (source, directory)
    }

    /// The clips' video one after another without re-encoding, each keeping
    /// its sample description.
    @available(macOS 15, iOS 18, *)
    private static func join(_ clips: [URL], into output: URL) async throws {
        let composition = AVMutableComposition()
        let track = try XCTUnwrap(
            composition.addMutableTrack(
                withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
        var end = CMTime.zero
        for clip in clips {
            let asset = AVURLAsset(url: clip)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            let source = try XCTUnwrap(tracks.first)
            let range = try await source.load(.timeRange)
            // A track holds its asset weakly, and inserting needs the asset.
            try withExtendedLifetime(asset) {
                try track.insertTimeRange(range, of: source, at: end)
            }
            end = end + range.duration
        }
        let session = try XCTUnwrap(
            AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough)
        )
        try await session.export(to: output, as: .mov)
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
