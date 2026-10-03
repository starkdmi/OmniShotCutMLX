import CryptoKit
import Foundation
import XCTest

@testable import OmniShotCutMLX

/// The whole detector against the official PyTorch implementation.
///
/// `omnishotcut-reference.json` is the official code's output for three
/// Blender open movies — `uva-cv-lab/OmniShotCut_v1.5`, fp32 on the CPU,
/// overlap 10 — written by `scripts/reference.py`. The port has to find the
/// same cuts with the same labels from its own decoder and converted weights.
///
/// `scripts/fetch-videos.sh` downloads the videos to `videos/`;
/// `OMNISHOTCUT_VIDEOS` names another directory. `OMNISHOTCUT_MODEL` is a
/// converted directory or Hub id (default ``OmniShotCut/defaultModel``).
/// Must run under `xcrun xctest`, see README.
final class ReferenceTests: XCTestCase {

    private struct Reference: Decodable {
        struct Video: Decodable {
            var sha256: String
            var fps: Double
            var frames: Int
            var ranges: [[Int]]
            var intra: [Int]
            var inter: [Int]
        }
        var videos: [String: Video]
    }

    func testSegmentsMatchTheOfficialImplementation() async throws {
        let environment = ProcessInfo.processInfo.environment
        let videos = Self.videoDirectory
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "omnishotcut-reference", withExtension: "json",
                subdirectory: "Fixtures"))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))

        let model = environment["OMNISHOTCUT_MODEL"] ?? OmniShotCut.defaultModel
        let local = URL(fileURLWithPath: (model as NSString).expandingTildeInPath)
        let detector =
            FileManager.default.fileExists(atPath: local.path)
            ? try OmniShotCut(directory: local) : try await OmniShotCut.pretrained(model)

        var compared = 0
        for name in reference.videos.keys.sorted() {
            let expected = reference.videos[name]!
            let video = videos.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: video.path) else {
                print("OmniShotCut: \(name) missing, skipped")
                continue
            }
            guard try Self.sha256(of: video) == expected.sha256 else {
                XCTFail("\(name) is not the file the reference was made from")
                continue
            }
            let started = Date()
            let segments = try await detector.segments(in: video)
            let elapsed = Date().timeIntervalSince(started)

            XCTAssertEqual(segments.last?.frames.upperBound, expected.frames, name)
            let theirs = Array(
                zip(expected.ranges, zip(expected.intra, expected.inter)).dropFirst())
            var matched = 0
            var sameFrame = 0
            var sameLabels = 0
            var identical = 0
            for (range, labels) in theirs {
                guard
                    let nearest = segments.dropFirst().min(by: {
                        abs($0.frames.lowerBound - range[0])
                            < abs($1.frames.lowerBound - range[0])
                    }),
                    abs(nearest.frames.lowerBound - range[0]) <= 1
                else { continue }
                matched += 1
                let onFrame = nearest.frames.lowerBound == range[0]
                let labelled =
                    nearest.kind.rawValue == labels.0 && nearest.boundary.rawValue == labels.1
                if onFrame { sameFrame += 1 }
                if labelled { sameLabels += 1 }
                if onFrame, labelled { identical += 1 }
            }
            print(
                "OmniShotCut \(name): \(segments.count) segments vs \(expected.ranges.count); "
                    + "of \(theirs.count) cuts \(matched) within a frame, \(sameFrame) on it, "
                    + "\(sameLabels) with the same labels, \(identical) both; "
                    + String(format: "%.1f s", elapsed))
            // The fp32 conversion reproduces every cut. fp16 weights flip a
            // near-tie that PyTorch flips too with the same rounding — one
            // label in the Sintel trailer, 0.0016 apart in fp64 — so one cut
            // a video may differ.
            XCTAssertEqual(segments.count, expected.ranges.count, name)
            XCTAssertGreaterThanOrEqual(identical, theirs.count - 1, name)
            compared += 1
        }
        try XCTSkipIf(
            compared == 0, "no reference videos in \(videos.path); run scripts/fetch-videos.sh")
    }

    /// `OMNISHOTCUT_VIDEOS`, or `videos/` in the package.
    static var videoDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["OMNISHOTCUT_VIDEOS"] {
            return URL(
                fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("videos", isDirectory: true)
    }

    private static func sha256(of url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let block = try file.read(upToCount: 1 << 20), !block.isEmpty {
            hash.update(data: block)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
