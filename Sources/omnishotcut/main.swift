import Foundation
import OmniShotCutMLX

// omnishotcut [--model <directory or Hub id>] [--json] <video>...

let usage = """
    usage: omnishotcut [--model <directory or Hub id>] [--json] <video>...

    Prints the shots and transitions of each video. The model is a directory
    written by scripts/convert.py or a Hub id (default \(OmniShotCut.defaultModel)).
    """

var model = OmniShotCut.defaultModel
var json = false
var videos: [URL] = []
var arguments = CommandLine.arguments.dropFirst()
while let argument = arguments.popFirst() {
    switch argument {
    case "--model":
        guard let value = arguments.popFirst() else {
            FileHandle.standardError.write(Data((usage + "\n").utf8))
            exit(2)
        }
        model = value
    case "--json": json = true
    case "-h", "--help":
        print(usage)
        exit(0)
    default: videos.append(URL(fileURLWithPath: (argument as NSString).expandingTildeInPath))
    }
}
guard !videos.isEmpty else {
    FileHandle.standardError.write(Data((usage + "\n").utf8))
    exit(2)
}

struct Output: Encodable {
    struct Segment: Encodable {
        var start: Double
        var end: Double
        var frames: [Int]
        var kind: String
        var boundary: String
    }
    var video: String
    var seconds: Double
    var segments: [Segment]
}

do {
    let local = URL(fileURLWithPath: (model as NSString).expandingTildeInPath)
    let detector =
        FileManager.default.fileExists(atPath: local.path)
        ? try OmniShotCut(directory: local)
        : try await OmniShotCut.pretrained(model)
    var outputs: [Output] = []
    for video in videos {
        let started = Date()
        let segments = try await detector.segments(in: video)
        let elapsed = Date().timeIntervalSince(started)
        if json {
            outputs.append(
                Output(
                    video: video.path, seconds: elapsed,
                    segments: segments.map {
                        Output.Segment(
                            start: $0.start, end: $0.end,
                            frames: [$0.frames.lowerBound, $0.frames.upperBound],
                            kind: $0.kind.name, boundary: $0.boundary.name)
                    }))
            continue
        }
        print(
            "\(video.lastPathComponent): \(segments.count) segments in "
                + String(format: "%.1f s", elapsed))
        for segment in segments {
            print(
                String(
                    format: "%9.3f %9.3f  %6d %6d  %@ %@", segment.start, segment.end,
                    segment.frames.lowerBound, segment.frames.upperBound,
                    segment.kind.name.padding(toLength: 9, withPad: " ", startingAt: 0),
                    segment.boundary.name))
        }
    }
    if json {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(outputs), as: UTF8.self))
    }
} catch {
    FileHandle.standardError.write(Data("omnishotcut: \(error)\n".utf8))
    exit(1)
}
