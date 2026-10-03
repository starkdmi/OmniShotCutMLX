import Foundation

/// One stretch of a video between two boundaries: a shot, or the transition
/// between two shots.
///
/// ``kind`` says what the stretch itself is — a shot, or a dissolve, wipe or
/// fade on its way to the next one — and ``boundary`` says how it began. A
/// dissolve's middle frame is two pictures at once and a fade's is half
/// black, which is why a frame picker should take frames from shots only.
public struct ShotSegment: Sendable, Hashable, Codable {

    /// What the segment is. Raw values are OmniShotCut's intra-relation ids.
    public enum Kind: Int, Sendable, Hashable, Codable, CaseIterable {
        case shot = 0
        case dissolve = 1
        case wipe = 2
        case push = 3
        case slide = 4
        case zoom = 5
        /// Fade to or from black or white, including a dip between shots.
        case fade = 6
        case doorway = 7
        /// The model's padding class; it marks frames past a window's end and
        /// should not survive stitching.
        case padding = 8

        /// Anything but a shot is on its way between pictures.
        public var isTransition: Bool { self != .shot }

        /// The official label.
        public var name: String {
            [
                "General", "Dissolve", "Wipes", "Push", "Slide", "Zoom", "Fade", "Doorway",
                "Padding",
            ][
                rawValue]
        }
    }

    /// How the segment starts. Raw values are OmniShotCut's inter-relation ids.
    public enum Boundary: Int, Sendable, Hashable, Codable, CaseIterable {
        case videoStart = 0
        case hardCut = 1
        case transitionSource = 2
        case transition = 3
        /// A jump within one camera setup — a cut in an interview.
        case suddenJump = 4
        case padding = 5

        /// The official label.
        public var name: String {
            [
                "New_Start", "Hard_Cut", "Transition_Source", "Transition", "Sudden_Jump",
                "Padding",
            ][
                rawValue]
        }
    }

    /// Presentation time of the first frame, in seconds.
    public var start: TimeInterval
    /// Presentation time of the next segment's first frame, or the video's
    /// duration for the last segment.
    public var end: TimeInterval
    /// Decoded frame indices, `start..<end`, as the official code numbers
    /// them.
    public var frames: Range<Int>
    public var kind: Kind
    public var boundary: Boundary

    public init(
        start: TimeInterval, end: TimeInterval, frames: Range<Int>, kind: Kind,
        boundary: Boundary
    ) {
        self.start = start
        self.end = end
        self.frames = frames
        self.kind = kind
        self.boundary = boundary
    }

    public var duration: TimeInterval { end - start }
}

public enum OmniShotCutError: Error, Sendable, CustomStringConvertible {
    case videoUnavailable(URL)
    case noVideoTrack(URL)
    case noFrames(URL)
    case decodingFailed(URL, String)
    case unsupportedPixelFormat(URL)
    case modelUnavailable(String, String)

    public var description: String {
        switch self {
        case .videoUnavailable(let url): return "\(url.path): no such file"
        case .noVideoTrack(let url): return "\(url.lastPathComponent): no video track"
        case .noFrames(let url): return "\(url.lastPathComponent): no readable frames"
        case .decodingFailed(let url, let detail):
            return "\(url.lastPathComponent): \(detail)"
        case .unsupportedPixelFormat(let url):
            return "\(url.lastPathComponent): decoder did not return 4:2:0 YUV"
        case .modelUnavailable(let model, let detail): return "\(model): \(detail)"
        }
    }
}
