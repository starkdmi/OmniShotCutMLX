import AVFoundation
import CoreVideo
import Foundation
@preconcurrency import MLX

// OmniShotCut's official code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`,
// and the model notices the difference. So the decoder's YUV is taken as it
// is — H.264 decoding is bit-exact — and scaled and converted with
// libswscale's own arithmetic (Swscale.swift).

/// Every frame of the first video track as the official pipeline sees it:
/// 128×96 RGB, stretched, turned and mirrored as ffmpeg auto-rotates.
///
/// The decoder's planes are copied out a few frames at a time and scaled and
/// converted on the GPU: resizing full-HD planes on the CPU cost four times
/// the decode itself.
final class FrameReader {
    /// Where the video track ends. Audio running past the last frame does not
    /// count.
    let end: TimeInterval
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    let url: URL
    private let width: Int
    private let height: Int
    private let orientation: Orientation
    private let matrices: MatrixLookup
    private var scaler: PlaneScaler?
    private var ready: [(pixels: [UInt8], time: TimeInterval)] = []
    /// A batch whose conversion is running on the GPU while the next one is
    /// decoded.
    private var converting: (rgb: MLXArray, times: [TimeInterval])?
    private var carried: CMSampleBuffer?
    private var finished = false

    /// Frames scaled together. Bounds the planes held at once — 16 frames
    /// of 4K is 200 MB — while keeping the GPU busy.
    private static let batch = 16

    init(url: URL, width: Int, height: Int) async throws {
        self.url = url
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw OmniShotCutError.noVideoTrack(url)
        }
        let (timeRange, transform) = try await track.load(.timeRange, .preferredTransform)
        end = timeRange.end.seconds
        orientation = Orientation(transform)
        matrices = try await MatrixLookup(track)

        reader = try AVAssetReader(asset: asset)
        // Offered both ranges, the decoder hands over each frame's values in
        // the range they were coded in, and the frame's format says which.
        // Asked for one, it converts the others.
        output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: [
                    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                ]
            ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw OmniShotCutError.decodingFailed(
                url, reader.error?.localizedDescription ?? "unreadable video")
        }
        self.width = width
        self.height = height
    }

    /// RGB bytes and presentation time of the next frame, `nil` at the end.
    func next() throws -> (pixels: [UInt8], time: TimeInterval)? {
        while ready.isEmpty, !finished || converting != nil {
            let previous = converting
            converting = nil
            if !finished { try fill() }
            if let previous { collect(previous) }
        }
        return ready.isEmpty ? nil : ready.removeFirst()
    }

    private func collect(_ batch: (rgb: MLXArray, times: [TimeInterval])) {
        let rgb = batch.rgb.asArray(UInt8.self)
        let frameBytes = width * height * 3
        for (index, time) in batch.times.enumerated() {
            ready.append((Array(rgb[(index * frameBytes)..<((index + 1) * frameBytes)]), time))
        }
    }

    /// Decode up to a batch of frames of one size and colour and start
    /// converting them.
    private func fill() throws {
        var planes = PlaneBatch()
        while planes.count < Self.batch {
            guard let sample = carried ?? output.copyNextSampleBuffer() else {
                finished = true
                break
            }
            carried = nil
            guard let image = CMSampleBufferGetImageBuffer(sample) else { continue }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            if !planes.append(image, time: time.seconds, matrix: matrices.matrix(at: time)) {
                // A new frame size or colour starts the next batch.
                carried = sample
                break
            }
        }
        if reader.status == .failed {
            throw OmniShotCutError.decodingFailed(
                url, reader.error?.localizedDescription ?? "decoding failed")
        }
        guard planes.count > 0 else { return }
        guard planes.isBiplanar else { throw OmniShotCutError.unsupportedPixelFormat(url) }

        let count = planes.count
        var luma = MLXArray(planes.luma, [count, planes.height, planes.width])
        var chroma = MLXArray(
            planes.chroma, [count, planes.chromaHeight, planes.chromaWidth, 2])
        // ffmpeg turns the picture before it scales it, so the filters run
        // along the displayed axes.
        luma = orientation.apply(to: luma)
        chroma = orientation.apply(to: chroma)
        let sourceWidth = luma.dim(2)
        let sourceHeight = luma.dim(1)
        if scaler?.matches(width: sourceWidth, height: sourceHeight, colour: planes.colour)
            != true
        {
            scaler = PlaneScaler(
                sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                chromaWidth: chroma.dim(2), chromaHeight: chroma.dim(1), width: width,
                height: height, colour: planes.colour)
        }
        let rgb = scaler!.convert(luma: luma, chroma: chroma)
        asyncEval(rgb)
        converting = (rgb, planes.times)
    }
}

/// How the stored picture is turned and mirrored for display, read from the
/// track's transform as ffmpeg's autorotate reads the display matrix: rows
/// and columns swapped or not, then either axis reversed. A mirror stays a
/// mirror rather than becoming a half turn.
struct Orientation {
    let transposed: Bool
    let reversesRows: Bool
    let reversesColumns: Bool

    init(_ transform: CGAffineTransform) {
        // x' = a·x + c·y and y' = b·x + d·y. After a quarter turn either way
        // the stored x runs down the screen, in the direction of b's sign,
        // and the stored y across it, in c's.
        transposed = abs(transform.b) > abs(transform.a)
        reversesRows = (transposed ? transform.b : transform.d) < 0
        reversesColumns = (transposed ? transform.c : transform.a) < 0
    }

    /// `[frames, height, width, ...]` as displayed, by ffmpeg's `transpose`,
    /// `vflip` and `hflip`.
    func apply(to planes: MLXArray) -> MLXArray {
        func reversed(_ array: MLXArray, axis: Int) -> MLXArray {
            let count = array.dim(axis)
            return array.take(MLXArray(Array((0..<Int32(count)).reversed())), axis: axis)
        }
        var planes = planes
        if transposed { planes = planes.swappedAxes(1, 2) }
        if reversesRows { planes = reversed(planes, axis: 1) }
        if reversesColumns { planes = reversed(planes, axis: 2) }
        return planes
    }
}

/// Each frame's YCbCr matrix, from its own sample description: a track
/// joined from clips encoded differently has one per clip. The decoder's tag
/// won't do, as VideoToolbox tags untagged video with a guess — BT.709 even
/// at 480p, where ffmpeg takes BT.601. A track that can't provide sample
/// cursors gets its first description's matrix throughout.
private struct MatrixLookup {
    private let first: SwscaleRGBTables.Matrix
    /// Set only when the descriptions disagree: the track, to find each
    /// frame's sample, and its edits, as sample cursors count time before
    /// them.
    private let track: AVAssetTrack?
    private let segments: [AVAssetTrackSegment]

    init(_ track: AVAssetTrack) async throws {
        let matrices = try await track.load(.formatDescriptions).map(Self.matrix)
        first = matrices.first ?? .bt601
        if Set(matrices).count > 1, try await track.load(.canProvideSampleCursors) {
            self.track = track
            segments = try await track.load(.segments)
        } else {
            self.track = nil
            segments = []
        }
    }

    func matrix(at time: CMTime) -> SwscaleRGBTables.Matrix {
        guard let track,
            let mapping = segments.first(where: {
                !$0.isEmpty && $0.timeMapping.target.containsTime(time)
            })?.timeMapping,
            let cursor = track.makeSampleCursor(
                presentationTimeStamp: CMTimeMapTimeFromRangeToRange(
                    time, fromRange: mapping.target, toRange: mapping.source))
        else { return first }
        return Self.matrix(cursor.copyCurrentSampleFormatDescription())
    }

    /// ffmpeg reads the stream's matrix; untagged video is BT.601 there, not
    /// guessed from the resolution.
    private static func matrix(_ description: CMFormatDescription) -> SwscaleRGBTables.Matrix {
        let tag =
            CMFormatDescriptionGetExtension(
                description, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String { return .bt709 }
        if tag == kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995 as String {
            return .smpte240m
        }
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as String { return .bt2020 }
        return .bt601
    }
}

/// The luma and interleaved chroma planes of a few frames, copied out of the
/// decoder's buffers without their row padding.
private struct PlaneBatch {
    var luma: [UInt8] = []
    var chroma: [UInt8] = []
    var times: [TimeInterval] = []
    var width = 0
    var height = 0
    var chromaWidth = 0
    var chromaHeight = 0
    var colour = SwscaleRGBTables.Colour(matrix: .bt601, fullRange: false)
    var isBiplanar = true
    var count: Int { times.count }

    /// `false` when the frame's size or colour differs from the batch's.
    mutating func append(
        _ image: CVPixelBuffer, time: TimeInterval, matrix: SwscaleRGBTables.Matrix
    ) -> Bool {
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard CVPixelBufferGetPlaneCount(image) == 2,
            let lumaBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
            let chromaBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else {
            isBiplanar = false
            times.append(time)
            return true
        }
        let w = CVPixelBufferGetWidthOfPlane(image, 0)
        let h = CVPixelBufferGetHeightOfPlane(image, 0)
        let cw = CVPixelBufferGetWidthOfPlane(image, 1)
        let ch = CVPixelBufferGetHeightOfPlane(image, 1)
        let colour = SwscaleRGBTables.Colour(
            matrix: matrix,
            fullRange: CVPixelBufferGetPixelFormatType(image)
                == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        if count > 0,
            (w, h, cw, ch) != (width, height, chromaWidth, chromaHeight)
                || colour != self.colour
        {
            return false
        }
        (width, height, chromaWidth, chromaHeight) = (w, h, cw, ch)
        self.colour = colour
        func copy(
            _ base: UnsafeMutableRawPointer, rowBytes: Int, rows: Int, bytes: Int,
            into: inout [UInt8]
        ) {
            let source = base.assumingMemoryBound(to: UInt8.self)
            for row in 0..<rows {
                into.append(
                    contentsOf: UnsafeBufferPointer(
                        start: source + row * rowBytes, count: bytes))
            }
        }
        copy(
            lumaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 0), rows: h, bytes: w,
            into: &luma)
        copy(
            chromaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 1), rows: ch,
            bytes: cw * 2,
            into: &chroma)
        times.append(time)
        return true
    }
}
