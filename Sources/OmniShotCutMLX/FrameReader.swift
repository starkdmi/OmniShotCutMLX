import AVFoundation
import CoreVideo
import Foundation
import Metal

// OmniShotCut's official code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`,
// and the model notices the difference. So the decoder's YUV is taken as it
// is — H.264 decoding is bit-exact — and scaled and converted to the same
// bytes as ffmpeg's (FrameScaler.swift, docs/frame-scaler-spec.md).

/// Every frame of the first video track as the official pipeline sees it:
/// 128×96 RGB, stretched, turned and mirrored as ffmpeg auto-rotates.
///
/// The decoder's planes are copied a batch at a time into shared Metal
/// buffers and scaled and converted on the GPU while the next batch decodes:
/// resizing full-HD planes on the CPU cost four times the decode itself.
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
    private let metal: MetalContext
    private var scaler: FrameScaler?
    private var scalerColour: Colour?
    private var ready: [(pixels: [UInt8], time: TimeInterval)] = []
    /// One batch converts on the GPU from one slot while the next decodes
    /// into the other; a batch's frames are copied out of its slot before
    /// the slot is filled again.
    private let slots = [Slot(), Slot()]
    private var fillSlot = 0
    /// A batch whose conversion is running on the GPU while the next one is
    /// decoded.
    private var converting: Batch?
    private var carried: CMSampleBuffer?
    private var finished = false

    /// Frames scaled together at most, enough to keep the GPU busy.
    private static let batch = 16
    /// Bytes of decoded planes a batch holds at most, in each of two slots:
    /// 16 frames up to 1080p, 5 of 4K.
    private static let planeBudget = 64 << 20

    private struct Batch {
        let slot: Int
        let times: [TimeInterval]
        let commandBuffer: MTLCommandBuffer
    }

    /// The matrix and range a frame is converted with.
    private struct Colour: Hashable {
        var matrix: YCbCrMatrix
        var fullRange: Bool
    }

    init(url: URL, width: Int, height: Int) async throws {
        guard let metal = MetalContext.shared else {
            throw OmniShotCutError.metalUnavailable("no Metal device")
        }
        self.metal = metal
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

    deinit {
        reader.cancelReading()
    }

    /// RGB bytes and presentation time of the next frame, `nil` at the end.
    func next() throws -> (pixels: [UInt8], time: TimeInterval)? {
        while ready.isEmpty, !finished || converting != nil {
            let previous = converting
            converting = nil
            if !finished { try fill() }
            if let previous { try collect(previous) }
        }
        return ready.isEmpty ? nil : ready.removeFirst()
    }

    private func collect(_ batch: Batch) throws {
        batch.commandBuffer.waitUntilCompleted()
        guard batch.commandBuffer.status == .completed else {
            let detail =
                batch.commandBuffer.error.map { "\($0)" } ?? "command buffer not completed"
            throw OmniShotCutError.decodingFailed(url, "GPU conversion failed: \(detail)")
        }
        let rgb = slots[batch.slot].rgb!.contents()
        let frameBytes = width * height * 3
        for (index, time) in batch.times.enumerated() {
            let frame = UnsafeRawBufferPointer(
                start: rgb + index * frameBytes, count: frameBytes)
            ready.append(([UInt8](frame), time))
        }
    }

    /// Decode up to a batch of frames of one size and colour into the free
    /// slot and start converting them.
    private func fill() throws {
        let slot = slots[fillSlot]
        let device = metal.device
        var layout: PlaneLayout?
        var colour: Colour?
        var limit = Self.batch
        var times: [TimeInterval] = []
        times.reserveCapacity(Self.batch)
        while times.count < limit {
            guard let sample = carried ?? output.copyNextSampleBuffer() else {
                finished = true
                break
            }
            carried = nil
            guard let image = CMSampleBufferGetImageBuffer(sample) else { continue }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            CVPixelBufferLockBaseAddress(image, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
            guard CVPixelBufferGetPlaneCount(image) == 2,
                let lumaBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
                let chromaBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
            else {
                throw OmniShotCutError.unsupportedPixelFormat(url)
            }
            let frame = PlaneLayout(
                width: CVPixelBufferGetWidthOfPlane(image, 0),
                height: CVPixelBufferGetHeightOfPlane(image, 0),
                chromaWidth: CVPixelBufferGetWidthOfPlane(image, 1),
                chromaHeight: CVPixelBufferGetHeightOfPlane(image, 1))
            let frameColour = Colour(
                matrix: matrices.matrix(at: time),
                fullRange: CVPixelBufferGetPixelFormatType(image)
                    == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
            if let layout, frame != layout || frameColour != colour {
                // A new frame size or colour starts the next batch.
                carried = sample
                break
            }
            if layout == nil {
                layout = frame
                colour = frameColour
                limit = min(
                    Self.batch,
                    max(1, Self.planeBudget / (frame.lumaBytes + frame.chromaBytes)))
                try slot.reserve(\.luma, bytes: limit * frame.lumaBytes, device: device)
                try slot.reserve(\.chroma, bytes: limit * frame.chromaBytes, device: device)
            }
            let index = times.count
            Self.copy(
                lumaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 0),
                rows: frame.height, bytes: frame.width,
                to: slot.luma!.contents() + index * frame.lumaBytes)
            Self.copy(
                chromaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 1),
                rows: frame.chromaHeight, bytes: frame.chromaWidth * 2,
                to: slot.chroma!.contents() + index * frame.chromaBytes)
            times.append(time.seconds)
        }
        if reader.status == .failed {
            throw OmniShotCutError.decodingFailed(
                url, reader.error?.localizedDescription ?? "decoding failed")
        }
        guard let layout, let colour, !times.isEmpty else { return }

        if scaler?.layout != layout || scalerColour != colour {
            scaler = try FrameScaler(
                layout: layout, quarterTurns: orientation.quarterTurns,
                flipped: orientation.flipped, width: width, height: height,
                matrix: colour.matrix, fullRange: colour.fullRange, metal: metal)
            scalerColour = colour
        }
        let scaler = scaler!
        // The intermediate values never leave the GPU.
        try slot.reserve(
            \.scratch, bytes: scaler.scratchBytes(frames: limit), device: device,
            options: .storageModePrivate)
        try slot.reserve(\.rgb, bytes: limit * width * height * 3, device: device)
        guard let commandBuffer = metal.queue.makeCommandBuffer() else {
            throw OmniShotCutError.metalUnavailable("could not create a command buffer")
        }
        try scaler.encode(
            frames: times.count, luma: slot.luma!, chroma: slot.chroma!, scratch: slot.scratch!,
            rgb: slot.rgb!, into: commandBuffer)
        commandBuffer.commit()
        converting = Batch(slot: fillSlot, times: times, commandBuffer: commandBuffer)
        fillSlot ^= 1
    }

    /// Copies a plane without its row padding.
    private static func copy(
        _ base: UnsafeMutableRawPointer, rowBytes: Int, rows: Int, bytes: Int,
        to destination: UnsafeMutableRawPointer
    ) {
        if rowBytes == bytes {
            destination.copyMemory(from: base, byteCount: rows * bytes)
            return
        }
        for row in 0..<rows {
            (destination + row * bytes).copyMemory(
                from: base + row * rowBytes, byteCount: bytes)
        }
    }
}

/// The stored size of a frame's luma plane and of its interleaved chroma
/// plane, in pixels.
struct PlaneLayout: Sendable, Hashable {
    var width: Int
    var height: Int
    var chromaWidth: Int
    var chromaHeight: Int

    var lumaBytes: Int { width * height }
    var chromaBytes: Int { chromaWidth * chromaHeight * 2 }
}

/// One batch's buffers. The planes and the RGB output are shared with the
/// CPU; each buffer grows to the largest batch it has held and is reused
/// after that.
private final class Slot {
    var luma: MTLBuffer?
    var chroma: MTLBuffer?
    var scratch: MTLBuffer?
    var rgb: MTLBuffer?

    func reserve(
        _ buffer: ReferenceWritableKeyPath<Slot, MTLBuffer?>, bytes: Int, device: MTLDevice,
        options: MTLResourceOptions = .storageModeShared
    ) throws {
        if let existing = self[keyPath: buffer], existing.length >= bytes { return }
        // Released first, so the old and the new buffer are not held at once.
        self[keyPath: buffer] = nil
        guard let made = device.makeBuffer(length: max(bytes, 1), options: options) else {
            throw OmniShotCutError.metalUnavailable("could not allocate \(bytes) bytes")
        }
        self[keyPath: buffer] = made
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

    /// The same orientation as `FrameScaler` takes it: clockwise quarter
    /// turns, then whether the result is mirrored.
    var quarterTurns: Int {
        transposed ? (reversesRows ? 3 : 1) : (reversesColumns ? 2 : 0)
    }

    var flipped: Bool {
        transposed ? reversesRows == reversesColumns : reversesRows != reversesColumns
    }
}

/// Each frame's YCbCr matrix, from its own sample description: a track
/// joined from clips encoded differently has one per clip. The decoder's tag
/// won't do, as VideoToolbox tags untagged video with a guess — BT.709 even
/// at 480p, where ffmpeg takes BT.601. A track that can't provide sample
/// cursors gets its first description's matrix throughout.
private struct MatrixLookup {
    private let first: YCbCrMatrix
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

    func matrix(at time: CMTime) -> YCbCrMatrix {
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
    private static func matrix(_ description: CMFormatDescription) -> YCbCrMatrix {
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
