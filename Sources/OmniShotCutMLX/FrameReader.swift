import AVFoundation
import CoreVideo
import Foundation
@preconcurrency import MLX

// OmniShotCut's official code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`,
// and the model notices the difference. So the decoder's YUV is taken as it
// is — H.264 decoding is bit-exact — and scaled and converted with
// libswscale's own arithmetic (Swscale.swift).

/// Every frame of the first video track as the official pipeline sees it:
/// 128×96 RGB, stretched, rotated as ffmpeg auto-rotates.
///
/// The decoder's planes are copied out a few frames at a time and scaled and
/// converted on the GPU: resizing full-HD planes on the CPU cost four times
/// the decode itself.
final class FrameReader {
    let duration: TimeInterval
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    let url: URL
    private let width: Int
    private let height: Int
    /// Quarter turns clockwise to display the stored picture.
    private let quarterTurns: Int
    private let tables: SwscaleRGBTables
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
        duration = try await asset.load(.duration).seconds
        let transform = try await track.load(.preferredTransform)
        let angle = atan2(transform.b, transform.a)
        quarterTurns = (Int((angle / (Double.pi / 2)).rounded()) % 4 + 4) % 4

        let descriptions = try await track.load(.formatDescriptions)
        let extensions = descriptions.first.flatMap {
            CMFormatDescriptionGetExtensions($0) as? [String: Any]
        }
        let matrix = extensions?[kCMFormatDescriptionExtension_YCbCrMatrix as String] as? String
        let fullRange =
            (extensions?[kCMFormatDescriptionExtension_FullRangeVideo as String] as? Bool)
            ?? false
        tables = SwscaleRGBTables(matrix: Self.matrix(matrix), fullRange: fullRange)

        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    fullRange
                    ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                    : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
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

    /// ffmpeg reads the stream's matrix; untagged video is BT.601 there, not
    /// guessed from the resolution.
    private static func matrix(_ tag: String?) -> SwscaleRGBTables.Matrix {
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String { return .bt709 }
        if tag == kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995 as String {
            return .smpte240m
        }
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as String { return .bt2020 }
        return .bt601
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

    /// Decode up to a batch of same-sized frames and start converting them.
    private func fill() throws {
        var planes = PlaneBatch()
        while planes.count < Self.batch {
            guard let sample = carried ?? output.copyNextSampleBuffer() else {
                finished = true
                break
            }
            carried = nil
            guard let image = CMSampleBufferGetImageBuffer(sample) else { continue }
            if !planes.append(
                image, time: CMSampleBufferGetPresentationTimeStamp(sample).seconds)
            {
                // A new frame size starts the next batch.
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
        luma = Self.rotate(luma, turns: quarterTurns)
        chroma = Self.rotate(chroma, turns: quarterTurns)
        let sourceWidth = luma.dim(2)
        let sourceHeight = luma.dim(1)
        if scaler?.matches(width: sourceWidth, height: sourceHeight) != true {
            scaler = PlaneScaler(
                sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                chromaWidth: chroma.dim(2), chromaHeight: chroma.dim(1), width: width,
                height: height, tables: tables)
        }
        let rgb = scaler!.convert(luma: luma, chroma: chroma)
        asyncEval(rgb)
        converting = (rgb, planes.times)
    }

    /// Turns `[frames, height, width, ...]` clockwise by quarter turns, as
    /// ffmpeg's `transpose=clock`, `hflip,vflip` and `transpose=cclock`.
    private static func rotate(_ planes: MLXArray, turns: Int) -> MLXArray {
        func reversed(_ array: MLXArray, axis: Int) -> MLXArray {
            let count = array.dim(axis)
            return array.take(MLXArray(Array((0..<Int32(count)).reversed())), axis: axis)
        }
        var axes = Array(0..<planes.ndim)
        axes.swapAt(1, 2)
        switch turns {
        case 1: return reversed(planes.transposed(axes: axes), axis: 2)
        case 2: return reversed(reversed(planes, axis: 1), axis: 2)
        case 3: return reversed(planes.transposed(axes: axes), axis: 1)
        default: return planes
        }
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
    var isBiplanar = true
    var count: Int { times.count }

    /// `false` when the frame's size differs from the batch's.
    mutating func append(_ image: CVPixelBuffer, time: TimeInterval) -> Bool {
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
        if count > 0, (w, h, cw, ch) != (width, height, chromaWidth, chromaHeight) {
            return false
        }
        (width, height, chromaWidth, chromaHeight) = (w, h, cw, ch)
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
