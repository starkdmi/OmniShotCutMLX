import AVFoundation
import CoreVideo
import Foundation
@preconcurrency import MLX

// OmniShotCut's official code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`,
// and the model notices the difference. AVFoundation's own scaling and
// BT.709 conversion are 4–5 levels away on average and 1–3 levels brighter,
// and that moves cuts, most of them in fades, where brightness is the
// signal. Even a uniform one-level offset moves some.
//
// So the decoder's YUV is taken as it is — H.264 decoding is bit-exact — and
// scaled and converted the way libswscale does it: bicubic (Keys, a = -0.6)
// stretched over the scale factor when shrinking, luma and chroma planes
// scaled separately, chroma to half the output width with each value shared
// by two pixels, and YUV→RGB through swscale's integer tables with the
// matrix the stream is tagged with. 99.7% of values come out identical to
// ffmpeg 9's and the rest within a few levels, from swscale's fixed-point
// filter coefficients (`FrameReaderTests`).

/// libswscale's YUV→RGB for 8-bit, 24-bit output (`ff_yuv2rgb_c_init_tables`,
/// `yuv2rgb_X_c_template`): one clipped luma table, indexed by luma plus an
/// integer offset per chroma value.
struct SwscaleRGBTables: Sendable {
    let luma: [UInt8]
    let redV: [Int]
    let greenU: [Int]
    let greenV: [Int]
    let blueU: [Int]

    /// `ff_yuv2rgb_coeffs`, by the stream's matrix.
    enum Matrix: Sendable {
        case bt709, bt601, smpte240m, bt2020, fcc

        var coefficients: (Int64, Int64, Int64, Int64) {
            switch self {
            case .bt709: return (117_489, 138_438, 13_975, 34_925)
            case .bt601: return (104_597, 132_201, 25_675, 53_279)
            case .smpte240m: return (117_579, 136_230, 16_907, 35_559)
            case .bt2020: return (110_013, 140_363, 12_277, 42_626)
            case .fcc: return (104_448, 132_798, 24_759, 53_109)
            }
        }
    }

    init(matrix: Matrix, fullRange: Bool) {
        // C division truncates toward zero; Swift's does too.
        var (crv, cbu, cgu, cgv) = matrix.coefficients
        cgu = -cgu
        cgv = -cgv
        var cy: Int64 = 1 << 16
        var oy: Int64 = 0
        if fullRange {
            crv = crv * 224 / 255
            cbu = cbu * 224 / 255
            cgu = cgu * 224 / 255
            cgv = cgv * 224 / 255
        } else {
            cy = cy * 255 / 219
            oy = 16 << 16
        }
        crv = (crv * 65_536 + 0x8000) / cy
        cbu = (cbu * 65_536 + 0x8000) / cy
        cgu = (cgu * 65_536 + 0x8000) / cy
        cgv = (cgv * 65_536 + 0x8000) / cy

        let headroom: Int64 = 512
        var yb = -(384 << 16) - headroom * cy - oy
        var luma = [UInt8](repeating: 0, count: 2_048)
        for index in luma.indices {
            luma[index] = UInt8(clamping: (yb + 0x8000) >> 16)
            yb += cy
        }
        self.luma = luma
        let offset = (fullRange ? 384 : 326) + Int(headroom)
        func table(_ increment: Int64, base: Int) -> [Int] {
            (0..<256).map { value in
                base - Int(increment >> 9) + Int((Int64(value) * increment) >> 16)
            }
        }
        redV = table(crv, base: offset)
        greenU = table(cgu, base: offset)
        greenV = table(cgv, base: 0)
        blueU = table(cbu, base: offset)
    }
}

/// libswscale's bicubic filter as a dense matrix, `[destination, source]`,
/// each row normalized to one. Shrinking stretches the kernel over the scale
/// factor, which is what keeps a 15× reduction from aliasing.
enum SwscaleBicubic {
    static func matrix(source: Int, destination: Int) -> [Float] {
        let a = -0.6
        let scale = Double(source) / Double(destination)
        let stretch = max(scale, 1)
        let support = 2 * stretch
        func kernel(_ distance: Double) -> Double {
            let x = abs(distance)
            if x < 1 { return (a + 2) * x * x * x - (a + 3) * x * x + 1 }
            if x < 2 { return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a }
            return 0
        }
        var weights = [Float](repeating: 0, count: destination * source)
        for row in 0..<destination {
            let center = (Double(row) + 0.5) * scale - 0.5
            var taps = [Double](repeating: 0, count: source)
            let first = Int((center - support).rounded(.down)) - 1
            let last = Int((center + support).rounded(.up)) + 1
            for index in first...last {
                // Past the edge, the edge pixel repeats.
                let clamped = min(max(index, 0), source - 1)
                taps[clamped] += kernel((Double(index) - center) / stretch)
            }
            let total = taps.reduce(0, +)
            for index in 0..<source where taps[index] != 0 {
                weights[row * source + index] = Float(taps[index] / total)
            }
        }
        return weights
    }
}

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
    private var converting: (rgb: MLXArray, times: [TimeInterval], width: Int, height: Int)?
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

    private func collect(
        _ batch: (rgb: MLXArray, times: [TimeInterval], width: Int, height: Int)
    ) {
        let rgb = batch.rgb.asArray(UInt8.self)
        let frameBytes = batch.width * batch.height * 3
        for (index, time) in batch.times.enumerated() {
            let frame = Array(rgb[(index * frameBytes)..<((index + 1) * frameBytes)])
            ready.append(
                (
                    Self.rotate(
                        frame, width: batch.width, height: batch.height, turns: quarterTurns),
                    time
                ))
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
        guard planes.isBiplanar else {
            throw OmniShotCutError.unsupportedPixelFormat(url)
        }
        // A picture displayed on its side is scaled to the transposed size
        // and turned afterwards: the same as turning first, then scaling.
        let sideways = quarterTurns % 2 == 1
        let scaledWidth = sideways ? height : width
        let scaledHeight = sideways ? width : height
        if scaler?.matches(planes) != true {
            scaler = PlaneScaler(planes, width: scaledWidth, height: scaledHeight)
        }
        let rgb = scaler!.convert(planes, tables: tables)
        asyncEval(rgb)
        converting = (rgb, planes.times, scaledWidth, scaledHeight)
    }

    private static func rotate(_ rgb: [UInt8], width: Int, height: Int, turns: Int) -> [UInt8] {
        guard turns != 0 else { return rgb }
        var out = [UInt8](repeating: 0, count: rgb.count)
        let outWidth = turns % 2 == 1 ? height : width
        for y in 0..<height {
            for x in 0..<width {
                let (nx, ny): (Int, Int)
                switch turns {
                case 1: (nx, ny) = (height - 1 - y, x)
                case 2: (nx, ny) = (width - 1 - x, height - 1 - y)
                default: (nx, ny) = (y, width - 1 - x)
                }
                for c in 0..<3 {
                    out[(ny * outWidth + nx) * 3 + c] = rgb[(y * width + x) * 3 + c]
                }
            }
        }
        return out
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

/// The separable resize and table conversion of a batch, on the GPU:
/// `vertical · plane · horizontalᵀ` per plane, rounded, then swscale's
/// tables looked up per pixel.
private struct PlaneScaler {
    let sourceWidth: Int
    let sourceHeight: Int
    let width: Int
    let height: Int
    let chromaWidth: Int
    private let lumaHorizontal: MLXArray
    private let lumaVertical: MLXArray
    private let chromaHorizontal: MLXArray
    private let chromaVertical: MLXArray

    init(_ planes: PlaneBatch, width: Int, height: Int) {
        sourceWidth = planes.width
        sourceHeight = planes.height
        self.width = width
        self.height = height
        // Packed RGB output keeps chroma at half its width (swscale without
        // full chroma interpolation) and full height.
        chromaWidth = (width + 1) / 2
        func matrix(_ source: Int, _ destination: Int) -> MLXArray {
            MLXArray(
                SwscaleBicubic.matrix(source: source, destination: destination),
                [destination, source])
        }
        lumaHorizontal = matrix(planes.width, width).transposed()
        lumaVertical = matrix(planes.height, height)
        chromaHorizontal = matrix(planes.chromaWidth, chromaWidth).transposed()
        chromaVertical = matrix(planes.chromaHeight, height)
    }

    func matches(_ planes: PlaneBatch) -> Bool {
        sourceWidth == planes.width && sourceHeight == planes.height
    }

    /// `[frames, height, width, 3]` RGB bytes, not yet evaluated.
    func convert(_ planes: PlaneBatch, tables: SwscaleRGBTables) -> MLXArray {
        let count = planes.count
        let luma = MLXArray(planes.luma, [count, planes.height, planes.width])
        let chroma = MLXArray(
            planes.chroma, [count, planes.chromaHeight, planes.chromaWidth, 2])
        // Bicubic rings past black and white. swscale clips chroma before its
        // tables and leaves luma alone — the tables carry 512 entries of
        // headroom either side for it.
        func resize(
            _ plane: MLXArray, vertical: MLXArray, horizontal: MLXArray, range: (Int, Int)
        ) -> MLXArray {
            let scaled = matmul(matmul(vertical, plane.asType(.float32)), horizontal)
            return clip(round(scaled), min: range.0, max: range.1).asType(.int32)
        }
        let y = resize(
            luma, vertical: lumaVertical, horizontal: lumaHorizontal, range: (-512, 767))
        // Each chroma value serves two neighbouring pixels.
        func widen(_ plane: MLXArray) -> MLXArray {
            repeated(plane, count: 2, axis: 2)[0..., 0..., ..<width]
        }
        let u = widen(
            resize(
                chroma[.ellipsis, 0], vertical: chromaVertical, horizontal: chromaHorizontal,
                range: (0, 255)))
        let v = widen(
            resize(
                chroma[.ellipsis, 1], vertical: chromaVertical, horizontal: chromaHorizontal,
                range: (0, 255)))
        let lumaTable = MLXArray(tables.luma.map { Int32($0) })
        func table(_ values: [Int]) -> MLXArray { MLXArray(values.map { Int32($0) }) }
        let red = lumaTable[table(tables.redV)[v] + y]
        let green = lumaTable[table(tables.greenU)[u] + table(tables.greenV)[v] + y]
        let blue = lumaTable[table(tables.blueU)[u] + y]
        return stacked([red, green, blue], axis: -1).asType(.uint8)
    }
}
