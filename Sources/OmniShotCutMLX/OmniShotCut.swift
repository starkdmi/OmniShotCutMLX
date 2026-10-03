import Foundation
import Hub
@preconcurrency import MLX
import MLXNN

/// Shots and transitions with OmniShotCut v1.5 on MLX.
///
/// Every frame is read, at the file's own rate, as the official code reads
/// them. Frames stream through one 100-frame window at a time, so an hour of
/// video costs one window of memory, not an hour of frames.
///
/// Windowing and stitching are the official `engine.py`'s, step for step:
/// windows of 100 frames overlapping by 10, each window trusted for its half
/// of the overlap, a cut emitted by the window whose valid region holds it,
/// and every segment labelled by the cut that begins it. `ReferenceTests`
/// holds it to the official PyTorch code's output on three open movies.
public actor OmniShotCut {

    /// The official v1.5 checkpoint converted by `scripts/convert.py
    /// --bits 0`: stored in fp16, 106 MB, computed in fp32. It finds every
    /// cut PyTorch does with the same labels. The 8-bit conversion
    /// (`starkdmi/OmniShotCut-v1.5-mlx-8bit`, 67 MB) relabels or moves a
    /// few, mostly in fades.
    public static let defaultModel = "starkdmi/OmniShotCut-v1.5-mlx-fp16"

    /// Frames shared by neighbouring windows. The official default is 10.
    public nonisolated let overlap: Int
    private let model: OmniShotCutModel

    /// Loads a directory written by `scripts/convert.py`.
    public init(directory: URL, overlap: Int = 10) throws {
        self.model = try Self.load(from: directory)
        self.overlap = max(0, overlap)
    }

    /// Downloads a converted model from the Hugging Face Hub, or reuses the
    /// copy already under `downloadBase` (`~/Documents/huggingface` unless
    /// given), and loads it.
    public static func pretrained(
        _ id: String = defaultModel, revision: String = "main", downloadBase: URL? = nil,
        overlap: Int = 10, progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> OmniShotCut {
        let directory: URL
        do {
            directory = try await HubApi(downloadBase: downloadBase).snapshot(
                from: Hub.Repo(id: id), revision: revision,
                matching: ["*.safetensors", "*.json"], progressHandler: progress)
        } catch {
            throw OmniShotCutError.modelUnavailable(id, String(describing: error))
        }
        return try OmniShotCut(directory: directory, overlap: overlap)
    }

    /// Segments covering the whole video, in order, without gaps.
    public func segments(in video: URL) async throws -> [ShotSegment] {
        guard FileManager.default.fileExists(atPath: video.path) else {
            throw OmniShotCutError.videoUnavailable(video)
        }
        let reader = try await FrameReader(
            url: video, width: model.configuration.width, height: model.configuration.height)
        return try segments(from: reader)
    }

    /// The whole pipeline over the model-sized RGB frames of one video.
    private func segments(from source: FrameReader) throws -> [ShotSegment] {
        let window = model.configuration.windowLength
        let overlap = min(self.overlap, window - 1)
        let frameBytes = model.configuration.width * model.configuration.height * 3

        var buffer: [UInt8] = []
        buffer.reserveCapacity((window + 1) * frameBytes)
        var times: [TimeInterval] = []
        var windowStart = 0
        var cuts: [Cut] = []
        // The model runs on one window while the next one's frames decode.
        var running:
            (predictions: Predictions, start: Int, validLength: Int, bounds: (Int, Int))?

        func finishRunning() {
            guard let pending = running else { return }
            running = nil
            let segments = Self.windowSegments(
                logits: pending.predictions.read(), validLength: pending.validLength)
            cuts += Self.cuts(
                from: segments, windowStart: pending.start, validStart: pending.bounds.0,
                validEnd: pending.bounds.1)
        }

        // A full window is started only once a frame past it has been read:
        // the last window is the one that reaches the end, and the official
        // code treats it differently.
        func start(final: Bool) {
            let available = buffer.count / frameBytes
            let validLength = min(window, available)
            let bounds = Self.validRange(
                windowStart: windowStart, window: window, overlap: overlap,
                total: final ? windowStart + available : Int.max)
            let predictions = predict(frames: buffer, count: validLength, window: window)
            finishRunning()
            running = (predictions, windowStart, validLength, (bounds.start, bounds.end))
        }

        while let frame = try source.next() {
            try Task.checkCancellation()
            buffer += frame.pixels
            times.append(frame.time)
            if buffer.count / frameBytes > window {
                start(final: false)
                let stride = window - overlap
                buffer.removeFirst(stride * frameBytes)
                windowStart += stride
            }
        }
        guard !times.isEmpty else { throw OmniShotCutError.noFrames(source.url) }
        start(final: true)
        finishRunning()

        return Self.assemble(cuts: cuts, frameCount: times.count).map { range in
            ShotSegment(
                start: times[range.start],
                end: range.end < times.count ? times[range.end] : source.duration,
                frames: range.start..<range.end,
                kind: ShotSegment.Kind(rawValue: range.intra) ?? .shot,
                boundary: ShotSegment.Boundary(rawValue: range.inter) ?? .hardCut)
        }
    }

    // MARK: - Inference

    /// One window's per-query argmaxes, computing on the GPU until read.
    struct Predictions {
        let intra: MLXArray
        let inter: MLXArray
        let end: MLXArray

        func read() -> WindowLogits {
            func values(_ array: MLXArray) -> [Int] { array.asArray(Int32.self).map(Int.init) }
            return WindowLogits(intra: values(intra), inter: values(inter), end: values(end))
        }
    }

    /// The official transform — RGB / 255, ImageNet mean and deviation,
    /// black frames past the end of the video — then the model, started
    /// without waiting for it.
    private func predict(frames: [UInt8], count: Int, window: Int) -> Predictions {
        let height = model.configuration.height
        let width = model.configuration.width
        let frameBytes = width * height * 3
        var pixels = MLXArray(
            Array(frames[..<(count * frameBytes)]), [count, height, width, 3])
        if count < window {
            // Padding is black before normalization, as in `split_videos`.
            pixels = concatenated(
                [pixels, MLXArray.zeros([window - count, height, width, 3], dtype: .uint8)],
                axis: 0)
        }
        let mean = MLXArray([0.485, 0.456, 0.406] as [Float])
        let deviation = MLXArray([0.229, 0.224, 0.225] as [Float])
        let input = (pixels.asType(.float32) / 255 - mean) / deviation
        let (intra, inter, end) = model(input)
        // Argmax over every class but the last, as the official code does
        // after its softmax. Softmax does not change the order.
        func argmax(_ logits: MLXArray) -> MLXArray {
            logits[0..., ..<(logits.dim(1) - 1)].argMax(axis: 1).asType(.int32)
        }
        let predictions = Predictions(
            intra: argmax(intra), inter: argmax(inter), end: argmax(end))
        asyncEval(predictions.intra, predictions.inter, predictions.end)
        return predictions
    }

    static func load(from directory: URL) throws -> OmniShotCutModel {
        let configuration: OmniShotCutConfiguration
        do {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            configuration = try JSONDecoder().decode(OmniShotCutConfiguration.self, from: data)
        } catch {
            throw OmniShotCutError.modelUnavailable(directory.path, String(describing: error))
        }
        let model = OmniShotCutModel(configuration)
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: file) {
                let renamed =
                    key
                    .replacingOccurrences(of: ".downsample.0.", with: ".downsample.conv.")
                    .replacingOccurrences(of: ".downsample.1.", with: ".downsample.norm.")
                weights[renamed] = value
            }
        }
        if let quantization = configuration.quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil
                    ? (quantization.groupSize, quantization.bits, .affine) : nil
            }
        }
        // Stored in fp16 to halve the download; computed in fp32, because
        // fp16 activations move cuts and change labels that fp16 weights do
        // not.
        for (key, value) in weights where value.dtype == .float16 {
            weights[key] = value.asType(.float32)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    // MARK: - Stitching (engine.py)

    struct WindowLogits: Sendable {
        var intra: [Int]
        var inter: [Int]
        var end: [Int]
    }

    struct Cut: Sendable, Hashable {
        var position: Int
        var intra: Int
        var inter: Int
    }

    struct FrameRange: Sendable, Hashable {
        var start: Int
        var end: Int
        var intra: Int
        var inter: Int
    }

    /// `split_videos`: the global frame positions `(start, end]` a window's
    /// cuts are trusted in. `total` is `Int.max` until the window that
    /// reaches the end of the video.
    static func validRange(windowStart: Int, window: Int, overlap: Int, total: Int)
        -> (start: Int, end: Int)
    {
        let left = overlap / 2
        let right = overlap - left
        let windowEnd = windowStart + window
        let start = windowStart == 0 ? 0 : windowStart + left
        let end = windowEnd >= total ? total : windowEnd - right
        return (start, end)
    }

    /// `decode_window_segments`: queries in order, each ending where it says
    /// and starting where the previous one ended; a query that does not move
    /// forward is skipped, and the window stops at its last real frame.
    static func windowSegments(logits: WindowLogits, validLength: Int) -> [FrameRange] {
        var segments: [FrameRange] = []
        var start = 0
        for query in logits.end.indices {
            let end = min(logits.end[query], validLength)
            if start >= end { continue }
            segments.append(
                FrameRange(
                    start: start, end: end, intra: logits.intra[query],
                    inter: logits.inter[query]))
            start = end
            if end >= validLength { break }
        }
        return segments
    }

    /// `collect_cuts_from_window`: every segment start but the window's first
    /// becomes a cut, if it falls in `(validStart, validEnd]`.
    static func cuts(
        from segments: [FrameRange], windowStart: Int, validStart: Int, validEnd: Int
    ) -> [Cut] {
        segments.dropFirst().compactMap { segment in
            let position = windowStart + segment.start
            guard validStart < position, position <= validEnd else { return nil }
            return Cut(position: position, intra: segment.intra, inter: segment.inter)
        }
    }

    /// `assemble_ranges`: sorted, de-duplicated cuts partition the video; the
    /// first range is the opening shot and every later one takes the labels
    /// of the cut that starts it.
    static func assemble(cuts: [Cut], frameCount: Int) -> [FrameRange] {
        var kept: [Cut] = []
        for cut in cuts.sorted(by: { $0.position < $1.position }) {
            if let last = kept.last, cut.position == last.position { continue }
            if cut.position <= 0 || cut.position >= frameCount { continue }
            kept.append(cut)
        }
        let opening = (
            intra: ShotSegment.Kind.shot.rawValue,
            inter: ShotSegment.Boundary.videoStart.rawValue
        )
        var ranges: [FrameRange] = []
        var previous = 0
        for (index, cut) in kept.enumerated() where cut.position > previous {
            let labels = index == 0 ? opening : (kept[index - 1].intra, kept[index - 1].inter)
            ranges.append(
                FrameRange(start: previous, end: cut.position, intra: labels.0, inter: labels.1)
            )
            previous = cut.position
        }
        if previous < frameCount {
            let labels = kept.last.map { ($0.intra, $0.inter) } ?? opening
            ranges.append(
                FrameRange(start: previous, end: frameCount, intra: labels.0, inter: labels.1))
        }
        return ranges
    }
}
