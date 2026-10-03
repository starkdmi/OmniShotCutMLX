import Foundation
import MLX
import MLXNN

// OmniShotCut (Wang et al., arXiv 2604.24762), ported from the official
// PyTorch code at UVA-Computer-Vision-Lab/OmniShotCut. Module names follow the
// official state dict, so a checkpoint converts by renaming nothing but
// `backbone.0.body.` and splitting each attention's packed `in_proj` — see
// scripts/convert.py. A ResNet18 reads every frame of a
// 100-frame window, a DETR encoder attends over all of them at once, and 24
// shot queries each predict where their shot ends and what kind it is.

/// `config.json` of a converted checkpoint.
struct OmniShotCutConfiguration: Codable, Sendable {
    struct Quantization: Codable, Sendable {
        var groupSize: Int
        var bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    var hiddenDimension: Int
    var encoderLayers: Int
    var decoderLayers: Int
    var heads: Int
    var feedForwardDimension: Int
    var queries: Int
    var intraClasses: Int
    var interClasses: Int
    var windowLength: Int
    var width: Int
    var height: Int
    var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case hiddenDimension = "hidden_dim"
        case encoderLayers = "enc_layers"
        case decoderLayers = "dec_layers"
        case heads = "nheads"
        case feedForwardDimension = "dim_feedforward"
        case queries = "num_queries"
        case intraClasses = "num_intra_relation_classes"
        case interClasses = "num_inter_relation_classes"
        case windowLength = "max_process_window_length"
        case width = "process_width"
        case height = "process_height"
        case quantization
    }
}

// MARK: - Backbone

/// torchvision's ResNet18 with DETR's frozen batch norm: fixed statistics,
/// and an epsilon before the square root.
final class OmniShotCutFrozenBatchNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVariance: MLXArray

    init(_ channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
        _runningMean.wrappedValue = MLXArray.zeros([channels])
        _runningVariance.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = weight * rsqrt(runningVariance + 1e-5)
        return x * scale + (bias - runningMean * scale)
    }
}

/// torchvision's `downsample` Sequential. Its children are `0` and `1` in the
/// checkpoint, which `ModuleParameters.unflattened` would read as an array;
/// ``OmniShotCut/load(from:)`` renames them.
final class OmniShotCutDownsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "norm") var normalization: OmniShotCutFrozenBatchNorm

    init(input: Int, output: Int, stride: Int) {
        _convolution.wrappedValue = Conv2d(
            inputChannels: input, outputChannels: output, kernelSize: 1,
            stride: [stride, stride],
            bias: false)
        _normalization.wrappedValue = OmniShotCutFrozenBatchNorm(output)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { normalization(convolution(x)) }
}

final class OmniShotCutBasicBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: OmniShotCutFrozenBatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: OmniShotCutFrozenBatchNorm
    @ModuleInfo(key: "downsample") var downsample: OmniShotCutDownsample?

    init(input: Int, output: Int, stride: Int) {
        _conv1.wrappedValue = Conv2d(
            inputChannels: input, outputChannels: output, kernelSize: 3,
            stride: [stride, stride],
            padding: 1, bias: false)
        _bn1.wrappedValue = OmniShotCutFrozenBatchNorm(output)
        _conv2.wrappedValue = Conv2d(
            inputChannels: output, outputChannels: output, kernelSize: 3, padding: 1,
            bias: false)
        _bn2.wrappedValue = OmniShotCutFrozenBatchNorm(output)
        _downsample.wrappedValue =
            (stride != 1 || input != output)
            ? OmniShotCutDownsample(input: input, output: output, stride: stride) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = bn2(conv2(relu(bn1(conv1(x)))))
        return relu(y + (downsample?(x) ?? x))
    }
}

final class OmniShotCutResNet18: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: OmniShotCutFrozenBatchNorm
    @ModuleInfo(key: "layer1") var layer1: [OmniShotCutBasicBlock]
    @ModuleInfo(key: "layer2") var layer2: [OmniShotCutBasicBlock]
    @ModuleInfo(key: "layer3") var layer3: [OmniShotCutBasicBlock]
    @ModuleInfo(key: "layer4") var layer4: [OmniShotCutBasicBlock]
    // Padded by hand: `MaxPool2d(padding:)` in mlx-swift 0.30 pads the last
    // two axes of an NHWC input — width and channels — not height and width.
    let pool = MaxPool2d(kernelSize: 3, stride: 2)

    override init() {
        _conv1.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: 64, kernelSize: 7, stride: 2, padding: 3,
            bias: false)
        _bn1.wrappedValue = OmniShotCutFrozenBatchNorm(64)
        func layer(_ input: Int, _ output: Int, _ stride: Int) -> [OmniShotCutBasicBlock] {
            [
                OmniShotCutBasicBlock(input: input, output: output, stride: stride),
                OmniShotCutBasicBlock(input: output, output: output, stride: 1),
            ]
        }
        _layer1.wrappedValue = layer(64, 64, 1)
        _layer2.wrappedValue = layer(64, 128, 2)
        _layer3.wrappedValue = layer(128, 256, 2)
        _layer4.wrappedValue = layer(256, 512, 2)
        super.init()
    }

    /// NHWC in, NHWC out at 1/32 of the input size.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let features = relu(bn1(conv1(x)))
        var x = pool(
            padded(
                features, widths: [0, 1, 1, 0],
                value: MLXArray(-Float.infinity, dtype: features.dtype)))
        for block in layer1 + layer2 + layer3 + layer4 { x = block(x) }
        return x
    }
}

// MARK: - Transformer

/// `nn.MultiheadAttention` with its packed input projection split into three
/// linears, so each can be quantized like any other.
final class OmniShotCutAttention: Module {
    let heads: Int
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        _queryProjection.wrappedValue = Linear(dimensions, dimensions)
        _keyProjection.wrappedValue = Linear(dimensions, dimensions)
        _valueProjection.wrappedValue = Linear(dimensions, dimensions)
        _outputProjection.wrappedValue = Linear(dimensions, dimensions)
        super.init()
    }

    /// Batch-first, `[B, L, D]`.
    func callAsFunction(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
        let (batch, queryLength, dimensions) = (query.dim(0), query.dim(1), query.dim(2))
        let keyLength = key.dim(1)
        let headDimension = dimensions / heads
        func split(_ x: MLXArray, _ length: Int) -> MLXArray {
            x.reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(queryProjection(query), queryLength),
            keys: split(keyProjection(key), keyLength),
            values: split(valueProjection(value), keyLength),
            scale: 1 / Float(headDimension).squareRoot(), mask: nil)
        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batch, queryLength, dimensions))
    }
}

/// DETR's post-norm encoder layer: the position is added to queries and keys
/// only.
final class OmniShotCutEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: OmniShotCutAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(_ configuration: OmniShotCutConfiguration) {
        let d = configuration.hiddenDimension
        _selfAttention.wrappedValue = OmniShotCutAttention(
            dimensions: d, heads: configuration.heads)
        _linear1.wrappedValue = Linear(d, configuration.feedForwardDimension)
        _linear2.wrappedValue = Linear(configuration.feedForwardDimension, d)
        _norm1.wrappedValue = LayerNorm(dimensions: d)
        _norm2.wrappedValue = LayerNorm(dimensions: d)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, position: MLXArray) -> MLXArray {
        let withPosition = x + position
        let x = norm1(x + selfAttention(query: withPosition, key: withPosition, value: x))
        return norm2(x + linear2(relu(linear1(x))))
    }
}

final class OmniShotCutDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: OmniShotCutAttention
    @ModuleInfo(key: "multihead_attn") var crossAttention: OmniShotCutAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(_ configuration: OmniShotCutConfiguration) {
        let d = configuration.hiddenDimension
        _selfAttention.wrappedValue = OmniShotCutAttention(
            dimensions: d, heads: configuration.heads)
        _crossAttention.wrappedValue = OmniShotCutAttention(
            dimensions: d, heads: configuration.heads)
        _linear1.wrappedValue = Linear(d, configuration.feedForwardDimension)
        _linear2.wrappedValue = Linear(configuration.feedForwardDimension, d)
        _norm1.wrappedValue = LayerNorm(dimensions: d)
        _norm2.wrappedValue = LayerNorm(dimensions: d)
        _norm3.wrappedValue = LayerNorm(dimensions: d)
        super.init()
    }

    func callAsFunction(
        _ target: MLXArray, memory: MLXArray, position: MLXArray, queryPosition: MLXArray
    ) -> MLXArray {
        let withPosition = target + queryPosition
        var target = norm1(
            target + selfAttention(query: withPosition, key: withPosition, value: target))
        target = norm2(
            target
                + crossAttention(
                    query: target + queryPosition, key: memory + position, value: memory))
        return norm3(target + linear2(relu(linear1(target))))
    }
}

final class OmniShotCutEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [OmniShotCutEncoderLayer]

    init(_ configuration: OmniShotCutConfiguration) {
        _layers.wrappedValue = (0..<configuration.encoderLayers).map { _ in
            OmniShotCutEncoderLayer(configuration)
        }
        super.init()
    }
}

final class OmniShotCutDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [OmniShotCutDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ configuration: OmniShotCutConfiguration) {
        _layers.wrappedValue = (0..<configuration.decoderLayers).map { _ in
            OmniShotCutDecoderLayer(configuration)
        }
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hiddenDimension)
        super.init()
    }
}

final class OmniShotCutTransformer: Module {
    @ModuleInfo(key: "encoder") var encoder: OmniShotCutEncoder
    @ModuleInfo(key: "decoder") var decoder: OmniShotCutDecoder

    init(_ configuration: OmniShotCutConfiguration) {
        _encoder.wrappedValue = OmniShotCutEncoder(configuration)
        _decoder.wrappedValue = OmniShotCutDecoder(configuration)
        super.init()
    }
}

// MARK: - Model

final class OmniShotCutModel: Module {
    let configuration: OmniShotCutConfiguration
    @ModuleInfo(key: "backbone") var backbone: OmniShotCutResNet18
    @ModuleInfo(key: "transformer") var transformer: OmniShotCutTransformer
    @ModuleInfo(key: "input_proj") var inputProjection: Conv2d
    @ModuleInfo(key: "query_embed") var queryEmbedding: Embedding
    @ModuleInfo(key: "intra_relation_class_embed") var intraHead: Linear
    @ModuleInfo(key: "inter_relation_class_embed") var interHead: Linear
    @ModuleInfo(key: "range_class_embed") var rangeHead: Linear

    /// Computed once per feature-map shape; every window has the same one.
    private var positionCache: (key: [Int], value: MLXArray)?

    init(_ configuration: OmniShotCutConfiguration) {
        self.configuration = configuration
        let d = configuration.hiddenDimension
        _backbone.wrappedValue = OmniShotCutResNet18()
        _transformer.wrappedValue = OmniShotCutTransformer(configuration)
        _inputProjection.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: d, kernelSize: 1)
        _queryEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.queries, dimensions: d)
        _intraHead.wrappedValue = Linear(d, configuration.intraClasses)
        _interHead.wrappedValue = Linear(d, configuration.interClasses)
        // Two classes past the window: "ends after the window" and padding.
        _rangeHead.wrappedValue = Linear(d, configuration.windowLength + 2)
        super.init()
    }

    /// One window, `[F, H, W, 3]`, ImageNet-normalized RGB. Returns per-query
    /// logits `[Q, classes]` for the segment kind, its boundary and its end.
    func callAsFunction(_ frames: MLXArray) -> (intra: MLXArray, inter: MLXArray, end: MLXArray)
    {
        let features = inputProjection(backbone(frames))
        let (count, height, width, d) = (
            features.dim(0), features.dim(1), features.dim(2), features.dim(3)
        )
        // Tokens run frame by frame, then row by row, then column by column —
        // `flatten(2).permute(2, 0, 1)` of the official (B, C, F, H·W) tensor.
        var memory = features.reshaped(1, count * height * width, d)
        let position = self.position(frames: count, height: height, width: width).asType(
            memory.dtype)
        for layer in transformer.encoder.layers {
            memory = layer(memory, position: position)
        }
        let queryPosition = queryEmbedding.weight.expandedDimensions(axis: 0)
        var target = MLXArray.zeros(like: queryPosition)
        for layer in transformer.decoder.layers {
            target = layer(
                target, memory: memory, position: position, queryPosition: queryPosition)
        }
        let decoded = transformer.decoder.norm(target)[0]
        return (intraHead(decoded), interHead(decoded), rangeHead(decoded))
    }

    /// VisTR's 3-D sine position embedding, normalized, with no padding mask:
    /// time, then y, then x, each `d / 3` wide with sine on even and cosine
    /// on odd channels.
    private func position(frames: Int, height: Int, width: Int) -> MLXArray {
        let key = [frames, height, width]
        if let cached = positionCache, cached.key == key { return cached.value }
        let d = configuration.hiddenDimension
        let features = d / 3
        let scale = 2 * Double.pi
        let frequencies = (0..<features).map { index in
            pow(10_000, 2 * Double(index / 2) / Double(features))
        }
        func encode(_ step: Int, of total: Int) -> [Float] {
            let value = Double(step + 1) / (Double(total) + 1e-6) * scale
            return frequencies.enumerated().map { index, frequency in
                let angle = value / frequency
                return Float(index % 2 == 0 ? sin(angle) : cos(angle))
            }
        }
        let time = (0..<frames).map { encode($0, of: frames) }
        let rows = (0..<height).map { encode($0, of: height) }
        let columns = (0..<width).map { encode($0, of: width) }
        var values = [Float]()
        values.reserveCapacity(frames * height * width * features * 3)
        for t in 0..<frames {
            for y in 0..<height {
                for x in 0..<width {
                    values += time[t]
                    values += rows[y]
                    values += columns[x]
                }
            }
        }
        let embedding = MLXArray(values, [1, frames * height * width, features * 3])
        positionCache = (key, embedding)
        return embedding
    }
}
