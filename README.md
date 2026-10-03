# OmniShotCutMLX

[OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut) v1.5 shot
boundary detection in Swift on [MLX](https://github.com/ml-explore/mlx-swift).
It splits a video into shots and the transitions between them — dissolves,
wipes, pushes, slides, zooms, fades, doorways — and says how each one began: a
hard cut, a transition, or a sudden jump within one camera setup.

The port is held to the official PyTorch code's output on three open movies
(see [Testing](#testing)).

## Use

```swift
.package(url: "https://github.com/starkdmi/OmniShotCutMLX", branch: "main")
```

```swift
import OmniShotCutMLX

let detector = try await OmniShotCut.pretrained()   // or OmniShotCut(directory:)
for segment in try await detector.segments(in: videoURL) {
    print(segment.start, segment.end, segment.frames, segment.kind, segment.boundary)
}
```

Segments cover the whole video in order, without gaps. `start` and `end` are
presentation times, `frames` the decoded frame indices as the official code
numbers them. Take keyframes from `kind == .shot`: a dissolve's middle frame is
two pictures at once, a fade's is half black.

The command line tool prints the same:

```bash
make cli
.derivedData/Build/Products/Release/omnishotcut video.mp4
```

`--json` for JSON, `--model` for a local conversion or another Hub id.

## Weights

Converted from [`uva-cv-lab/OmniShotCut_v1.5`](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5)
by `scripts/convert.py`, and always computed in fp32:

| Hub id | Size | Cuts identical to PyTorch |
|---|---|---|
| [`starkdmi/OmniShotCut-v1.5-mlx-fp16`](https://huggingface.co/starkdmi/OmniShotCut-v1.5-mlx-fp16) (default) | 106 MB | 218 of 219 |
| [`starkdmi/OmniShotCut-v1.5-mlx-8bit`](https://huggingface.co/starkdmi/OmniShotCut-v1.5-mlx-8bit) | 67 MB | 214 of 219 |

"Identical" is the same frame and the same labels. Unconverted fp32 weights
match all 219. Rounding weights moves logits — by up to 0.012 for fp16, 0.09
for 8 bits — and that decides near-ties: the fp16 weights' one difference is a
label PyTorch itself prefers by 0.0016 and flips the same way when its weights
are rounded to fp16. Biases and normalization parameters stay fp32 in both
conversions (0.2 MB): rounding the backbone's batch norms alone moved logits
as much as rounding every matrix. In fp32 the port's logits are within 1.6e-4
of PyTorch's in fp64, as close as PyTorch's own fp32 (1.4e-4).

## Build with Xcode, not SwiftPM

`swift build`, `swift run` and `swift test` produce binaries without MLX's
compiled Metal library, and the first kernel fails with `Failed to load the
default metallib` — which looks like a broken model. Build with `xcodebuild`
(the Makefile does), or from an Xcode project that depends on the package.

## How it matches PyTorch

- **Every frame, at the file's own rate**, as the official code reads them.
  Frames stream through one 100-frame window at a time, so an hour of video
  costs one window of memory.
- **Windowing and stitching are `engine.py`'s**, step for step: windows of 100
  frames overlapping by 10, each trusted for its half of the overlap, every
  segment labelled by the cut that begins it.
- **Decoding is ffmpeg's, bit for bit.** The official code decodes with
  `ffmpeg -s 128x96 -pix_fmt rgb24`, and the model is sensitive to it: a float
  bicubic matching 99.7% of ffmpeg's values moved logits more than rounding
  every weight to fp16. `FrameReader` takes the decoder's YUV and runs
  libswscale's own arithmetic on the GPU — its fixed-point bicubic filters,
  14-bit horizontal and 12-bit vertical passes, half-width chroma and integer
  YUV→RGB tables, the stream's matrix and range — and turns rotated video
  before scaling it, as ffmpeg does. Every frame of the three test videos and
  of rotated copies is identical to ffmpeg 9's. Chroma is sited at the centre
  whatever the stream says: ffmpeg 9's scale filter replaces each frame's
  chroma location with its own, unspecified, option.
- **mlx-swift 0.30's `MaxPool2d(padding:)`** pads width and channels of an NHWC
  input, not height and width; the backbone pads by hand.

## Testing

```bash
make videos   # 388 MB of Blender open movies into videos/, checked by SHA-256
make test
```

| Test | Needs | Checks |
|---|---|---|
| `StitchingTests` | nothing | windowing and stitching against `engine.py` |
| `FrameReaderTests` | videos, ffmpeg | every decoded frame, and rotated ones, byte for byte against ffmpeg's |
| `ReferenceTests` | videos, model | segments against the official PyTorch output |

`ReferenceTests` downloads the default model unless `OMNISHOTCUT_MODEL` names
a local conversion or another Hub id; `OMNISHOTCUT_VIDEOS` points both video
tests at another directory. `Fixtures/omnishotcut-reference.json` was written by
`scripts/reference.py`, which runs the official package unmodified but for the
CPU and ffmpeg 8's `-fps_mode`:

| Video | Frames | Cuts | fp32 | fp16 | 8-bit |
|---|---|---|---|---|---|
| Sintel trailer | 1,253 | 28 | 28 | 27 | 25 |
| Big Buck Bunny trailer | 812 | 30 | 30 | 30 | 30 |
| Tears of Steel | 17,620 | 161 | 161 | 161 | 159 |

The videos are © Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/),
[blender.org](https://www.blender.org); they are downloaded, not redistributed.

## Converting

```bash
pip install torch numpy mlx
curl -LO https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/main/OmniShotCut_ckpt.pth
python scripts/convert.py OmniShotCut_ckpt.pth OmniShotCut-v1.5-mlx-fp16 --bits 0
```

## License

MIT, as are OmniShotCut's code and weights. See [LICENSE](LICENSE).

```bibtex
@article{wang2026omnishotcut,
  title={OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer},
  author={Wang, Boyang and Xu, Guangyi and Zhang, Jiahui and Tang, Zhipeng and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2604.24762},
  year={2026}
}
```
