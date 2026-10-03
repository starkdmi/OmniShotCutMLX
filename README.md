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

"Identical" is the same frame and the same labels. The fp16 weights' other
cut is a frame apart. The 8-bit weights miss the Sintel trailer's opening
fade-in, relabel two of its cuts and place two of Tears of Steel's a frame
apart. Unconverted fp32 weights match all 219, 4-bit ones 197. Computing in
fp16 relabels two cuts and moves two even with the fp16 weights, which is why
compute stays fp32.

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
- **Decoding reproduces ffmpeg.** The official code decodes with
  `ffmpeg -s 128x96 -pix_fmt rgb24`, and the model is sensitive to it: a few
  levels of difference in scaling or colour conversion move cuts in fades,
  where brightness is the signal. `FrameReader` takes the decoder's YUV and
  does what libswscale does — bicubic a = -0.6 stretched over the scale,
  half-width chroma, swscale's integer YUV→RGB tables, the stream's tagged
  matrix — on the GPU. 99.7% of values are identical to ffmpeg 9's, the rest
  within 4 levels.
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
| `FrameReaderTests` | videos, ffmpeg | decoded frames against ffmpeg's |
| `ReferenceTests` | videos, model | segments against the official PyTorch output |

`ReferenceTests` downloads the default model unless `OMNISHOTCUT_MODEL` names
a local conversion or another Hub id; `OMNISHOTCUT_VIDEOS` points both video
tests at another directory. `Fixtures/omnishotcut-reference.json` was written by
`scripts/reference.py`, which runs the official package unmodified but for the
CPU and ffmpeg 8's `-fps_mode`:

| Video | Frames | Cuts | fp32 | fp16 | 8-bit |
|---|---|---|---|---|---|
| Sintel trailer | 1,253 | 28 | 28 | 28 | 25 |
| Big Buck Bunny trailer | 812 | 30 | 30 | 30 | 30 |
| Tears of Steel | 17,620 | 161 | 161 | 160 | 159 |

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
