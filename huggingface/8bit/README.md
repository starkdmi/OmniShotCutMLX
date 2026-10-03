---
license: mit
library_name: mlx
base_model: uva-cv-lab/OmniShotCut_v1.5
tags:
  - mlx
  - shot-boundary-detection
  - scene-detection
  - video
  - transformer
arxiv: 2604.24762
---

# OmniShotCut v1.5 — MLX, 8-bit

[uva-cv-lab/OmniShotCut_v1.5](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5)
converted for [MLX](https://github.com/ml-explore/mlx) by
[`scripts/convert.py`](https://github.com/starkdmi/OmniShotCutMLX/blob/main/scripts/convert.py),
for the Swift package [OmniShotCutMLX](https://github.com/starkdmi/OmniShotCutMLX):

```swift
import OmniShotCutMLX

let detector = try await OmniShotCut.pretrained("starkdmi/OmniShotCut-v1.5-mlx-8bit")
let segments = try await detector.segments(in: videoURL)
```

- Every linear but the query embedding is quantized to 8 bits in groups of 64;
  everything else is fp16. 67 MB; computed in fp32.
- Key names are the official ones, but `backbone.0.body.` is dropped, each
  attention's packed `in_proj` is split into `q_proj`/`k_proj`/`v_proj`, and
  convolutions are OHWI.

Against the official PyTorch code's 219 cuts on three Blender open movies, the
package with these weights finds 214 on the same frame with the same labels:
it misses one fade-in, relabels two cuts and places two a frame apart. The
[fp16 conversion](https://huggingface.co/starkdmi/OmniShotCut-v1.5-mlx-fp16),
the package's default, finds 218. See the package's README for the test.

## License

MIT, as is the original. Copyright (c) 2026 UVA Computer Vision Lab.

```bibtex
@article{wang2026omnishotcut,
  title={OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer},
  author={Wang, Boyang and Xu, Guangyi and Zhang, Jiahui and Tang, Zhipeng and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2604.24762},
  year={2026}
}
```
