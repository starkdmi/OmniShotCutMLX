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

# OmniShotCut v1.5 — MLX, fp16

[uva-cv-lab/OmniShotCut_v1.5](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5)
converted for [MLX](https://github.com/ml-explore/mlx) by
[`scripts/convert.py --bits 0`](https://github.com/starkdmi/OmniShotCutMLX/blob/main/scripts/convert.py),
for the Swift package [OmniShotCutMLX](https://github.com/starkdmi/OmniShotCutMLX):

```swift
import OmniShotCutMLX

let detector = try await OmniShotCut.pretrained()   // this model
let segments = try await detector.segments(in: videoURL)
```

- Stored in fp16, 106 MB; computed in fp32.
- Key names are the official ones, but `backbone.0.body.` is dropped, each
  attention's packed `in_proj` is split into `q_proj`/`k_proj`/`v_proj`, and
  convolutions are OHWI.

Against the official PyTorch code's 219 cuts on three Blender open movies, the
package with these weights finds 218 on the same frame with the same labels and
the last a frame apart. The smaller
[8-bit conversion](https://huggingface.co/starkdmi/OmniShotCut-v1.5-mlx-8bit)
finds 214. See the package's README for the test.

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
