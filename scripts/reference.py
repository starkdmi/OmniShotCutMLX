#!/usr/bin/env python3
"""Write the OmniShotCut reference the Swift port is tested against.

Runs the official package, unmodified, on the CPU in fp32:

    pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git
    scripts/fetch-videos.sh videos
    python scripts/reference.py OmniShotCut_ckpt.pth videos/* \\
        > Tests/OmniShotCutMLXTests/Fixtures/omnishotcut-reference.json

Two things are adapted and nothing else: the official code moves the model
and each window to "cuda", mapped here to the CPU; and it decodes with
`-vsync passthrough`, which ffmpeg 8 removed, replaced by its successor
`-fps_mode passthrough`. `inference(mode="default")` decodes with ffmpeg at
128x96 and runs windows of 100 frames overlapping by 10.
"""
import hashlib
import json
import os
import sys

import torch

_module_to = torch.nn.Module.to
_tensor_to = torch.Tensor.to


def _cpu(args):
    return tuple("cpu" if a == "cuda" else a for a in args)


torch.nn.Module.to = lambda self, *args, **kwargs: _module_to(self, *_cpu(args), **kwargs)
torch.Tensor.to = lambda self, *args, **kwargs: _tensor_to(self, *_cpu(args), **kwargs)

import omnishotcut  # noqa: E402
import omnishotcut.util.misc as misc  # noqa: E402
from omnishotcut.datasets.utils import _video_fps  # noqa: E402

# torchvision would download ImageNet weights only to overwrite them.
misc.is_main_process = lambda: False
import omnishotcut.architecture.backbone as backbone  # noqa: E402

backbone.is_main_process = lambda: False


def _decode_video(path, width, height):
    import ffmpeg
    import numpy as np
    stream, _ = (
        ffmpeg.input(path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}",
                fps_mode="passthrough")
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(stream, np.uint8).reshape(-1, height, width, 3)


omnishotcut._decode_video = _decode_video

INTRA = {"General": 0, "Dissolve": 1, "Wipes": 2, "Push": 3, "Slide": 4, "Zoom": 5, "Fade": 6,
         "Doorway": 7, "Padding": 8}
INTER = {"New_Start": 0, "Hard_Cut": 1, "Transition_Source": 2, "Transition": 3,
         "Sudden_Jump": 4, "Padding": 5}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(checkpoint, videos):
    torch.set_num_threads(os.cpu_count() or 8)
    model = omnishotcut.load(checkpoint)
    out = {}
    for path in videos:
        ranges, intra, inter = model.inference(path, mode="default")
        out[os.path.basename(path)] = {
            "sha256": _sha256(path),
            "fps": _video_fps(path),
            "frames": ranges[-1][1],
            "ranges": ranges,
            "intra": [INTRA[x] for x in intra],
            "inter": [INTER[x] for x in inter],
        }
        print(f"{path}: {len(ranges)} segments", file=sys.stderr)
    json.dump({
        "model": "uva-cv-lab/OmniShotCut_v1.5",
        "generator": "scripts/reference.py (official PyTorch, fp32, CPU, overlap 10)",
        "videos": out,
    }, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
