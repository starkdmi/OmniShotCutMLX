#!/usr/bin/env python3
"""Convert the official OmniShotCut checkpoint to the MLX weights the package loads.

    pip install torch numpy mlx
    curl -LO https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/main/OmniShotCut_ckpt.pth
    python scripts/convert.py OmniShotCut_ckpt.pth OmniShotCut-v1.5-mlx-fp16 --bits 0
    python scripts/convert.py OmniShotCut_ckpt.pth OmniShotCut-v1.5-mlx-8bit

Key names stay the official ones, with two changes: `backbone.0.body.` is
dropped, and each attention's packed `in_proj_weight`/`in_proj_bias` is split
into `q_proj`, `k_proj` and `v_proj` so they quantize like any other linear.
Convolutions go OIHW -> OHWI. With `--bits`, every linear (2-D weight but the
query embedding) is quantized from fp32 in groups of 64, its scales and
biases stored in fp16. Everything else is stored as `--dtype`. The Swift
loader computes in fp32.

Against the official PyTorch code's 219 cuts on the reference videos
(scripts/fetch-videos.sh), cuts found on the same frame with the same labels:
fp32 219, fp16 storage 218 (the other a frame apart), 8 bits 214, 4 bits 197.
"""
import argparse
import json
import os

import mlx.core as mx
import numpy as np
import torch

CONFIG_KEYS = [
    "hidden_dim", "enc_layers", "dec_layers", "nheads", "dim_feedforward", "num_queries",
    "num_intra_relation_classes", "num_inter_relation_classes", "max_process_window_length",
    "process_width", "process_height", "pre_norm",
]


def main(checkpoint, destination, bits=8, group_size=64, dtype="float16"):
    storage = getattr(mx, dtype)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state["args"]
    weights = {}
    for key, value in state["model"].items():
        value = value.float().numpy()
        key = key.replace("backbone.0.body.", "backbone.")
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            base, name = key.rsplit(".", 1)
            kind = name.split("_")[-1]
            third = value.shape[0] // 3
            for index, projection in enumerate("qkv"):
                weights[f"{base}.{projection}_proj.{kind}"] = value[index * third:(index + 1) * third]
            continue
        weights[key] = value

    arrays = {}
    for key, value in weights.items():
        array = mx.array(np.ascontiguousarray(value))
        prefix = key[: -len(".weight")]
        if bits and array.ndim == 2 and key.endswith(".weight") and key != "query_embed.weight":
            quantized, scales, biases = mx.quantize(array, group_size=group_size, bits=bits)
            arrays[key] = quantized
            arrays[f"{prefix}.scales"] = scales.astype(mx.float16)
            arrays[f"{prefix}.biases"] = biases.astype(mx.float16)
        else:
            arrays[key] = array.astype(storage)

    os.makedirs(destination, exist_ok=True)
    mx.save_safetensors(os.path.join(destination, "model.safetensors"), arrays, metadata={"format": "mlx"})
    config = {key: getattr(args, key) for key in CONFIG_KEYS}
    if bits:
        config["quantization"] = {"group_size": group_size, "bits": bits}
    with open(os.path.join(destination, "config.json"), "w") as file:
        json.dump(config, file, indent=1)
    print(f"{len(arrays)} arrays -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("destination")
    parser.add_argument("--bits", type=int, default=8, choices=[0, 4, 8],
                        help="linear weights; 0 keeps them unquantized (default 8)")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                        help="storage of everything not quantized (default float16)")
    options = parser.parse_args()
    main(options.checkpoint, options.destination, options.bits, options.group_size, options.dtype)
