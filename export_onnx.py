#!/usr/bin/env python3
"""
Export Depth Anything 3 to ONNX (fixed square resolution).

Example:
    python export_onnx.py \
        --model depth-anything/DA3-SMALL \
        --process-res 504 \
        --output DA3-SMALL-504.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure local package import works when running from repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from depth_anything_3.api import DepthAnything3


class OnnxWrapper(torch.nn.Module):
    """
    Thin wrapper around DepthAnything3 that accepts a single image tensor
    of shape (1, 3, H, W) and outputs depth (1, 1, H, W).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # underlying nn.Module

    def forward(self, image):
        # image: (1, 3, H, W) -> (B=1, N=1, 3, H, W)
        x = image.unsqueeze(0)
        out = self.model(x, None, None, export_feat_layers=[], infer_gs=False)
        return out["depth"].squeeze(0)  # (1, H, W)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-SMALL",
        help="Model name or path",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Fixed square resolution (matches run_depth_inference.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="DA3-SMALL-504.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu")
    print("Loading PyTorch model: {}".format(args.model))
    pt_model = DepthAnything3.from_pretrained(args.model)
    pt_model = pt_model.to(device)
    pt_model.eval()

    wrapper = OnnxWrapper(pt_model)

    dummy_input = torch.zeros(1, 3, args.process_res, args.process_res, device=device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=["image"],
        output_names=["depth"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print("Saved ONNX model to {}".format(args.output))


if __name__ == "__main__":
    main()
