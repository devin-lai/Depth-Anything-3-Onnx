#!/usr/bin/env python3
"""
Export Depth Anything 3 to ONNX.

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
    of shape (B, 3, H, W) and outputs depth (B, 1, H, W).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # underlying nn.Module

    def forward(self, image):
        # image: (B, 3, H, W) -> (B, N=1, 3, H, W)
        x = image.unsqueeze(1)
        out = self.model(x, None, None, export_feat_layers=[], infer_gs=False)
        return out["depth"]  # (B, 1, H, W)


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
        help="Square resolution used for the export example input",
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
    parser.add_argument(
        "--static",
        action="store_true",
        help="Export a fixed-shape model instead of dynamic batch/height/width axes",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=("none", "basic", "extended", "all"),
        default="basic",
        help="Run ONNX Runtime graph optimization after export. Use 'basic' for portable ONNX.",
    )
    return parser.parse_args()


def optimize_onnx_model(model_path: str, level: str):
    if level == "none":
        return

    import onnxruntime as ort

    optimization_levels = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    output_path = Path(model_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".opt.tmp")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = optimization_levels[level]
    sess_options.optimized_model_filepath = str(tmp_path)

    ort.InferenceSession(str(output_path), sess_options, providers=["CPUExecutionProvider"])
    os.replace(tmp_path, output_path)
    remove_unused_opset_imports(output_path)


def remove_unused_opset_imports(model_path: Path):
    import onnx

    model = onnx.load(model_path)
    used_domains = {node.domain for node in model.graph.node}
    kept_imports = [opset for opset in model.opset_import if opset.domain in used_domains]
    if len(kept_imports) == len(model.opset_import):
        return

    del model.opset_import[:]
    model.opset_import.extend(kept_imports)
    onnx.save(model, model_path)


def main():
    args = parse_args()

    device = torch.device("cpu")
    print("Loading PyTorch model: {}".format(args.model))
    pt_model = DepthAnything3.from_pretrained(args.model)
    pt_model = pt_model.to(device)
    pt_model.eval()

    wrapper = OnnxWrapper(pt_model)
    wrapper.eval()

    dummy_input = torch.zeros(1, 3, args.process_res, args.process_res, device=device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    dynamic_axes = None
    if not args.static:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
        }

    print("Exporting to ONNX...")
    export_kwargs = {
        "input_names": ["image"],
        "output_names": ["depth"],
        "opset_version": args.opset,
        # PyTorch 2.9+ defaults to the new dynamo exporter. This model still
        # needs the legacy exporter path for reliable dynamic ONNX export.
        "dynamo": False,
    }
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        **export_kwargs,
    )
    if args.optimize != "none":
        print("Optimizing ONNX graph ({})...".format(args.optimize))
        optimize_onnx_model(args.output, args.optimize)
    print("Saved ONNX model to {}".format(args.output))


if __name__ == "__main__":
    main()
