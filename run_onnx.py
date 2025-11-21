#!/usr/bin/env python3
"""
Run Depth Anything 3 using an ONNX model.

Mirrors the PyTorch pipeline:
    - fixed-square preprocessing (default 504)
    - visualization with matplotlib (optional)
    - depth_vis export resized back to the original image size

Example:
    python run_onnx.py --image input.jpg --visualize --export-dir ./output --onnx DA3-SMALL-504.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

# Ensure local package import works when running from repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.visualize import visualize_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 (ONNX)")
    parser.add_argument(
        "--image",
        type=str,
        default="assets/examples/SOH/000.png",
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="DA3-SMALL-504.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Fixed square processing resolution",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Directory to export depth_vis images",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization (requires display)",
    )
    return parser.parse_args()


def get_orig_hw(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as im:
        w, h = im.size
    return h, w


def preprocess_image(image_path: str, process_res: int):
    ip = InputProcessor()
    imgs_cpu, _, _ = ip(
        image=[image_path],
        process_res=process_res,
        process_res_method="fixed_square",
        num_workers=1,
        sequential=True,
    )
    # imgs_cpu: (1, N=1, 3, H, W); take first image -> (3, H, W)
    img = imgs_cpu[0]
    orig_hw = get_orig_hw(image_path)
    return img, orig_hw


def run_onnx(model_path: str, image_path: str, process_res: int):
    print("Loading ONNX model: {}".format(model_path))
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    img, orig_hw = preprocess_image(image_path, process_res)
    input_image = img.unsqueeze(0).numpy()  # (1, 3, H, W)

    print("Running ONNX inference on: {}".format(image_path))
    outputs = sess.run(None, {"image": input_image})
    depth = outputs[0]  # expect (1, 1, H, W) or (1, H, W)
    depth = np.array(depth)
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]

    # Resize depth back to original resolution
    target_h, target_w = orig_hw
    if depth.shape != (target_h, target_w):
        depth = cv2.resize(depth, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)

    return depth, orig_hw


def save_depth_vis(depth: np.ndarray, export_dir: str, index: int = 0):
    os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    depth_vis = visualize_depth(depth).astype(np.uint8)
    save_path = os.path.join(export_dir, f"depth_vis/{index:04d}.jpg")
    import imageio

    imageio.imwrite(save_path, depth_vis, quality=95)
    return save_path


def main():
    args = parse_args()

    depth, orig_hw = run_onnx(args.onnx, args.image, args.process_res)
    print("✓ Inference complete. Depth shape (original scale): {}".format(depth.shape))

    if args.export_dir:
        saved = save_depth_vis(depth, args.export_dir, index=0)
        print("  - Saved depth visualization to: {}".format(saved))

    if args.visualize:
        print("\nGenerating visualization...")
        depth_vis = visualize_depth(depth)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(depth_vis)
        ax.set_title("Depth")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        print("✓ Visualization complete!")


if __name__ == "__main__":
    main()

