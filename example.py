#!/usr/bin/env python3
"""
Simple example showing how to use Depth Anything 3 ONNX model.

This example demonstrates:
1. Loading an ONNX model
2. Processing an image
3. Running inference
4. Visualizing results

Usage:
    python example.py --image input.jpg --model DA3-SMALL-504.onnx
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

# Add src to path for depth_anything_3 imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from depth_anything_3.utils.visualize import visualize_depth


def load_onnx_model(model_path: str, use_gpu: bool = False):
    """Load ONNX model with specified execution provider."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"Model loaded on: {session.get_providers()}")
    return session


def preprocess_image(image_path: str, target_size: int = 504):
    """
    Load and preprocess image for depth estimation.

    Args:
        image_path: Path to input image
        target_size: Target size for model input (square)

    Returns:
        preprocessed: Image tensor (1, 3, H, W) in range [0, 1]
        original_size: Original (height, width) for resizing output
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size[::-1]  # (height, width)

    # Resize to square
    img = img.resize((target_size, target_size), Image.BILINEAR)

    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0

    # Transpose to CHW format and add batch dimension
    img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, ...]

    return img_tensor, original_size


def run_inference(session, image_tensor):
    """Run ONNX inference."""
    outputs = session.run(None, {"image": image_tensor})
    depth = outputs[0]  # Shape: (1, 1, H, W) or (1, H, W)

    # Remove batch and channel dimensions
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]

    return depth


def postprocess_depth(depth, original_size):
    """Resize depth map to original image size."""
    h, w = original_size
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_resized


def visualize_result(image_path: str, depth: np.ndarray, save_path: str = None):
    """Visualize input image and depth map side by side."""
    # Load original image
    img = Image.open(image_path).convert("RGB")

    # Create depth visualization
    depth_vis = visualize_depth(depth)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(depth_vis)
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 ONNX Example")
    parser.add_argument("--image", type=str, default="assets/examples/input.jpg", help="Input image path")
    parser.add_argument("--model", type=str, default="DA3-SMALL-504.onnx", help="ONNX model path")
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--save", type=str, default=None, help="Save visualization to path")
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Anything 3 ONNX Example")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1/4] Loading ONNX model: {args.model}")
    session = load_onnx_model(args.model, args.use_gpu)

    # 2. Preprocess image
    print(f"[2/4] Preprocessing image: {args.image}")
    image_tensor, original_size = preprocess_image(args.image, args.process_res)
    print(f"      Original size: {original_size}")
    print(f"      Input tensor shape: {image_tensor.shape}")

    # 3. Run inference
    print("[3/4] Running inference...")
    depth = run_inference(session, image_tensor)
    print(f"      Depth shape: {depth.shape}")
    print(f"      Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # 4. Postprocess and visualize
    print("[4/4] Postprocessing and visualizing...")
    depth_final = postprocess_depth(depth, original_size)
    print(f"      Final depth shape: {depth_final.shape}")

    visualize_result(args.image, depth_final, args.save)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
