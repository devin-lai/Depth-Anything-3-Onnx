# Depth Anything 3 ONNX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.16%2B-orange.svg)](https://onnx.ai/)

ONNX-compatible implementation of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for efficient depth estimation inference.

This repository provides scripts to export Depth Anything 3 models to ONNX format and run inference using ONNXRuntime, enabling deployment on a wide range of platforms with minimal dependencies.

**[Get Started in 5 minutes →](GETTING_STARTED.md)**

## Features

- **ONNX Export**: Convert Depth Anything 3 PyTorch models to ONNX format
- **Minimal Dependencies**: Inference requires only ONNXRuntime, OpenCV, and NumPy
- **Multiple Model Sizes**: Support for DA3-Small, DA3-Base, DA3-Large models
- **Dynamic Input Support**: Flexible input resolution handling
- **Easy Integration**: Simple API for depth estimation in your projects

## Performance

ONNX models provide significant performance benefits:
- **Reduced Memory**: No PyTorch runtime overhead
- **Cross-Platform**: Runs on CPU, CUDA, TensorRT, DirectML, Coreml, etc.
- **Optimized**: ONNXRuntime automatically optimizes the graph
- Depth Anything 3 ONNX Performance Report
  - **Model:** `depth-anything/DA3-SMALL`
  - **Platform:** macOS M4 (CPU)
  - **Runtime:** ONNX Runtime with CPUExecutionProvider


| Resolution | Model Size | Mean Time | Std Dev | Min | Max | Median | P95 | P99 | FPS |
|------------|------------|-----------|---------|-----|-----|--------|-----|-----|-----|
| 224x224 | 96.1 MB | 92.44ms | 2.28ms | 88.89ms | 96.30ms | 92.49ms | 96.11ms | 96.26ms | 10.82 |
| 392x392 | 96.1 MB | 279.67ms | 8.04ms | 271.08ms | 298.78ms | 278.13ms | 298.48ms | 298.72ms | 3.58 |
| 504x504 | 96.1 MB | 506.84ms | 14.97ms | 490.04ms | 545.88ms | 504.01ms | 541.25ms | 544.95ms | 1.97 |
| 672x672 | 96.1 MB | 1101.54ms | 15.72ms | 1078.86ms | 1131.54ms | 1097.70ms | 1121.84ms | 1129.60ms | 0.91 |
| 840x840 | 96.1 MB | 2206.96ms | 63.76ms | 2115.01ms | 2365.68ms | 2209.16ms | 2298.87ms | 2352.32ms | 0.45 |

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### For Export Only

If you only need to run inference with pre-exported ONNX models:

```bash
pip install onnxruntime opencv-python pillow imageio numpy matplotlib
```

### For Export

To export PyTorch models to ONNX, you'll need PyTorch:

```bash
pip install torch>=2.0.0 torchvision
pip install -r requirements.txt
```

## Quick Start

### 1. Export Model to ONNX

Export a Depth Anything 3 model from HuggingFace to ONNX format:

```bash
python export_onnx.py \
    --model depth-anything/DA3-SMALL \
    --process-res 504 \
    --output DA3-SMALL-504.onnx
```

By default the exported model supports dynamic batch, height, and width axes, and
the script runs portable ONNX Runtime basic graph optimization.

**Available Models:**
- `depth-anything/DA3-SMALL` (80M params)
- `depth-anything/DA3-BASE` (120M params)
- `depth-anything/DA3-LARGE` (350M params)

**Note:** The first run will download the model from HuggingFace. Models are cached in `~/.cache/huggingface/`.

For mobile converters that prefer a smaller fixed-shape graph, export a static model:

```bash
python export_onnx.py \
    --model depth-anything/DA3-SMALL \
    --process-res 504 \
    --output DA3-SMALL-504-static.onnx \
    --static
```

### 2. Run Inference

Run depth estimation on an image using the exported ONNX model:

```bash
python run_onnx.py \
    --image assets/examples/input.jpg \
    --onnx DA3-SMALL-504.onnx \
    --process-res 504 \
    --export-dir ./output \
    --visualize
```

**Arguments:**
- `--image`: Path to input image
- `--onnx`: Path to ONNX model file
- `--process-res`: Processing resolution (default: 504)
- `--export-dir`: Directory to save depth visualization
- `--visualize`: Display visualization with matplotlib

### 3. Use in Your Code

```python
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load ONNX model
sess = ort.InferenceSession("DA3-SMALL-504.onnx")

# Load and preprocess image
image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (504, 504))
image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Run inference
depth = sess.run(None, {"image": image})[0]

# Depth shape: (1, 1, 504, 504)
print(f"Depth map shape: {depth.shape}")
```

## Model Export Details

The export script creates ONNX models with:
- **Input**: `image` tensor of shape `(batch, 3, height, width)`, dtype `float32`, range `[0, 1]`
- **Output**: `depth` tensor of shape `(batch, 1, height, width)`, dtype `float32`
- **Dynamic Axes**: Batch size, height, and width are dynamic unless `--static` is used
- **Input Size Constraint**: Height and width must be divisible by the DA3 patch size (`14`). `run_onnx.py` handles this through preprocessing.
- **ONNX Opset**: 17 (configurable via `--opset`)
- **Graph Optimization**: ONNX Runtime basic optimization is enabled by default

### Export Options

```bash
python export_onnx.py --help
```

**Key Parameters:**
- `--model`: HuggingFace model ID or local path
- `--process-res`: Fixed square resolution for export (e.g., 504, 672)
- `--output`: Output ONNX file path
- `--opset`: ONNX opset version (default: 17)
- `--static`: Export fixed batch/height/width axes for mobile-friendly conversion
- `--optimize`: ONNX Runtime graph optimization level: `none`, `basic`, `extended`, or `all` (default: `basic`)

## Inference Details

The inference script handles:
1. Image loading and preprocessing
2. Fixed-square resizing to the requested processing resolution
3. ONNX model inference
4. Depth map post-processing and visualization
5. Resizing output back to original image resolution

Dynamic ONNX exports can be used with different `--process-res` values at
inference time. Static exports must be run at the same resolution used during
export.

### Inference Options

```bash
python run_onnx.py --help
```

### Recommended Execution Providers

```python
import onnxruntime as ort

# For CUDA GPU
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

# For CPU
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# For TensorRT (fastest on NVIDIA GPUs)
sess = ort.InferenceSession("model.onnx", providers=["TensorrtExecutionProvider"])
```

## Project Structure

```
Depth-Anything-3-Onnx/
├── src/
│   └── depth_anything_3/       # Source code from Depth Anything 3
├── assets/
│   └── examples/               # Sample images
├── export_onnx.py              # Export PyTorch to ONNX
├── run_onnx.py                 # Run inference with ONNX
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 License
└── README.md                   # This file
```

## Troubleshooting

### Issue: "No module named 'depth_anything_3'"

Make sure you're running scripts from the project root directory where the `src/` folder is located.

### Issue: Model download fails

If you experience network issues downloading from HuggingFace:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Issue: ONNX inference is slow on CPU

Try using a smaller model (DA3-SMALL) or enabling optimizations:
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

## Citation

If you use Depth Anything 3 in your research, please cite:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

## Acknowledgments

- **Depth Anything 3**: [Original Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Depth Anything ONNX**: [Reference Implementation](https://github.com/fabio-sim/Depth-Anything-ONNX) for Depth Anything V1/V2
- Thanks to the Depth Anything team for their amazing work on monocular depth estimation

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The Depth Anything 3 models have varying licenses:
- **Apache 2.0**: DA3-SMALL, DA3-BASE, DA3METRIC-LARGE, DA3MONO-LARGE
- **CC BY-NC 4.0**: DA3-GIANT, DA3-LARGE, DA3NESTED-GIANT-LARGE

Please refer to the [official Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3) for model-specific license information.

## Related Projects

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) - Official PyTorch implementation
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - Previous version
- [Depth Anything ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) - ONNX implementation for V1/V2

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
