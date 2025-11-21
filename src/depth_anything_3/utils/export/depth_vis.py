# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np
import cv2

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth


def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
):
    os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    for idx in range(prediction.depth.shape[0]):
        depth = prediction.depth[idx]

        # Resize depth map back to original image size if available
        target_h, target_w = (
            prediction.orig_hw[idx]
            if prediction.orig_hw is not None and len(prediction.orig_hw) > idx
            else depth.shape
        )
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(
                depth,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_LINEAR,
            )

        depth_vis = visualize_depth(depth).astype(np.uint8)
        save_path = os.path.join(export_dir, f"depth_vis/{idx:04d}.jpg")
        imageio.imwrite(save_path, depth_vis, quality=95)
