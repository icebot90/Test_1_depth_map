"""
Metric depth estimation wrapper using Depth Anything V2.
"""

from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import DepthConfig, get_project_root


class DepthAnythingV2Estimator:
    """Wrapper around Depth Anything V2 for metric depth estimation."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, config: DepthConfig = None, device: str = "cuda:0"):
        self.config = config or DepthConfig()
        self.device = self._resolve_device(device)
        self.model = self._init_model()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        dev = torch.device(requested)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                print("CUDA requested for depth model but not available. Falling back to CPU.")
                return "cpu"
            try:
                torch.zeros(1, device=dev)
                return requested
            except Exception as e:
                print(f"CUDA initialization failed for depth model ({e}). Falling back to CPU.")
                return "cpu"
        return "cpu"

    def _init_model(self) -> torch.nn.Module:
        metric_root = get_project_root() / "Depth-Anything-V2" / "metric_depth"
        if not metric_root.is_dir():
            raise FileNotFoundError(f"Depth Anything V2 metric_depth directory not found at: {metric_root}")

        metric_str = str(metric_root.resolve())
        if metric_str not in sys.path:
            sys.path.insert(0, metric_str)

        from depth_anything_v2.dpt import DepthAnythingV2

        encoder = self.config.encoder.lower()
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported encoder '{encoder}'. Supported encoders: {list(self.MODEL_CONFIGS.keys())}"
            )

        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found at: {checkpoint_path}.\n"
                f"Please ensure the official metric checkpoint has been downloaded.\n"
                f"Official download URL for {self.config.dataset.upper()} ({encoder}):\n"
                f"  Outdoor (Virtual KITTI 2): https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true\n"
                f"  Indoor (Hypersim): https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
            )

        model = DepthAnythingV2(**{**self.MODEL_CONFIGS[encoder], "max_depth": self.config.max_depth})
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def estimate_depth(self, bgr_image: np.ndarray, input_size: Optional[int] = None) -> np.ndarray:
        """
        Runs metric depth estimation on a single BGR OpenCV frame.
        
        Args:
            bgr_image: np.ndarray (H, W, 3) in BGR format
            input_size: optional input resolution override (must be divisible by 14)
            
        Returns:
            np.ndarray: (H, W) float32 metric depth map in meters
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty or None")

        size = input_size or self.config.input_size
        # Ensure size is a multiple of 14 for ViT patch tokenization
        size = int(round(size / 14) * 14)

        with torch.inference_mode():
            depth = self.model.infer_image(bgr_image, input_size=size)

        return depth.astype(np.float32)
