"""
Human detector wrapper using trained D-FINE architecture.
"""

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import sys
from typing import List, Tuple
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import DetectorConfig, get_project_root


@dataclass
class Detection:
    """Represents a single detected human object."""
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in original image pixels
    confidence: float
    class_id: int
    label: str = "person"

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class DFineDetector:
    """Encapsulates the trained D-FINE human detector."""

    def __init__(self, config: DetectorConfig = None, device: str = "cuda:0"):
        self.config = config or DetectorConfig()
        self.device = self._resolve_device(device)
        self.model = self._init_model()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            try:
                torch.zeros(1, device=device)
            except RuntimeError as error:
                print(f"CUDA initialization failed ({error}). Falling back to CPU.")
                return torch.device("cpu")
        return device

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> dict:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Detector checkpoint not found at: {checkpoint_path}")

        if checkpoint.is_dir():
            archive_root = checkpoint / checkpoint.stem
            if not archive_root.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory must contain an extracted archive folder: {archive_root}"
                )
            archive = BytesIO()
            with ZipFile(archive, "w", ZIP_STORED) as zip_file:
                for path in archive_root.rglob("*"):
                    if path.is_file():
                        archive_name = Path(archive_root.name) / path.relative_to(archive_root)
                        zip_file.write(path, archive_name.as_posix())
            archive.seek(0)
            loaded = torch.load(archive, map_location="cpu")
        else:
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if not isinstance(loaded, dict):
            raise TypeError(f"Checkpoint must be a dictionary, got {type(loaded).__name__}.")
        return loaded

    def _init_model(self) -> nn.Module:
        repo_root = get_project_root() / "HUMAN_DETECTION_MODEL" / "HUMAN_DETECTION_MODEL"
        if not repo_root.is_dir():
            raise FileNotFoundError(f"D-FINE repository directory not found at: {repo_root}")

        # Ensure repo root is on sys.path for D-FINE internal imports
        repo_str = str(repo_root.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from src.core import YAMLConfig

        config_path = Path(self.config.config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"D-FINE config file not found at: {config_path}")

        cfg = YAMLConfig(str(config_path))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = self._load_checkpoint(self.config.checkpoint_path)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        cfg.model.load_state_dict(state)

        class ModelDeploy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(
                self, images: torch.Tensor, original_sizes: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                outputs = self.model(images)
                return self.postprocessor(outputs, original_sizes)

        model = ModelDeploy().to(self.device)
        model.eval()
        return model

    def detect(self, bgr_image: np.ndarray, confidence: float = None) -> List[Detection]:
        """
        Runs human detection on a single BGR OpenCV frame.
        
        Args:
            bgr_image: np.ndarray (H, W, 3) BGR image
            confidence: optional confidence threshold override
            
        Returns:
            List of Detection objects filtered for human class
        """
        if bgr_image is None or bgr_image.size == 0:
            return []

        conf_thresh = confidence if confidence is not None else self.config.confidence_threshold
        height, width = bgr_image.shape[:2]

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.to(self.device, non_blocking=True)

        target_h, target_w = self.config.input_size
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        original_size = torch.tensor([[width, height]], device=self.device)

        with torch.inference_mode():
            labels, boxes, scores = self.model(resized, original_size)

        detections: List[Detection] = []
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            confidence_value = float(score.item())
            class_id = int(label.item())

            if confidence_value < conf_thresh:
                continue

            if class_id != self.config.target_class_id:
                continue

            coords = [float(v) for v in box.tolist()]
            # Clamp box to image boundaries
            x1 = max(0.0, min(coords[0], float(width)))
            y1 = max(0.0, min(coords[1], float(height)))
            x2 = max(0.0, min(coords[2], float(width)))
            y2 = max(0.0, min(coords[3], float(height)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    box=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                    confidence=round(confidence_value, 4),
                    class_id=class_id,
                    label=self.config.target_class_name,
                )
            )

        return detections
