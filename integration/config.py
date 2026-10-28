"""
Configuration module for D-FINE Human Detection and Depth Anything V2 integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def get_project_root() -> Path:
    """Returns the root directory of the Test_1_depth_map project."""
    # This file is in <root>/integration/config.py
    return Path(__file__).resolve().parent.parent


@dataclass
class DetectorConfig:
    """Configuration for the D-FINE Human Detector."""
    config_path: str = field(
        default_factory=lambda: str(
            get_project_root() / "HUMAN_DETECTION_MODEL" / "HUMAN_DETECTION_MODEL" / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom.yml"
        )
    )
    checkpoint_path: str = field(
        default_factory=lambda: str(
            get_project_root() / "HUMAN_DETECTION_MODEL" / "HUMAN_DETECTION_MODEL" / "best.pt"
        )
    )
    confidence_threshold: float = 0.4
    target_class_id: int = 0
    target_class_name: str = "person"
    input_size: Tuple[int, int] = (640, 640)


@dataclass
class DepthConfig:
    """Configuration for Depth Anything V2 Metric Depth."""
    encoder: str = "vits"  # 'vits', 'vitb', 'vitl'
    dataset: str = "vkitti"  # 'vkitti' for outdoor (up to 80m), 'hypersim' for indoor (up to 20m)
    checkpoint_path: str = field(
        default_factory=lambda: str(
            get_project_root() / "checkpoints" / "depth_anything_v2_metric_vkitti_vits.pth"
        )
    )
    max_depth: float = 80.0
    input_size: int = 518  # Must be divisible by 14 (e.g. 392, 518)


@dataclass
class DistanceConfig:
    """Configuration for associating metric depth and 3D Euclidean distance with human detections."""
    # Sampling central sub-region inside bounding box to avoid background intrusion
    crop_ratio_x: float = 0.5  # Central 50% width
    crop_ratio_y: float = 0.5  # Central 50% height
    min_box_area: float = 100.0  # Minimum pixel area to consider
    min_valid_pixels: int = 10  # Minimum number of valid depth pixels required
    min_valid_ratio: float = 0.1  # Minimum fraction of valid pixels in ROI
    min_depth_cutoff: float = 0.1  # Minimum valid physical depth in meters
    max_depth_cutoff: float = 80.0  # Maximum valid physical depth in meters

    # Camera intrinsic parameters for 3D Euclidean distance calculation
    # If fx/fy are None, horizontal field of view (hfov_deg) is used to compute focal length
    hfov_deg: float = 65.0  # Default standard webcam field of view (~60-70 degrees)
    fx: Optional[float] = None  # Focal length x in pixels
    fy: Optional[float] = None  # Focal length y in pixels
    cx: Optional[float] = None  # Principal point x (defaults to frame_width / 2)
    cy: Optional[float] = None  # Principal point y (defaults to frame_height / 2)


@dataclass
class VisualizationConfig:
    """Configuration for two-panel visualization."""
    colormap: int = 2  # cv2.COLORMAP_JET = 2, cv2.COLORMAP_INFERNO = 8
    max_viz_depth: float = 15.0  # Max distance in meters mapped to the colormap scale
    show_fps: bool = True
    show_roi_boxes: bool = False  # If True, draws the inner sampling box for debugging
    box_color: Tuple[int, int, int] = (0, 255, 0)  # BGR green
    text_color: Tuple[int, int, int] = (255, 255, 255)  # BGR white
    text_bg_color: Tuple[int, int, int] = (40, 40, 40)
    font_scale: float = 0.55
    thickness: int = 2


@dataclass
class PipelineConfig:
    """Unified configuration for the full integration pipeline."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    device: str = "cuda:0"  # 'cuda:0' or 'cpu'
    source: str = "0"  # Webcam index (e.g. '0'), video file path, or image file/directory
    output_dir: Optional[str] = "output"
    no_display: bool = False
    save_video: bool = False
    max_frames: Optional[int] = None
    camera_backend: str = "auto"  # 'auto', 'dshow', 'msmf', 'v4l2', 'any'
