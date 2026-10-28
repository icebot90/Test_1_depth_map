"""
Distance estimator associating metric depth maps with human bounding boxes
and computing true 3D Euclidean distance (range) from the camera.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .config import DistanceConfig
from .human_detector import Detection


@dataclass
class DistanceResult:
    """Estimated depth and 3D Euclidean distance result for a single person detection."""
    depth_m: Optional[float]           # Perpendicular optical depth along camera Z axis (m)
    distance_m: Optional[float]        # True 3D Euclidean range from camera lens sqrt(X^2 + Y^2 + Z^2) (m)
    is_valid: bool
    status_message: str
    valid_pixels: int
    total_roi_pixels: int
    roi_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2) of sampled central region
    coords_3d: Optional[Tuple[float, float, float]] = None  # (X, Y, Z) in meters relative to camera


class PersonDistanceEstimator:
    """
    Associates metric depth with detected persons by sampling a central ROI inside
    the person's bounding box and computing:
      1. Median optical depth Z (meters along camera axis).
      2. 3D spatial position (X, Y, Z) via pinhole camera back-projection.
      3. True 3D Euclidean straight-line distance from camera to person: sqrt(X^2 + Y^2 + Z^2).
    """

    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()

    def get_camera_intrinsics(self, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Resolves focal lengths (fx, fy) and principal center (cx, cy).
        """
        cx = self.config.cx if self.config.cx is not None else width / 2.0
        cy = self.config.cy if self.config.cy is not None else height / 2.0

        if self.config.fx is not None and self.config.fy is not None:
            return self.config.fx, self.config.fy, cx, cy

        # Derive focal length from horizontal FOV: fx = (w / 2) / tan(hfov / 2)
        hfov_rad = math.radians(self.config.hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx  # Assume square pixels by default
        return fx, fy, cx, cy

    def estimate_person_depth(
        self,
        detection: Detection,
        depth_map: np.ndarray,
    ) -> DistanceResult:
        """
        Estimates the metric depth and 3D distance of a detected person from a metric depth map.

        Args:
            detection: Detection object containing bounding box (x1, y1, x2, y2)
            depth_map: 2D numpy array (H, W) of metric depth in meters

        Returns:
            DistanceResult with estimated depth Z, 3D Euclidean distance, and 3D camera coordinates.
        """
        if depth_map is None or depth_map.ndim != 2:
            return DistanceResult(
                depth_m=None,
                distance_m=None,
                is_valid=False,
                status_message="Invalid or missing depth map",
                valid_pixels=0,
                total_roi_pixels=0,
                roi_coords=(0, 0, 0, 0),
                coords_3d=None,
            )

        h, w = depth_map.shape[:2]

        # 1-3. Convert coordinates to integers and clip to image boundaries
        x1 = max(0, min(int(np.floor(detection.x1)), w - 1))
        y1 = max(0, min(int(np.floor(detection.y1)), h - 1))
        x2 = max(0, min(int(np.ceil(detection.x2)), w))
        y2 = max(0, min(int(np.ceil(detection.y2)), h))

        box_w = x2 - x1
        box_h = y2 - y1

        # 4. Reject invalid or too-small bounding boxes
        if box_w <= 0 or box_h <= 0 or (box_w * box_h) < self.config.min_box_area:
            return DistanceResult(
                depth_m=None,
                distance_m=None,
                is_valid=False,
                status_message="Bounding box too small or out of bounds",
                valid_pixels=0,
                total_roi_pixels=0,
                roi_coords=(x1, y1, x2, y2),
                coords_3d=None,
            )

        # 5-6. Central region sampling to reduce background intrusion
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0
        roi_w = max(1.0, box_w * self.config.crop_ratio_x)
        roi_h = max(1.0, box_h * self.config.crop_ratio_y)

        rx1 = max(0, min(int(np.floor(cx_box - roi_w / 2.0)), w - 1))
        rx2 = max(rx1 + 1, min(int(np.ceil(cx_box + roi_w / 2.0)), w))
        ry1 = max(0, min(int(np.floor(cy_box - roi_h / 2.0)), h - 1))
        ry2 = max(ry1 + 1, min(int(np.ceil(cy_box + roi_h / 2.0)), h))

        depth_roi = depth_map[ry1:ry2, rx1:rx2]
        total_roi_pixels = depth_roi.size

        if total_roi_pixels == 0:
            return DistanceResult(
                depth_m=None,
                distance_m=None,
                is_valid=False,
                status_message="Empty depth ROI",
                valid_pixels=0,
                total_roi_pixels=0,
                roi_coords=(rx1, ry1, rx2, ry2),
                coords_3d=None,
            )

        # 7. Filter non-finite (NaN, inf) and out-of-range physical values
        finite_mask = np.isfinite(depth_roi)
        range_mask = (depth_roi >= self.config.min_depth_cutoff) & (
            depth_roi <= self.config.max_depth_cutoff
        )
        valid_mask = finite_mask & range_mask

        valid_pixels = int(np.count_nonzero(valid_mask))
        valid_ratio = valid_pixels / float(total_roi_pixels)

        # 8. Apply configurable validity thresholds
        if valid_pixels < self.config.min_valid_pixels:
            return DistanceResult(
                depth_m=None,
                distance_m=None,
                is_valid=False,
                status_message=f"Insufficient valid pixels ({valid_pixels} < {self.config.min_valid_pixels})",
                valid_pixels=valid_pixels,
                total_roi_pixels=total_roi_pixels,
                roi_coords=(rx1, ry1, rx2, ry2),
                coords_3d=None,
            )

        if valid_ratio < self.config.min_valid_ratio:
            return DistanceResult(
                depth_m=None,
                distance_m=None,
                is_valid=False,
                status_message=f"Valid pixel ratio too low ({valid_ratio:.1%} < {self.config.min_valid_ratio:.1%})",
                valid_pixels=valid_pixels,
                total_roi_pixels=total_roi_pixels,
                roi_coords=(rx1, ry1, rx2, ry2),
                coords_3d=None,
            )

        # 9. Compute median of valid depth values (optical depth Z)
        valid_depth_values = depth_roi[valid_mask]
        z_depth = float(np.median(valid_depth_values))

        # 10. Compute 3D Camera Coordinates (X, Y, Z) and Euclidean Distance (Range)
        # Using pinhole camera projection:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Range = sqrt(X^2 + Y^2 + Z^2)
        fx, fy, cam_cx, cam_cy = self.get_camera_intrinsics(w, h)
        x_m = (cx_box - cam_cx) * z_depth / fx
        y_m = (cy_box - cam_cy) * z_depth / fy
        euclidean_dist = math.sqrt(x_m * x_m + y_m * y_m + z_depth * z_depth)

        return DistanceResult(
            depth_m=round(z_depth, 2),
            distance_m=round(euclidean_dist, 2),
            is_valid=True,
            status_message="OK",
            valid_pixels=valid_pixels,
            total_roi_pixels=total_roi_pixels,
            roi_coords=(rx1, ry1, rx2, ry2),
            coords_3d=(round(x_m, 2), round(y_m, 2), round(z_depth, 2)),
        )
