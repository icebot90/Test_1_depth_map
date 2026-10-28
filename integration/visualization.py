"""
Side-by-side visualization module for human detection and metric depth estimation.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import VisualizationConfig
from .distance_estimator import DistanceResult
from .human_detector import Detection


class CombinedVisualizer:
    """
    Renders a two-panel side-by-side display:
      Left panel:  Original camera frame with human bounding boxes and depth labels.
      Right panel: Color-mapped metric depth representation with consistent scale.
    """

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

    def colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts a single-channel metric depth map (meters) into a 3-channel BGR image.
        Uses a fixed physical scale (0 to max_viz_depth) so identical colors across
        different frames correspond to identical physical distances.
        """
        h, w = depth_map.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Mask valid finite positive values
        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(valid_mask):
            return colored

        # Clip values to [0, max_viz_depth]
        clipped = np.clip(depth_map, 0.0, self.config.max_viz_depth)
        # Normalize: closer objects = brighter / warmer, farther = darker / cooler
        # Normalize 0..1 then convert to 0..255 uint8
        norm = (clipped / self.config.max_viz_depth * 255.0).astype(np.uint8)

        # Invert so close objects stand out depending on colormap
        # With INFERNO: 0 (dark) -> 255 (yellow/white). We invert so near = bright, far = dark
        norm_inverted = 255 - norm
        colored_valid = cv2.applyColorMap(norm_inverted, self.config.colormap)
        colored[valid_mask] = colored_valid[valid_mask]

        # Draw a small color legend bar on bottom-right of depth panel
        self._draw_depth_legend(colored)

        return colored

    def _draw_depth_legend(self, depth_bgr: np.ndarray) -> None:
        """Draws a metric depth scale legend on the depth map panel."""
        h, w = depth_bgr.shape[:2]
        bar_w = 120
        bar_h = 14
        margin_x = 15
        margin_y = 20

        x_start = w - bar_w - margin_x
        y_start = h - bar_h - margin_y

        if x_start < 0 or y_start < 0:
            return

        # Generate gradient 0..255 (near to far)
        gradient = np.linspace(255, 0, bar_w, dtype=np.uint8).reshape(1, bar_w)
        gradient_bgr = cv2.applyColorMap(gradient, self.config.colormap)
        gradient_bar = cv2.resize(gradient_bgr, (bar_w, bar_h))

        depth_bgr[y_start : y_start + bar_h, x_start : x_start + bar_w] = gradient_bar
        cv2.rectangle(depth_bgr, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (255, 255, 255), 1)

        # Scale labels
        cv2.putText(
            depth_bgr,
            "Near 0m",
            (x_start, y_start - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        max_label = f">{int(self.config.max_viz_depth)}m"
        cv2.putText(
            depth_bgr,
            max_label,
            (x_start + bar_w - 30, y_start - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        distance_results: List[DistanceResult],
    ) -> np.ndarray:
        """
        Draws bounding boxes and person depth annotations on the camera frame.
        """
        annotated = frame.copy()

        for idx, (det, dist) in enumerate(zip(detections, distance_results), start=1):
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # Draw outer bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                self.config.box_color,
                self.config.thickness,
            )

            # Optionally draw the central sampling ROI box for debugging
            if self.config.show_roi_boxes and dist.is_valid:
                rx1, ry1, rx2, ry2 = dist.roi_coords
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 1)

            # Label text: Display true 3D straight-line distance from camera
            if dist.is_valid and dist.distance_m is not None:
                dist_str = f"Dist: {dist.distance_m:.2f} m"
                if dist.depth_m is not None and abs(dist.distance_m - dist.depth_m) >= 0.05:
                    dist_str += f" (Z: {dist.depth_m:.2f}m)"
            elif dist.is_valid and dist.depth_m is not None:
                dist_str = f"Dist: {dist.depth_m:.2f} m"
            else:
                dist_str = "Dist: N/A"

            label = f"Person {idx} | Conf: {det.confidence:.2f} | {dist_str}"

            # Calculate label box size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.config.font_scale
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label above bounding box if space allows
            label_y = y1 - 8 if y1 - text_h - 12 > 0 else y1 + text_h + 10
            bg_y1 = label_y - text_h - 4
            bg_y2 = label_y + baseline + 2
            bg_x1 = max(0, x1)
            bg_x2 = min(annotated.shape[1], bg_x1 + text_w + 8)

            cv2.rectangle(
                annotated,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                self.config.text_bg_color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (bg_x1 + 4, label_y),
                font,
                font_scale,
                self.config.text_color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def render_side_by_side(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        detections: List[Detection],
        distance_results: List[DistanceResult],
        perf_stats: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Creates the complete side-by-side visualization.
        
        Args:
            frame: Original BGR camera frame (H, W, 3)
            depth_map: Metric depth map (H, W) in meters
            detections: List of Detection objects
            distance_results: List of DistanceResult objects
            perf_stats: Dictionary containing latencies and FPS
            
        Returns:
            np.ndarray: Combined side-by-side image of shape (H, 2*W, 3)
        """
        h, w = frame.shape[:2]

        # Draw left panel: annotated original frame
        left_panel = self.draw_annotated_frame(frame, detections, distance_results)

        # Draw right panel: color-coded depth map
        right_panel = self.colorize_depth(depth_map)

        # Ensure right panel matches left panel dimensions exactly
        if right_panel.shape[:2] != (h, w):
            right_panel = cv2.resize(right_panel, (w, h), interpolation=cv2.INTER_NEAREST)

        # Add panel titles
        self._draw_panel_title(left_panel, "Camera Feed & D-FINE Detection")
        self._draw_panel_title(right_panel, "Depth Anything V2 Metric Depth")

        # Combine horizontally
        combined = np.hstack([left_panel, right_panel])

        # Draw performance stats overlay on top if enabled
        if self.config.show_fps and perf_stats:
            self._draw_perf_overlay(combined, perf_stats)

        return combined

    @staticmethod
    def _draw_panel_title(image: np.ndarray, title: str) -> None:
        """Draws a clean top header bar on an image panel."""
        bar_h = 28
        cv2.rectangle(image, (0, 0), (image.shape[1], bar_h), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            image,
            title,
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_perf_overlay(image: np.ndarray, stats: Dict[str, float]) -> None:
        """Draws performance timings overlay at the bottom left of the combined image."""
        det_ms = stats.get("det_ms", 0.0)
        depth_ms = stats.get("depth_ms", 0.0)
        total_ms = stats.get("total_ms", 0.0)
        fps = stats.get("fps", 0.0)
        device = stats.get("device", "N/A")

        text = (
            f"Device: {device} | D-FINE: {det_ms:.1f}ms | Depth: {depth_ms:.1f}ms | "
            f"Total: {total_ms:.1f}ms | FPS: {fps:.1f}"
        )

        h, w = image.shape[:2]
        bar_h = 24
        cv2.rectangle(image, (0, h - bar_h), (w, h), (15, 15, 15), cv2.FILLED)
        cv2.putText(
            image,
            text,
            (10, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )
