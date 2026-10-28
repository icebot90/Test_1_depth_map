"""
Combined Inference Pipeline for D-FINE Human Detection and Depth Anything V2 Metric Depth.
Supports Webcam, Video File, and Image Input.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import (
    DepthConfig,
    DetectorConfig,
    DistanceConfig,
    PipelineConfig,
    VisualizationConfig,
    get_project_root,
)
from .depth_estimator import DepthAnythingV2Estimator
from .distance_estimator import DistanceResult, PersonDistanceEstimator
from .human_detector import Detection, DFineDetector
from .visualization import CombinedVisualizer


class CombinedInferencePipeline:
    """
    Coordinates simultaneous human detection, metric depth estimation,
    depth-to-person association, and synchronized side-by-side visualization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device_str = self.config.device
        self.use_cuda = "cuda" in self.device_str and torch.cuda.is_available()

        print(f"Initializing D-FINE Detector on {self.device_str}...")
        self.detector = DFineDetector(self.config.detector, device=self.device_str)

        print(f"Initializing Depth Anything V2 Estimator on {self.device_str}...")
        self.depth_estimator = DepthAnythingV2Estimator(self.config.depth, device=self.device_str)

        self.distance_estimator = PersonDistanceEstimator(self.config.distance)
        self.visualizer = CombinedVisualizer(self.config.viz)

        print("Combined pipeline initialized successfully.")

    def _sync(self) -> None:
        """Synchronizes GPU execution when CUDA is enabled for accurate timing."""
        if self.use_cuda:
            torch.cuda.synchronize()

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Detection], List[DistanceResult], Dict[str, float]]:
        """
        Processes a single BGR camera frame through detection, depth, and association.

        Args:
            frame: np.ndarray (H, W, 3) BGR image

        Returns:
            Tuple of:
              - combined_viz: side-by-side annotated visualization image (H, 2*W, 3)
              - depth_map: raw float32 metric depth map (H, W) in meters
              - detections: list of Detection objects
              - distance_results: list of DistanceResult objects
              - perf_stats: dictionary of timing metrics (ms) and FPS
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        total_start = time.perf_counter()

        # 1. Run D-FINE Human Detection
        self._sync()
        det_start = time.perf_counter()
        detections = self.detector.detect(frame)
        self._sync()
        det_ms = (time.perf_counter() - det_start) * 1000.0

        # 2. Run Depth Anything V2 Metric Depth
        self._sync()
        depth_start = time.perf_counter()
        depth_map = self.depth_estimator.estimate_depth(frame)
        self._sync()
        depth_ms = (time.perf_counter() - depth_start) * 1000.0

        # 3. Associate Depth with Each Person Detection
        assoc_start = time.perf_counter()
        distance_results = [
            self.distance_estimator.estimate_person_depth(det, depth_map)
            for det in detections
        ]
        assoc_ms = (time.perf_counter() - assoc_start) * 1000.0

        # 4. Measure Total Time
        self._sync()
        total_ms = (time.perf_counter() - total_start) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        perf_stats = {
            "det_ms": det_ms,
            "depth_ms": depth_ms,
            "assoc_ms": assoc_ms,
            "total_ms": total_ms,
            "fps": fps,
            "device": self.device_str,
        }

        # 5. Side-by-Side Visualization
        combined_viz = self.visualizer.render_side_by_side(
            frame=frame,
            depth_map=depth_map,
            detections=detections,
            distance_results=distance_results,
            perf_stats=perf_stats,
        )

        return combined_viz, depth_map, detections, distance_results, perf_stats


def run_pipeline(config: PipelineConfig) -> None:
    """Executes the pipeline on the configured source (webcam, video, or image)."""
    pipeline = CombinedInferencePipeline(config)
    source = config.source

    output_dir = Path(config.output_dir) if config.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if source is a single image or image directory
    source_path = Path(source)
    is_image = source_path.is_file() and source_path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]

    if is_image:
        _process_single_image(pipeline, source_path, config, output_dir)
        return

    # Check if source is an image folder
    if source_path.is_dir():
        _process_image_folder(pipeline, source_path, config, output_dir)
        return

    # Otherwise treat as camera index or video file
    _process_video_stream(pipeline, source, config, output_dir)


def is_display_available() -> bool:
    """Checks if a graphical desktop/display is available for cv2.imshow."""
    import os
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _process_single_image(
    pipeline: CombinedInferencePipeline,
    image_path: Path,
    config: PipelineConfig,
    output_dir: Optional[Path],
) -> None:
    """Processes a single still image."""
    print(f"Reading image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    combined, depth, detections, distances, stats = pipeline.process_frame(frame)

    print(f"\n--- Inference Results for {image_path.name} ---")
    print(f"Image Dimensions: {frame.shape[1]}x{frame.shape[0]}")
    print(f"D-FINE Inference Time: {stats['det_ms']:.2f} ms")
    print(f"Depth Inference Time:  {stats['depth_ms']:.2f} ms")
    print(f"Total Frame Time:      {stats['total_ms']:.2f} ms ({stats['fps']:.1f} FPS)")
    print(f"Persons Detected:      {len(detections)}")

    for i, (det, dist) in enumerate(zip(detections, distances), start=1):
        if dist.is_valid and dist.distance_m is not None:
            dist_label = f"{dist.distance_m:.2f} m (Z-depth: {dist.depth_m:.2f} m)"
            if dist.coords_3d is not None:
                x_3d, y_3d, z_3d = dist.coords_3d
                dist_label += f", 3D=[X:{x_3d:+.2f}m, Y:{y_3d:+.2f}m, Z:{z_3d:.2f}m]"
        elif dist.is_valid and dist.depth_m is not None:
            dist_label = f"{dist.depth_m:.2f} m"
        else:
            dist_label = f"N/A ({dist.status_message})"

        print(
            f"  Person {i}: Conf={det.confidence:.2f}, Box=[{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}], "
            f"Distance={dist_label}"
        )

    if output_dir:
        out_path = output_dir / f"result_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        # Also save raw depth map as .npy for exact numerical validation
        depth_out_path = output_dir / f"depth_{image_path.stem}.npy"
        np.save(str(depth_out_path), depth)
        print(f"Saved annotated side-by-side visualization to: {out_path}")
        print(f"Saved metric depth map (numpy array) to:        {depth_out_path}")

    if not config.no_display:
        if not is_display_available():
            print("Notice: No graphical display detected (DISPLAY is unset on Linux). Skipping preview window.")
        else:
            try:
                cv2.imshow("D-FINE + Depth Anything V2 Integration", combined)
                print("\nPress any key in preview window to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"Notice: GUI preview failed ({e}).")


def _process_image_folder(
    pipeline: CombinedInferencePipeline,
    folder_path: Path,
    config: PipelineConfig,
    output_dir: Optional[Path],
) -> None:
    """Processes all images in a folder."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(ext))
    image_files.sort()

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    print(f"Found {len(image_files)} images in {folder_path}")
    for img_path in image_files:
        _process_single_image(pipeline, img_path, config, output_dir)


def get_csi_pipeline_string(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Returns a GStreamer pipeline string for NVIDIA Jetson onboard CSI cameras."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def open_video_capture(source: str, backend_name: str = "auto") -> Tuple[cv2.VideoCapture, bool, str]:
    """
    Opens a video capture from a camera index, device node (/dev/videoX), CSI stream,
    GStreamer pipeline, or video file with robust cross-platform backend selection.

    Compatible with:
      - Linux / Ubuntu (V4L2, GStreamer, CSI on Jetson)
      - Windows (DirectShow, MSMF)
      - macOS (AVFoundation)
    """
    import re
    source_str = str(source).strip()

    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
        "any": cv2.CAP_ANY,
    }

    # 1. Jetson CSI camera shortcut: csi://0, csi://1, or csi
    if source_str.lower().startswith("csi://") or source_str.lower() == "csi":
        sensor_id = 0
        if source_str.lower().startswith("csi://"):
            try:
                sensor_id = int(source_str.split("://")[1])
            except ValueError:
                sensor_id = 0
        pipeline = get_csi_pipeline_string(sensor_id=sensor_id)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cap, True, f"NVIDIA Jetson CSI Camera (sensor-id={sensor_id})"

    # 2. Raw GStreamer pipeline string
    if "!" in source_str or "appsink" in source_str:
        cap = cv2.VideoCapture(source_str, cv2.CAP_GSTREAMER)
        return cap, True, "GStreamer pipeline stream"

    # 3. Linux device path (/dev/video0, /dev/video1, ...)
    dev_match = re.match(r"^/dev/video(\d+)$", source_str)
    if dev_match:
        cam_index = int(dev_match.group(1))
        is_live = True
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, is_live, f"V4L2 Camera ({source_str})"
        cap = cv2.VideoCapture(source_str)
        return cap, is_live, f"Camera ({source_str})"

    # 4. Integer camera index
    try:
        cam_index = int(source_str)
        is_live = True

        # Explicit backend requested
        if backend_name.lower() in backend_map and backend_name.lower() != "auto":
            chosen_backend = backend_map[backend_name.lower()]
            cap = cv2.VideoCapture(cam_index, chosen_backend)
            if cap.isOpened():
                return cap, is_live, f"Webcam (index {cam_index}, backend {backend_name.upper()})"

        # Auto backend selection based on OS
        if sys.platform.startswith("linux"):
            # Linux: try V4L2 first
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, is_live, f"Webcam (index {cam_index}, backend V4L2)"
                cap.release()
            # Fallback to default
            cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
            return cap, is_live, f"Webcam (index {cam_index}, backend Default)"

        elif sys.platform.startswith("win"):
            # Windows: try DirectShow first to avoid MSMF format negotiation error
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, is_live, f"Webcam (index {cam_index}, backend DirectShow)"
                cap.release()

            # Fallback to MSMF / default
            cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, is_live, f"Webcam (index {cam_index}, backend MSMF)"
                cap.release()

            cap = cv2.VideoCapture(cam_index)
            return cap, is_live, f"Webcam (index {cam_index}, backend Default)"

        else:
            cap = cv2.VideoCapture(cam_index)
            return cap, is_live, f"Webcam (index {cam_index})"

    except ValueError:
        # Source is a video file or stream URL (RTSP/HTTP/MP4/AVI)
        cap = cv2.VideoCapture(source_str)
        is_live = False
        return cap, is_live, f"Video stream/file ({source_str})"


def _process_video_stream(
    pipeline: CombinedInferencePipeline,
    source: str,
    config: PipelineConfig,
    output_dir: Optional[Path],
) -> None:
    """Processes live camera feed or video file."""
    cap, is_live, source_desc = open_video_capture(source, config.camera_backend)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream or camera from source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Opened {source_desc} | Resolution: {w}x{h} | Nominal FPS: {fps:.1f}")

    writer = None
    if config.save_video and output_dir:
        out_video_path = output_dir / "output_combined.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Combined frame has width 2*w
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (2 * w, h))
        print(f"Saving output video to: {out_video_path}")

    frame_idx = 0
    total_det_ms = 0.0
    total_depth_ms = 0.0
    total_time_ms = 0.0
    consecutive_drops = 0
    max_consecutive_drops = 15

    gui_active = not config.no_display
    if gui_active and not is_display_available():
        print("Notice: No graphical display detected (DISPLAY is unset on Linux). Running in headless mode.")
        gui_active = False

    if gui_active:
        print("Live preview active. Press 'q' or ESC in preview window to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_live:
                    consecutive_drops += 1
                    if consecutive_drops <= max_consecutive_drops:
                        time.sleep(0.01)
                        continue
                    print(f"Warning: Failed to read {max_consecutive_drops} consecutive frames from camera.")
                break

            consecutive_drops = 0

            combined, depth, detections, distances, stats = pipeline.process_frame(frame)

            total_det_ms += stats["det_ms"]
            total_depth_ms += stats["depth_ms"]
            total_time_ms += stats["total_ms"]
            frame_idx += 1

            if writer:
                writer.write(combined)

            if config.max_frames and frame_idx >= config.max_frames:
                print(f"Reached configured limit of {config.max_frames} frame(s).")
                break

            if gui_active:
                try:
                    cv2.imshow("D-FINE + Depth Anything V2 Integration", combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:  # 'q' or ESC
                        print("User requested exit.")
                        break
                except cv2.error as e:
                    print(f"Notice: GUI window error ({e}). Switching to headless mode.")
                    gui_active = False
    finally:
        cap.release()
        if writer:
            writer.release()
        if gui_active:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if frame_idx > 0:
        avg_det = total_det_ms / frame_idx
        avg_depth = total_depth_ms / frame_idx
        avg_total = total_time_ms / frame_idx
        avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        print("\n=== Processing Summary ===")
        print(f"Total Frames:          {frame_idx}")
        print(f"Avg D-FINE Latency:    {avg_det:.1f} ms")
        print(f"Avg Depth Latency:     {avg_depth:.1f} ms")
        print(f"Avg Total Frame Time:  {avg_total:.1f} ms")
        print(f"Average Application FPS: {avg_fps:.1f}")


def parse_args() -> PipelineConfig:
    """Parses command-line arguments into a PipelineConfig object."""
    root = get_project_root()
    parser = argparse.ArgumentParser(
        description="D-FINE Human Detection + Depth Anything V2 Metric Depth Integration"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source: camera index (e.g. '0'), path to video file, or path to image/directory (default: '0')",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device: 'cuda:0' or 'cpu' (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="D-FINE human detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--detector-config",
        default=str(
            root / "HUMAN_DETECTION_MODEL" / "HUMAN_DETECTION_MODEL" / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom.yml"
        ),
        help="Path to D-FINE detector YAML configuration",
    )
    parser.add_argument(
        "--detector-ckpt",
        default=str(root / "HUMAN_DETECTION_MODEL" / "HUMAN_DETECTION_MODEL" / "best.pt"),
        help="Path to D-FINE trained checkpoint (best.pt)",
    )
    parser.add_argument(
        "--depth-encoder",
        choices=["vits", "vitb", "vitl"],
        default="vits",
        help="Depth Anything V2 encoder variant (default: vits)",
    )
    parser.add_argument(
        "--depth-dataset",
        choices=["vkitti", "hypersim"],
        default="vkitti",
        help="Depth model dataset: 'vkitti' for outdoor (up to 80m), 'hypersim' for indoor (up to 20m)",
    )
    parser.add_argument(
        "--depth-ckpt",
        default=None,
        help="Path to metric depth checkpoint (.pth). If omitted, defaults to checkpoints/ directory based on dataset/encoder",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Depth model input image size (must be divisible by 14, default: 518)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Max depth in meters (default: 80.0 for vkitti, 20.0 for hypersim)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory to save images, video, and depth numpy files (default: output)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening GUI preview window",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated side-by-side output video",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.5,
        help="Central ROI crop ratio for depth sampling (default: 0.5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process from video or camera stream",
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf", "v4l2", "gstreamer", "any"],
        default="auto",
        help="OpenCV video capture backend for camera (auto: DirectShow on Windows, V4L2 on Linux, GStreamer for CSI)",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=65.0,
        help="Camera horizontal field of view in degrees for 3D distance calculation (default: 65.0)",
    )

    args = parser.parse_args()

    # Determine default depth checkpoint if not explicitly provided
    if args.depth_ckpt:
        depth_ckpt_path = args.depth_ckpt
    else:
        depth_ckpt_path = str(
            root / "checkpoints" / f"depth_anything_v2_metric_{args.depth_dataset}_{args.depth_encoder}.pth"
        )

    # Determine max depth if not specified
    if args.max_depth is not None:
        max_d = args.max_depth
    else:
        max_d = 80.0 if args.depth_dataset == "vkitti" else 20.0

    config = PipelineConfig(
        detector=DetectorConfig(
            config_path=args.detector_config,
            checkpoint_path=args.detector_ckpt,
            confidence_threshold=args.confidence,
        ),
        depth=DepthConfig(
            encoder=args.depth_encoder,
            dataset=args.depth_dataset,
            checkpoint_path=depth_ckpt_path,
            max_depth=max_d,
            input_size=args.input_size,
        ),
        distance=DistanceConfig(
            crop_ratio_x=args.crop_ratio,
            crop_ratio_y=args.crop_ratio,
            max_depth_cutoff=max_d,
            hfov_deg=args.hfov,
        ),
        viz=VisualizationConfig(
            max_viz_depth=15.0 if args.depth_dataset == "hypersim" else 25.0
        ),
        device=args.device,
        source=args.source,
        output_dir=args.output,
        no_display=args.no_display,
        save_video=args.save_video,
        max_frames=args.max_frames,
        camera_backend=args.camera_backend,
    )
    return config


def main() -> None:
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
