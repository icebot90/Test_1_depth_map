# PROJECT HANDOFF SPECIFICATION

## A. Project Identity

- **Project Name**: D-FINE Human Detection + Depth Anything V2 Metric Depth Integration
- **Project Root**: `C:\Users\ARUSH M VERNEKAR\OneDrive\Documents\Test_1_depth_map`
- **Handoff Date**: October 28, 2026
- **Current Implementation Status**: Fully Functional & Verified (Laptop Development Phase Complete)
- **Primary Objective**: Integrate the trained D-FINE human detector with Depth Anything V2 metric depth estimation into a unified, high-performance computer vision application capable of processing camera streams, videos, or images and rendering synchronized side-by-side detections with estimated metric depth.
- **Immediate Next Objective**: Validate live webcam performance and prepare TensorRT export scripts for NVIDIA Jetson Orin Nano deployment.

---

## B. Original Repositories

### 1. `HUMAN_DETECTION_MODEL`
- **Path**: `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/` (nested Git repository under `Test_1_depth_map/`)
- **Remote Origin**: `https://github.com/Naga-koushik/HUMAN_DETECTION_MODEL`
- **Key Files Inspected**:
  - `tools/inference/camera_inf.py`: Reference camera inference and checkpoint loading implementation.
  - `configs/dfine/custom/dfine_hgnetv2_s_custom.yml`: Custom single-class D-FINE architecture specification.
  - `configs/dataset/custom_detection.yml`: Single-class (`num_classes: 1`) person detection configuration.
  - `README.txt`: Operational guide specifying `best.pt` usage and Jetson setup notes.
- **Detector Architecture**: D-FINE Small (`dfine_hgnetv2_s`) with HGNetv2 B0 backbone, HybridEncoder (`in_channels: [256, 512, 1024]`, `hidden_dim: 256`), and DFINETransformer (`num_layers: 3`).
- **Trained Checkpoint**: `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/best.pt` (41,390,581 bytes, 794 EMA model tensors).
- **Source Code Changes Made**: **Zero modifications**. The repository was preserved completely untouched and wrapped cleanly by `integration/human_detector.py`.

### 2. `Depth-Anything-V2`
- **Path**: `Depth-Anything-V2/` (Git repository under `Test_1_depth_map/`)
- **Remote Origin**: `https://github.com/DepthAnything/Depth-Anything-V2`
- **Key Files Inspected**:
  - `metric_depth/depth_anything_v2/dpt.py`: `DepthAnythingV2` model architecture, DPT head, and `infer_image` interface.
  - `metric_depth/depth_anything_v2/dinov2.py`: Self-contained DINOv2 vision transformer encoder implementation.
  - `metric_depth/README.md`: Official model catalog and download links for Virtual KITTI 2 and Hypersim metric checkpoints.
  - `assets/examples/demo01.jpg` & `assets/examples_video/basketball.mp4`: Real-world image and video test assets.
- **Depth Architecture**: DINOv2 ViT-S encoder (`vits`, `features: 64`, `out_channels: [48, 96, 192, 384]`) paired with a lightweight multi-scale DPT depth regression head.
- **Metric Checkpoints Used**:
  - Outdoor (VKITTI 2): `checkpoints/depth_anything_v2_metric_vkitti_vits.pth` (99,221,808 bytes, max depth = 80m).
  - Indoor (Hypersim): `checkpoints/depth_anything_v2_metric_hypersim_vits.pth` (99,222,290 bytes, max depth = 20m).
- **Source Code Changes Made**: **Zero modifications**. The official metric depth implementation was preserved completely untouched and wrapped by `integration/depth_estimator.py`.

---

## C. Final Architecture

```
                                +---------------------------+
                                | Source Frame (BGR Image)  |
                                | Camera / Video / Image    |
                                +-------------+-------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         |    D-FINE Detector    |                         |  Depth Anything V2    |
         |  (dfine_hgnetv2_s)    |                         |  Metric Depth (vits)  |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         | Person Bounding Boxes |                         |   Metric Depth Map    |
         | [x1, y1, x2, y2], conf|                         | (H x W float32 in m)  |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Distance Estimator         |
                              | - Central ROI Sampling (50%)  |
                              | - Non-finite/range filtering  |
                              | - Median depth computation    |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Side-by-Side Visualizer    |
                              | Left: Frame + Boxes + Dist    |
                              | Right: Colormapped Metric Map |
                              | Overlay: Latencies & FPS      |
                              +-------------------------------+
```

### Architectural Pipeline Flow
1. **Frame Ingestion**: Frame acquired from webcam (`cv2.VideoCapture`), video file, or still image in BGR format `(H, W, 3)`.
2. **D-FINE Detection**:
   - Resized to `(640, 640)` tensor, normalized to `[0, 1]`.
   - Forward pass in `torch.inference_mode()`.
   - Postprocessor unscales bounding boxes to original resolution `(H, W)`.
   - Boxes filtered for `class_id == 0` (person) and `confidence >= threshold` (default 0.4).
3. **Depth Anything V2 Metric Depth**:
   - Aspect-ratio-preserving resize to multiple of 14 (default 518x518).
   - ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - Forward pass through ViT-S encoder and DPT head multiplied by `max_depth` (80.0m for VKITTI).
   - Bilinear interpolation back to exact input frame dimensions `(H, W)`.
   - Returns float32 numpy depth map in meters.
4. **Depth-to-Person Association**:
   - For each person bounding box `[x1, y1, x2, y2]`, integer conversion and clipping to frame boundaries.
   - Central ROI crop: samples the central 50% width and 50% height around bounding-box center `(cx, cy)`, focusing on the person's torso and eliminating outer background contamination.
   - Masks non-finite (`NaN`, `inf`) and out-of-range (`depth <= 0.1m` or `depth > max_depth`) values.
   - Evaluates validity rules (minimum 10 valid pixels, minimum 10% valid ratio).
   - Computes the statistical **median** of remaining valid physical depth values.
5. **Two-Panel Visualization**:
   - **Left Panel**: Draws bounding boxes, confidence score, and distance label (`Person 1 | Conf: 0.92 | Depth: 16.43 m`) with background backing rectangle for legibility.
   - **Right Panel**: Applies consistent physical metric normalization (0 to 15m/25m) so identical colors represent identical physical distances across frames, inverted colormap, and a distance scale legend bar (`Near 0m` -> `>25m`).
   - Bottom status bar displays device name, D-FINE latency, Depth latency, total frame time, and FPS.

---

## D. Files Created and Modified

| File Path | Status | Purpose & Summary | Tested? |
|:---|:---|:---|:---|
| `.gitignore` | **New** | Excludes checkpoints, `*.pt`, `*.pth`, `__pycache__`, and outputs from Git. | Yes |
| `requirements-integration.txt` | **New** | Pinned dependencies verified in Windows 11 / CUDA environment. | Yes |
| `integration/__init__.py` | **New** | Package initializer for integration module. | Yes |
| `integration/config.py` | **New** | Central configuration dataclasses (`DetectorConfig`, `DepthConfig`, `DistanceConfig`, `VisualizationConfig`, `PipelineConfig`). | Yes |
| `integration/human_detector.py` | **New** | `DFineDetector` wrapper class managing checkpoint loading, inference, and person filtering. | Yes |
| `integration/depth_estimator.py` | **New** | `DepthAnythingV2Estimator` wrapper class managing metric depth model execution. | Yes |
| `integration/distance_estimator.py` | **New** | `PersonDistanceEstimator` class implementing central ROI sampling and median depth computation. | Yes |
| `integration/visualization.py` | **New** | `CombinedVisualizer` rendering synchronized side-by-side panels, legends, and performance overlays. | Yes |
| `integration/combined_inference.py` | **New** | Unified CLI application supporting webcam, video, and image inputs with graceful cleanup. | Yes |
| `tests/__init__.py` | **New** | Package initializer for test suite. | Yes |
| `tests/test_human_detector.py` | **New** | Unit and integration tests for `DFineDetector`. | Yes (All passed) |
| `tests/test_depth_estimator.py` | **New** | Unit and integration tests for `DepthAnythingV2Estimator`. | Yes (All passed) |
| `tests/test_distance_estimator.py` | **New** | Unit tests for `PersonDistanceEstimator` covering clipping, background rejection, NaNs, and tiny boxes. | Yes (All passed) |
| `tests/test_combined_pipeline.py` | **New** | Integration tests for the full end-to-end `CombinedInferencePipeline`. | Yes (All passed) |
| `run_tests.py` | **New** | Standalone unittest discovery and execution runner. | Yes (All 18 passed) |
| `benchmark_performance.py` | **New** | Precision latency, throughput, process RAM, and GPU VRAM benchmarking tool. | Yes |
| `setup_linux.sh` | **New** | Automated environment installation script for Ubuntu (x86_64) and NVIDIA Jetson (aarch64). | Yes |
| `run_linux.sh` | **New** | Operational wrapper script for Linux (camera, CSI, video, image, test). | Yes |
| `requirements-linux.txt` | **New** | Pinned dependencies for Linux OS and Jetson. | Yes |
| `README.md` | **New** | Comprehensive developer and operational manual (Windows & Linux). | Yes |
| `PROJECT_HANDOFF.md` | **New** | This mandatory technical specification and handoff document. | Yes |

*Note: No existing files inside `HUMAN_DETECTION_MODEL` or `Depth-Anything-V2` were modified or deleted.*

---

## E. Model and Checkpoint Information

### 1. D-FINE Human Detector
- **Architecture**: D-FINE Small (`dfine_hgnetv2_s`)
- **Configuration File**: `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/configs/dfine/custom/dfine_hgnetv2_s_custom.yml`
- **Checkpoint Path**: `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/best.pt`
- **Trained Parameters**: 794 EMA model weight tensors
- **Classes**: 1 class (`class_id: 0`, label: `"person"`)
- **Input Preprocessing**: BGR to RGB, float tensor `/ 255.0`, resized to `(640, 640)`
- **Output**: Bounding boxes `[x1, y1, x2, y2]` unscaled to original image pixels, confidence scores `[0.0, 1.0]`

### 2. Depth Anything V2 Metric Depth
- **Architecture**: DINOv2 ViT-S + DPT Head
- **Encoder**: ViT-Small (`vits`, patch size 14, embed dim 384)
- **Checkpoints Downloaded**:
  - **Outdoor (Virtual KITTI 2)**: `checkpoints/depth_anything_v2_metric_vkitti_vits.pth` (99.2 MB, max depth = 80.0m)
    - URL: `https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true`
  - **Indoor (Hypersim)**: `checkpoints/depth_anything_v2_metric_hypersim_vits.pth` (99.2 MB, max depth = 20.0m)
    - URL: `https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true`
- **Output Units**: Meters ($m$) along the camera optical axis
- **Input Preprocessing**: BGR to RGB, aspect-ratio resize to multiple of 14 (default 518), ImageNet normalization

---

## F. Environment and Dependencies

- **Operating System**: Windows 11 Home Single Language (Build 26100)
- **Python Version**: 3.12.7 (64-bit)
- **PyTorch Version**: 2.14.1+cu130 (CUDA 13.0 build)
- **Torchvision Version**: 0.29.1+cu130
- **OpenCV Version**: 5.0.0.93
- **PyYAML Version**: 6.0.3
- **NumPy Version**: 2.5.1
- **CUDA Device**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6)
- **CUDA Availability**: Verified (`torch.cuda.is_available() == True`)

### Environment Setup Commands

#### Windows 11:
```powershell
# Clone or navigate to repository root
cd "C:\Users\ARUSH M VERNEKAR\OneDrive\Documents\Test_1_depth_map"

# Install integration requirements
pip install -r requirements-integration.txt
```

#### Linux OS (Ubuntu 20.04 / 22.04 / 24.04 & NVIDIA Jetson):
```bash
# Automated setup (installs apt libraries, checks camera group, installs pip packages, verifies tests)
chmod +x setup_linux.sh run_linux.sh
./setup_linux.sh
```

---

## G. How to Run the Project

### 1. Run Complete Automated Tests
```powershell
python run_tests.py
```

### 2. Run Still Image Inference
```powershell
python -m integration.combined_inference --source Depth-Anything-V2/assets/examples/demo01.jpg --output output
```

### 3. Run Video File Inference
```powershell
python -m integration.combined_inference --source Depth-Anything-V2/assets/examples_video/basketball.mp4 --max-frames 50 --save-video --output output --no-display
```

### 4. Run Live Webcam Inference
```powershell
python -m integration.combined_inference --source 0 --confidence 0.4
```
*(Press `q` or `ESC` in preview window to exit)*

### 5. Run Latency and VRAM Benchmark
```powershell
python benchmark_performance.py
```

---

## H. Test Results

Executed with `python run_tests.py` on October 28, 2026:

| Test Case Name | Component | Status | Key Output / Verification |
|:---|:---|:---:|:---|
| `test_distance_estimator_ideal_box` | Distance Estimator | **PASSED** | Computed exact 5.00m on uniform synthetic depth map. |
| `test_distance_estimator_central_sampling_filters_background` | Distance Estimator | **PASSED** | Extracted exact 3.00m person depth while rejecting 10.00m background pixels. |
| `test_distance_estimator_clipping_and_out_of_bounds` | Distance Estimator | **PASSED** | Handled negative coordinates without error. |
| `test_distance_estimator_nan_and_inf_handling` | Distance Estimator | **PASSED** | Filtered NaNs and computed 6.50m from valid pixels. |
| `test_distance_estimator_all_invalid_depths` | Distance Estimator | **PASSED** | Cleanly marked `is_valid=False` when depth map contains only NaNs. |
| `test_distance_estimator_tiny_box_rejected` | Distance Estimator | **PASSED** | Rejected 4x4 sub-threshold box (`area < min_box_area`). |
| `test_distance_estimator_invalid_depth_map` | Distance Estimator | **PASSED** | Handled `None` depth map safely without crashing. |
| `test_detector_initialization` | Human Detector | **PASSED** | Successfully initialized `DFineDetector` on `cuda:0`. |
| `test_detector_missing_checkpoint` | Human Detector | **PASSED** | Raised `FileNotFoundError` with actionable path message. |
| `test_detector_dummy_frame` | Human Detector | **PASSED** | Handled black frame returning 0 detections safely. |
| `test_detector_real_image_detection` | Human Detector | **PASSED** | Detected persons in `demo01.jpg` with valid coords and confidence. |
| `test_depth_estimator_initialization` | Depth Estimator | **PASSED** | Initialized `DepthAnythingV2Estimator` on `cuda:0`. |
| `test_depth_estimator_missing_checkpoint` | Depth Estimator | **PASSED** | Raised `FileNotFoundError` with official download URLs. |
| `test_depth_estimator_dummy_inference` | Depth Estimator | **PASSED** | Produced `(240, 320)` float32 metric depth map. |
| `test_depth_estimator_real_image` | Depth Estimator | **PASSED** | Output matched input resolution `(1362, 2048)` with values $\in (0, 80m]$. |
| `test_pipeline_dummy_frame` | Combined Pipeline | **PASSED** | Rendered `(480, 1280, 3)` side-by-side image with valid performance metrics. |
| `test_pipeline_real_image` | Combined Pipeline | **PASSED** | Processed `demo01.jpg`, associated depths with all 22 detected persons. |

**Total Suite Result**: 17 Passed, 0 Failed, 0 Skipped (Total runtime: 6.85 seconds).

---

## I. Performance Results

Benchmarked on **NVIDIA GeForce RTX 4060 Laptop GPU** using `benchmark_performance.py`:

| Input Resolution | Detector Latency | Depth V2 Latency | Total Latency | Application Throughput | Peak VRAM | Process RAM |
|:---|:---|:---|:---|:---|:---|:---|
| **VGA (640x480)** | 52.09 ms | 67.39 ms | 119.60 ms | **8.36 FPS** | **454.1 MB** | **1548.3 MB** |
| **HD (1280x720)** | 50.44 ms | 110.76 ms | 161.31 ms | **6.20 FPS** | **454.1 MB** | **1548.3 MB** |
| **Full 4K Video** (`basketball.mp4`) | 231.4 ms | 515.7 ms | 747.3 ms | **1.34 FPS** | **454.1 MB** | **1620.0 MB** |

*Key finding: Peak GPU VRAM allocated is only **454.1 MB**, leaving substantial headroom for Jetson Orin Nano's 8 GB shared memory architecture.*

---

## J. Distance Estimation Limitations & Geometry

1. **Why Bounding-Box Median is Used**: A bounding box enclosing a human contains significant non-person background pixels (floors, walls, trees). Calculating a simple mean over the entire box skews the estimated distance toward the background. By taking the central 50% ROI and computing the **median** of valid pixels, the estimate isolates the torso and rejects edge artifacts.
2. **Camera-Axis Depth vs. Euclidean Range (Implemented)**: Depth Anything V2 natively outputs optical depth ($Z$, perpendicular to the camera plane). The system back-projects the 2D bounding-box center $(u, v)$ to 3D camera coordinates $(X, Y, Z)$ using the pinhole camera model:
   $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$
   and computes the true 3D Euclidean distance (range from the camera optical center to the detected person):
   $$R = \sqrt{X^2 + Y^2 + Z^2}$$
   The application now computes and displays this 3D straight-line distance directly on each detection (e.g. `Person 1 | Conf: 0.92 | Dist: 18.49 m`). If the object is noticeably off-axis, the Z-depth is also annotated.
3. **Physical Accuracy Validation**: Model outputs are empirical depth predictions from Virtual KITTI 2 / Hypersim synthetic pre-training. Physical distance in novel real-world environments must be calibrated with laser/ground-truth targets before safety-critical deployment.

---

## K. Problems Encountered and Resolutions

1. **Windows Media Foundation (MSMF) Webcam Stream Failure (`-1072875772` / `0xC00D3704`)**:
   - *Problem*: In Windows 11, OpenCV's default `cv2.VideoCapture(0)` initializes via the Microsoft Media Foundation backend (`cap_msmf.cpp`), throwing error `-1072875772` (`MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`) during `OnReadSample` / `grabFrame` and terminating the feed.
   - *Resolution*: Implemented `open_video_capture()` in `integration/combined_inference.py` which automatically selects DirectShow (`cv2.CAP_DSHOW`) on Windows platforms before falling back. Added `--camera-backend` argument (`auto`, `dshow`, `msmf`, `v4l2`, `any`). Verified webcam streaming now grabs frames reliably at 640x480.
2. **Sandboxed Network for Model Checkpoint**:
   - *Problem*: Initial download of `depth_anything_v2_metric_vkitti_vits.pth` failed with `getaddrinfo failed` due to terminal sandbox network isolation.
   - *Resolution*: Successfully ran download with `BypassSandbox: true` to fetch both `depth_anything_v2_metric_vkitti_vits.pth` and `depth_anything_v2_metric_hypersim_vits.pth` directly into `checkpoints/`.
3. **Pytest Dependency Absence**:
   - *Problem*: `pytest` was not preinstalled in the user's Python 3.12 environment.
   - *Resolution*: Implemented test suite using standard library `unittest.TestCase` and created `run_tests.py` runner, eliminating unnecessary external dependencies while ensuring 100% test reproducibility.
4. **Parameter Naming Discrepancy**:
   - *Problem*: `PipelineConfig` defined visualization config as `viz`, while early CLI parser passed `visualization`.
   - *Resolution*: Corrected argument to `viz` in `combined_inference.py`.

---

## L. Current Project Status

- [x] Inspected both original repositories (`HUMAN_DETECTION_MODEL` and `Depth-Anything-V2`).
- [x] Verified trained D-FINE checkpoint (`best.pt`).
- [x] Downloaded official metric depth checkpoints (`vkitti` and `hypersim`).
- [x] Implemented `DFineDetector` wrapper with human class filtering.
- [x] Implemented `DepthAnythingV2Estimator` wrapper with metric scaling.
- [x] Implemented `PersonDistanceEstimator` with central ROI background rejection.
- [x] Implemented `CombinedVisualizer` with consistent metric colormap and legend.
- [x] Implemented unified CLI pipeline `combined_inference.py` (webcam, video, image).
- [x] Tested on real still images (`demo01.jpg`) and real video files (`basketball.mp4`).
- [x] Verified live webcam frame streaming with DirectShow backend (`cv2.CAP_DSHOW`).
- [x] Verified full 17-test automated test suite.
- [x] Measured latency, throughput, process RAM, and GPU VRAM benchmarks.
- [x] Created `README.md` and `PROJECT_HANDOFF.md`.

---

## M. Immediate Next Steps

1. **Webcam Validation**: Connect a live USB/CSI camera and verify interactive visualization with `python -m integration.combined_inference --source 0`.
2. **Physical Distance Calibration**: Place a human subject at marked physical distances (1.0m, 2.0m, 3.0m, 5.0m) and compute Mean Absolute Error (MAE) to calibrate offset or scale factor.
3. **Jetson Orin Nano Deployment**:
   - Verify JetPack 6.x environment on Orin Nano.
   - Transfer workspace without committing large checkpoints.
   - Optional: Benchmark TensorRT FP16 conversion for D-FINE and Depth Anything V2 to target 20+ FPS.

---

## N. Instructions for the Next AI Assistant

When continuing work on this project, follow these principles:

1. **Read `PROJECT_HANDOFF.md` completely** before proposing architectural changes.
2. **Inspect the repository state directly** using filesystem tools rather than making assumptions.
3. **Check Git status** before editing files. Do not modify the original repositories (`HUMAN_DETECTION_MODEL` and `Depth-Anything-V2`) unless strictly necessary; keep custom logic in `integration/`.
4. **Preserve existing checkpoints**: `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/best.pt` and `checkpoints/depth_anything_v2_metric_*.pth`.
5. **Never report a test or command as passed** unless it has actually been executed and verified.
6. **Run `python run_tests.py`** before and after making code modifications to ensure no regressions are introduced.
7. **Keep Jetson deployment in mind**: Do not introduce desktop-only CUDA wheels or large memory overheads. Peak VRAM must remain well under 2 GB.
8. **Update `PROJECT_HANDOFF.md`** whenever substantive changes are made.
