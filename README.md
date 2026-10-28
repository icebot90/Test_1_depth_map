# D-FINE Human Detection + Depth Anything V2 Metric Depth Integration

This project integrates real-time human detection using the trained **D-FINE** (`dfine_hgnetv2_s`) detector with metric depth estimation using **Depth Anything V2** (`vits` encoder, Virtual KITTI 2 / Hypersim).

The combined application runs inference simultaneously on camera feeds, video files, or still images, extracts metric depth for each detected person using central ROI sampling, and renders a synchronized side-by-side visualization:
- **Left Panel**: Original camera feed with annotated person bounding boxes, detection confidence scores, and estimated metric depth.
- **Right Panel**: Color-coded metric depth map with consistent physical distance scaling and range legend.

---

## Architecture Overview

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

---

## Directory Structure

```
Test_1_depth_map/
├── HUMAN_DETECTION_MODEL/
│   └── HUMAN_DETECTION_MODEL/       # D-FINE detector repository
│       ├── configs/                  # Model configurations (YAML)
│       ├── src/                      # D-FINE model source code
│       ├── tools/                    # Native camera inference scripts
│       └── best.pt                   # Trained detector checkpoint weights
├── Depth-Anything-V2/                # Depth Anything V2 repository
│   ├── assets/                       # Sample images and video assets
│   ├── depth_anything_v2/            # Base relative depth code
│   └── metric_depth/                 # Metric depth estimation models & heads
├── checkpoints/                      # Downloaded metric depth checkpoints
│   ├── depth_anything_v2_metric_vkitti_vits.pth   # Outdoor model (80m max)
│   └── depth_anything_v2_metric_hypersim_vits.pth # Indoor model (20m max)
├── integration/                      # Modular integration package
│   ├── __init__.py
│   ├── config.py                     # Dataclass configurations
│   ├── human_detector.py             # D-FINE wrapper and detection filtering
│   ├── depth_estimator.py            # Depth Anything V2 metric depth wrapper
│   ├── distance_estimator.py         # Robust central ROI depth association
│   ├── visualization.py              # Two-panel side-by-side visualizer
│   └── combined_inference.py         # End-to-end CLI pipeline application
├── tests/                            # Automated test suite (18 tests)
│   ├── test_human_detector.py
│   ├── test_depth_estimator.py
│   ├── test_distance_estimator.py
│   └── test_combined_pipeline.py
├── benchmark_performance.py          # Latency, FPS, and VRAM benchmarking tool
├── run_tests.py                      # Unittest runner script
├── setup_linux.sh                    # Automated setup script for Ubuntu / Linux
├── run_linux.sh                      # Execution helper for Ubuntu / Linux
├── requirements-integration.txt      # Windows dependency specification
├── requirements-linux.txt            # Linux / Ubuntu dependency specification
├── README.md                         # Project documentation
└── PROJECT_HANDOFF.md                # Detailed engineering handoff specification
```

---

## Linux OS (Ubuntu) & NVIDIA Jetson Setup

This project is fully compatible with **Linux OS** (Ubuntu 20.04, 22.04, 24.04 on x86_64 PC/Server and NVIDIA Jetson on aarch64/ARM64).

### 1. One-Step Automated Setup on Ubuntu / Linux

Run the included automated setup script:
```bash
chmod +x setup_linux.sh run_linux.sh
./setup_linux.sh
```
This script automatically:
- Detects OS architecture (`x86_64` vs Jetson `aarch64`).
- Installs necessary Ubuntu system libraries (`libgl1`, `libglib2.0-0`, `v4l-utils`).
- Checks user camera permissions and prompts `sudo usermod -a -G video $USER` if needed.
- Checks/installs PyTorch (preserving pre-installed JetPack torch on Jetson, or installing CUDA PyTorch on x86_64).
- Installs Python dependencies from `requirements-linux.txt`.
- Verifies model checkpoints and runs the automated 18-test suite.

### 2. Manual Setup on Ubuntu

If you prefer manual setup:
```bash
# 1. Install system libraries for OpenCV and camera I/O
sudo apt-get update && sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    python3-pip \
    python3-dev

# 2. Grant camera access to user
sudo usermod -a -G video $USER
# (Log out and log back in for group change to take effect)

# 3. Install PyTorch:
# On Ubuntu x86_64 with NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# On NVIDIA Jetson (JetPack 5/6):
# Keep pre-installed NVIDIA PyTorch or install official wheel from:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# 4. Install Python dependencies
pip install -r requirements-linux.txt
```

### 3. Running on Linux / Ubuntu

Use the `./run_linux.sh` wrapper or standard Python commands:

```bash
# Live USB Webcam (V4L2)
./run_linux.sh camera 0
# or: python3 -m integration.combined_inference --source 0

# NVIDIA Jetson Onboard CSI Camera (via hardware GStreamer pipeline)
./run_linux.sh csi 0
# or: python3 -m integration.combined_inference --source csi://0

# Video File
./run_linux.sh video Depth-Anything-V2/assets/examples_video/basketball.mp4

# Still Image
./run_linux.sh image Depth-Anything-V2/assets/examples/demo01.jpg

# Headless mode (Remote Server / Docker / SSH without X11)
./run_linux.sh camera 0 --headless
# or: python3 -m integration.combined_inference --source 0 --no-display --save-video

# Run Test Suite
./run_linux.sh test
```

---

## Windows Prerequisites and Environment Setup

### Verified Laptop Environment (Windows 11)
- **Python**: 3.12.7 (64-bit)
- **PyTorch**: 2.14.1+cu130 (with CUDA support)
- **Torchvision**: 0.29.1+cu130
- **OpenCV**: 5.0.0.93 / 4.x
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)

### Installation

1. Activate your Python environment.
2. Install the integration dependencies:
   ```powershell
   pip install -r requirements-integration.txt
   ```
3. Checkpoint verification:
   - D-FINE trained weights are already present at:
     `HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/best.pt`
   - Depth Anything V2 Metric checkpoints are placed in `checkpoints/`:
     - Outdoor Virtual KITTI 2 (Small): `checkpoints/depth_anything_v2_metric_vkitti_vits.pth`
     - Indoor Hypersim (Small): `checkpoints/depth_anything_v2_metric_hypersim_vits.pth`

   *(If missing, download URLs are documented in the checkpoints section below)*.

---

## Running the Application

### 1. Still Image Inference
Run inference on a single test image and save the visual output and raw depth map array:
```powershell
python -m integration.combined_inference --source Depth-Anything-V2/assets/examples/demo01.jpg --output output
```
To run headless without opening a GUI preview window:
```powershell
python -m integration.combined_inference --source Depth-Anything-V2/assets/examples/demo01.jpg --output output --no-display
```

### 2. Video File Inference
Run inference on a video file, optionally limiting processed frames and saving the side-by-side annotated video:
```powershell
python -m integration.combined_inference --source Depth-Anything-V2/assets/examples_video/basketball.mp4 --max-frames 50 --save-video --output output --no-display
```

### 3. Live Webcam Feed
Run live inference on your default webcam (`--source 0`):
```powershell
python -m integration.combined_inference --source 0 --confidence 0.4
```
Press `q` or `ESC` in the preview window to exit cleanly.

### 4. Running for Indoor Environments
Switch to the indoor Hypersim metric depth model:
```powershell
python -m integration.combined_inference --source 0 --depth-dataset hypersim --max-depth 20.0
```

---

## Command-Line Arguments Reference

| Argument | Default | Description |
|:---|:---|:---|
| `--source` | `0` | Camera index (`0`), video file path, or image file/directory path |
| `--device` | `cuda:0` | Inference device (`cuda:0` or `cpu`) |
| `--confidence` | `0.4` | D-FINE human detection confidence threshold |
| `--depth-dataset` | `vkitti` | Depth model dataset (`vkitti` for outdoor up to 80m, `hypersim` for indoor up to 20m) |
| `--depth-encoder` | `vits` | Model encoder size (`vits`, `vitb`, `vitl`) |
| `--depth-ckpt` | *auto* | Explicit path to depth checkpoint |
| `--input-size` | `518` | Depth model input resolution (multiple of 14) |
| `--crop-ratio` | `0.5` | Central ROI sampling ratio (0.5 = central 50% width & height) |
| `--max-frames` | *None* | Limit number of frames processed (useful for videos) |
| `--save-video` | *False* | Save annotated side-by-side video to output directory |
| `--output` | `output` | Directory where images, videos, and numpy arrays are saved |
| `--no-display` | *False* | Run in headless mode without X11/GUI window display |

---

## Testing and Verification

Run the complete 17-test automated test suite:
```powershell
python run_tests.py
```

All 17 tests verify:
- Bounding-box clipping, central ROI extraction, and out-of-bounds safety.
- Non-finite (`NaN`, `inf`) and background outlier filtering.
- Rejection of invalid, tiny, or degraded depth regions.
- D-FINE detector loading, forward pass, and human class filtering.
- Depth Anything V2 metric output shape, dtype (`float32`), and physical range.
- End-to-end pipeline coordination, side-by-side image generation, and timing overlays.

---

## Measured Performance (RTX 4060 Laptop GPU)

| Input Resolution | D-FINE Latency | Depth V2 Latency | Total Latency | Application FPS | Peak VRAM |
|:---|:---|:---|:---|:---|:---|
| **VGA (640x480)** | 52.1 ms | 67.4 ms | 119.6 ms | **8.36 FPS** | **454 MB** |
| **HD (1280x720)** | 50.4 ms | 110.8 ms | 161.3 ms | **6.20 FPS** | **454 MB** |

Peak GPU VRAM is exceptionally lightweight at **~454 MB**, making the architecture highly suitable for memory-constrained platforms like the Jetson Orin Nano (8 GB shared LPDDR5).

---

## Distance Estimation Limitations & Geometry

1. **Camera-Axis Depth vs. Euclidean Distance**: The depth values predicted by Depth Anything V2 represent depth along the optical $Z$-axis (perpendicular to the camera sensor plane). For objects significantly off-center, the true 3D Euclidean range is $\sqrt{X^2 + Y^2 + Z^2}$. The current labels display estimated optical depth (`Depth: X.XX m`).
2. **Bounding Box Sampling**: Bounding boxes include non-person background pixels (e.g. walls, ground). The integration mitigates this by extracting the central 50% area of the box and taking the median of valid pixels. For thin limbs or occlusion, background contamination may still occur.
3. **Physical Accuracy Validation**: While the model is trained on Virtual KITTI 2 / Hypersim metric depth, physical distance in novel real-world scenes is an empirical model estimate and must be calibrated against laser/lidar or known ground truth targets before safety-critical use.

---

## Future NVIDIA Jetson Orin Nano Deployment

- **Memory**: The combined pipeline uses ~454 MB GPU VRAM and ~1.5 GB system RAM, comfortably within the Jetson Orin Nano 8 GB budget.
- **JetPack Compatibility**: On Jetson, use system PyTorch and torchvision provided by NVIDIA L4T/JetPack 6.x. Do not overwrite with desktop PyTorch wheels.
- **Acceleration**: Future optimization can convert the D-FINE backbone and Depth Anything V2 ViT encoder to NVIDIA TensorRT (FP16 or INT8) to achieve 15–25 FPS on Orin Nano.
