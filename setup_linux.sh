#!/usr/bin/env bash
# ==============================================================================
# Linux / Ubuntu Setup Script for D-FINE + Depth Anything V2 Integration
# Supports Ubuntu 20.04 / 22.04 / 24.04 (x86_64) and NVIDIA Jetson (aarch64)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Linux Environment Setup for D-FINE + Depth Anything V2"
echo " Working directory: $SCRIPT_DIR"
echo "======================================================================"

# 1. Detect Architecture and Operating System
OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Linux" ]; then
    echo "Error: This script is intended for Linux (Ubuntu), but detected $OS."
    exit 1
fi

echo "[1/6] Detected OS: $OS | Architecture: $ARCH"

# 2. Check Ubuntu system dependencies for OpenCV and Camera video capture
echo "[2/6] Checking system package dependencies..."
MISSING_PACKAGES=()

check_pkg() {
    dpkg -s "$1" >/dev/null 2>&1 || MISSING_PACKAGES+=("$1")
}

# Check essential libraries for OpenCV GUI and videoio
if command -v dpkg >/dev/null 2>&1; then
    check_pkg "libgl1" || check_pkg "libgl1-mesa-glx"
    check_pkg "libglib2.0-0"
    check_pkg "v4l-utils"
    check_pkg "python3-pip"

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo "Missing system packages detected: ${MISSING_PACKAGES[*]}"
        echo "Installing system packages via apt..."
        if [ "$EUID" -ne 0 ]; then
            sudo apt-get update && sudo apt-get install -y "${MISSING_PACKAGES[@]}"
        else
            apt-get update && apt-get install -y "${MISSING_PACKAGES[@]}"
        fi
    else
        echo "All required system libraries are installed."
    fi
else
    echo "dpkg not found, skipping apt package check."
fi

# 3. Check video group permissions for camera access (/dev/video*)
echo "[3/6] Checking user video group permissions..."
if groups | grep -q "\bvideo\b"; then
    echo "User '$USER' is already in the 'video' group."
else
    echo "Notice: User '$USER' is not in the 'video' group."
    echo "To access USB/CSI cameras without sudo, run:"
    echo "  sudo usermod -a -G video $USER"
    echo "  (Then log out and log back in)"
fi

# 4. Check / Install PyTorch
echo "[4/6] Verifying PyTorch installation..."
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

HAS_TORCH=false
if $PYTHON_BIN -c "import torch" >/dev/null 2>&1; then
    HAS_TORCH=true
    TORCH_VER=$($PYTHON_BIN -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$($PYTHON_BIN -c "import torch; print(torch.cuda.is_available())")
    echo "PyTorch is installed: version $TORCH_VER | CUDA available: $CUDA_AVAIL"
else
    echo "PyTorch is NOT installed."
fi

if [ "$HAS_TORCH" = false ]; then
    if [ "$ARCH" = "aarch64" ]; then
        echo "----------------------------------------------------------------------"
        echo "WARNING: NVIDIA Jetson platform detected ($ARCH)."
        echo "Do NOT install standard desktop PyTorch wheels via pip on Jetson."
        echo "Please install the official NVIDIA JetPack PyTorch wheel:"
        echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo "----------------------------------------------------------------------"
        exit 1
    else
        echo "Installing PyTorch with CUDA support for x86_64 Ubuntu..."
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "NVIDIA GPU detected via nvidia-smi. Installing CUDA 12.1 PyTorch wheel..."
            $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            echo "No NVIDIA GPU detected. Installing CPU PyTorch build..."
            $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
fi

# 5. Install Python dependencies
echo "[5/6] Installing integration dependencies..."
$PYTHON_BIN -m pip install -r requirements-linux.txt

# 6. Check Checkpoints
echo "[6/6] Verifying model checkpoints..."
mkdir -p checkpoints

DFINE_CKPT="HUMAN_DETECTION_MODEL/HUMAN_DETECTION_MODEL/best.pt"
if [ -f "$DFINE_CKPT" ]; then
    echo "D-FINE checkpoint found: $DFINE_CKPT"
else
    echo "ERROR: D-FINE checkpoint missing at $DFINE_CKPT"
    exit 1
fi

DEPTH_CKPT="checkpoints/depth_anything_v2_metric_vkitti_vits.pth"
if [ -f "$DEPTH_CKPT" ]; then
    echo "Depth Anything V2 checkpoint found: $DEPTH_CKPT"
else
    echo "Downloading official Depth Anything V2 VKITTI Small checkpoint..."
    VKITTI_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DEPTH_CKPT" "$VKITTI_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$DEPTH_CKPT" "$VKITTI_URL"
    else
        $PYTHON_BIN -c "import urllib.request; urllib.request.urlretrieve('$VKITTI_URL', '$DEPTH_CKPT')"
    fi
fi

echo "======================================================================"
echo " Setup complete! Running integration test suite to verify..."
echo "======================================================================"
$PYTHON_BIN run_tests.py

echo "======================================================================"
echo " SUCCESS: The system is ready to run on Linux / Ubuntu!"
echo " Quick start commands:"
echo "   Test image:  $PYTHON_BIN -m integration.combined_inference --source Depth-Anything-V2/assets/examples/demo01.jpg"
echo "   USB Webcam:  $PYTHON_BIN -m integration.combined_inference --source 0"
echo "   Jetson CSI:  $PYTHON_BIN -m integration.combined_inference --source csi://0"
echo "   Headless:    $PYTHON_BIN -m integration.combined_inference --source 0 --no-display --save-video"
echo "======================================================================"
