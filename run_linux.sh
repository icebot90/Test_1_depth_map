#!/usr/bin/env bash
# ==============================================================================
# Linux / Ubuntu Execution Helper for D-FINE + Depth Anything V2 Integration
# Usage:
#   ./run_linux.sh camera [index] [--headless]
#   ./run_linux.sh csi [sensor_id] [--headless]
#   ./run_linux.sh video [path_to_video] [--headless]
#   ./run_linux.sh image [path_to_image]
#   ./run_linux.sh test
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

MODE="${1:-camera}"
ARG2="${2:-}"
ARG3="${3:-}"

# Check for headless flag in any argument
EXTRA_FLAGS=""
if [[ "$*" == *"--headless"* ]] || [[ "$*" == *"--no-display"* ]]; then
    EXTRA_FLAGS="--no-display"
fi

case "$MODE" in
    camera|cam|webcam)
        CAM_IDX="${ARG2:-0}"
        # If user passed --headless as 2nd argument
        if [ "$CAM_IDX" = "--headless" ]; then
            CAM_IDX="0"
        fi
        echo "Running on USB/V4L2 camera (index: $CAM_IDX)..."
        $PYTHON_BIN -m integration.combined_inference --source "$CAM_IDX" $EXTRA_FLAGS
        ;;

    csi)
        SENSOR_ID="${ARG2:-0}"
        if [ "$SENSOR_ID" = "--headless" ]; then
            SENSOR_ID="0"
        fi
        echo "Running on NVIDIA Jetson CSI camera (sensor-id: $SENSOR_ID)..."
        $PYTHON_BIN -m integration.combined_inference --source "csi://$SENSOR_ID" $EXTRA_FLAGS
        ;;

    video)
        VIDEO_PATH="${ARG2:-Depth-Anything-V2/assets/examples_video/basketball.mp4}"
        echo "Running on video file: $VIDEO_PATH..."
        $PYTHON_BIN -m integration.combined_inference --source "$VIDEO_PATH" --save-video $EXTRA_FLAGS
        ;;

    image|img)
        IMG_PATH="${ARG2:-Depth-Anything-V2/assets/examples/demo01.jpg}"
        echo "Running on image: $IMG_PATH..."
        $PYTHON_BIN -m integration.combined_inference --source "$IMG_PATH" $EXTRA_FLAGS
        ;;

    test)
        echo "Running complete integration test suite..."
        $PYTHON_BIN run_tests.py
        ;;

    benchmark)
        echo "Running latency and VRAM performance benchmark..."
        $PYTHON_BIN benchmark_performance.py
        ;;

    *)
        echo "Usage: $0 {camera|csi|video|image|test|benchmark} [source_arg] [--headless]"
        echo "Examples:"
        echo "  $0 camera 0                     # Live USB webcam"
        echo "  $0 csi 0                        # Jetson CSI camera"
        echo "  $0 video path/to/video.mp4      # Video file"
        echo "  $0 image path/to/image.jpg      # Still image"
        echo "  $0 camera 0 --headless          # Headless server mode"
        echo "  $0 test                         # Run automated test suite"
        exit 1
        ;;
esac
