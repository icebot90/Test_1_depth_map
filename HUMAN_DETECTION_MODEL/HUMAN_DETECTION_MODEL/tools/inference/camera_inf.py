"""Run a D-FINE checkpoint on a live camera and save detected frames."""

import argparse
from io import BytesIO
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def load_checkpoint(checkpoint_path: str) -> dict:
    checkpoint = Path(checkpoint_path)
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
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(loaded).__name__}.")
    return loaded


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    cfg = YAMLConfig(config_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = load_checkpoint(checkpoint_path)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(
            self, images: torch.Tensor, original_sizes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.model(images)
            return self.postprocessor(outputs, original_sizes)

    model = Model().to(device)
    model.eval()
    return model


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )
        try:
            torch.zeros(1, device=device)
        except RuntimeError as error:
            gpu_name = torch.cuda.get_device_name(device)
            raise RuntimeError(
                f"PyTorch cannot execute on {gpu_name} with this installation. "
                f"PyTorch {torch.__version__} supports CUDA {torch.version.cuda}. "
                "Install a newer CUDA PyTorch wheel that supports this GPU."
            ) from error
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")
    return device


def detect(
    model: nn.Module,
    frame: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    confidence: float,
) -> list[dict[str, object]]:
    frame = frame.to(device, non_blocking=True)
    resized = torch.nn.functional.interpolate(
        frame.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False
    )
    original_size = torch.tensor([[width, height]], device=device)

    with torch.inference_mode():
        labels, boxes, scores = model(resized, original_size)

    detections = []
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        confidence_value = float(score.item())
        if confidence_value < confidence:
            continue
        coordinates = [round(float(value), 2) for value in box.tolist()]
        detections.append(
            {
                "class_id": int(label.item()),
                "confidence": round(confidence_value, 6),
                "box": {
                    "x1": coordinates[0],
                    "y1": coordinates[1],
                    "x2": coordinates[2],
                    "y2": coordinates[3],
                },
            }
        )
    return detections


def draw_detections(
    frame,
    detections: list[dict[str, object]],
    label_name: str,
) -> None:
    for detection in detections:
        box = detection["box"]
        x1, y1, x2, y2 = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
        text = f"{label_name} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.config, args.resume, device)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    frame_number = 0
    saved_count = 0
    print("Camera inference started. Press q in the preview window to stop.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            detections = detect(model, tensor, width, height, device, args.confidence)

            preview = frame.copy()
            draw_detections(preview, detections, args.label)
            if detections:
                stem = f"frame_{frame_number:06d}"
                image_path = output_dir / f"{stem}.jpg"
                json_path = output_dir / f"{stem}.json"
                record = {
                    "frame": frame_number,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "detections": detections,
                }
                if not cv2.imwrite(str(image_path), preview):
                    raise RuntimeError(f"Could not save detection image to {image_path}.")
                json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                saved_count += 1

            if not args.no_display:
                cv2.imshow("D-FINE camera inference", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved_count} detected frame(s) to {output_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a D-FINE checkpoint on a camera and save detected frames."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/dfine/custom/dfine_hgnetv2_s_custom.yml",
        help="D-FINE YAML config.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=r"best.pt",
        help="Checkpoint file or extracted checkpoint directory.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--output", default="camera_results", help="Folder for annotated images and JSON."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4, help="Minimum confidence (default: 0.4)."
    )
    parser.add_argument("--label", default="person", help="Label drawn on each box.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--no-display", action="store_true", help="Run without opening a preview window."
    )
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")
    run(args)


if __name__ == "__main__":
    main()
