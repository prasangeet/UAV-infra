from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "runs/inference/images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO crack/damage inference on one image.")
    parser.add_argument("source", type=Path, help="Input image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()

    source, model_path = args.source.expanduser().resolve(), args.model.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {source.suffix}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if args.imgsz < 32 or not 0 <= args.conf <= 1 or not 0 < args.iou < 1:
        raise ValueError("Invalid imgsz/conf/iou value.")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"Expected detection model, got {model.task!r}.")

    results = model.predict(
        source=str(source), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, save=True, show=args.show,
        project=str(args.project.resolve()), name=args.name or f"image_{source.stem}",
        exist_ok=False, line_width=args.line_width, verbose=False,
    )

    result = results[0]
    boxes = result.boxes
    print(f"Image: {source}")
    if boxes is None or len(boxes) == 0:
        print("No detections above the confidence threshold.")
    else:
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls.item())
            cls_name = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            print(f"{i:>3}. {cls_name:<12} confidence={conf:.3f} bbox={xyxy}")

    print(f"Annotated image: {Path(model.predictor.save_dir).resolve()}")


if __name__ == "__main__":
    main()
