from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "runs/inference/videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO crack/damage inference on a video.")
    parser.add_argument("source", type=Path, help="Input video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--vid-stride", type=int, default=1,
                        help="Process every Nth frame; 1 means every frame.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()

    source, model_path = args.source.expanduser().resolve(), args.model.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {source.suffix}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if args.imgsz < 32 or not 0 <= args.conf <= 1 or not 0 < args.iou < 1:
        raise ValueError("Invalid imgsz/conf/iou value.")
    if args.vid_stride < 1:
        raise ValueError("--vid-stride must be >= 1.")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"Expected detection model, got {model.task!r}.")

    counts = Counter()
    confidence_sums = defaultdict(float)
    frames = frames_with_detections = total_detections = 0

    stream = model.predict(
        source=str(source), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, stream=True, save=True, show=args.show,
        project=str(args.project.resolve()), name=args.name or f"video_{source.stem}",
        exist_ok=False, vid_stride=args.vid_stride, line_width=args.line_width,
        verbose=False,
    )

    for result in stream:
        frames += 1
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        frames_with_detections += 1
        total_detections += len(boxes)
        for cls_tensor, conf_tensor in zip(boxes.cls, boxes.conf):
            cls_id = int(cls_tensor.item())
            cls_name = result.names.get(cls_id, str(cls_id))
            counts[cls_name] += 1
            confidence_sums[cls_name] += float(conf_tensor.item())
        if frames % 100 == 0:
            print(f"Processed {frames} frames; detections so far: {total_detections}")

    print("\nInference summary")
    print(f"Frames processed:       {frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections:       {total_detections}")
    for cls_name, count in counts.most_common():
        mean_conf = confidence_sums[cls_name] / count
        print(f"{cls_name:<12} {count:>6} detections (mean confidence {mean_conf:.3f})")

    print(f"Annotated video: {Path(model.predictor.save_dir).resolve()}")


if __name__ == "__main__":
    main()
