from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Latest 1024px YOLO11n experiment.
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "cubit_det_yolo11n_1024-2"
    / "weights"
    / "best.pt"
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "cubit_det" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained YOLO detector on CUBIT-Det. "
            "Defaults target the 1024px YOLO11n checkpoint."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Trained detection checkpoint (.pt).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Ultralytics dataset YAML.")
    parser.add_argument(
        "--split", choices=["val", "test", "train"], default="val",
        help="Dataset split to evaluate (default: val; use test only for final evaluation)."
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index, or 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold used for metric computation.")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU threshold.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=None,
                        help="Output run name; generated automatically if omitted.")
    parser.add_argument("--plots", choices=["true", "false"], default="true")
    parser.add_argument("--exist-ok", action="store_true",
                        help="Allow an existing output directory to be reused.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.model = args.model.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.project = args.project.expanduser().resolve()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32.")
    if args.batch < 1 or args.workers < 0:
        raise ValueError("--batch must be >= 1 and --workers cannot be negative.")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0, 1].")
    if not 0.0 < args.iou < 1.0:
        raise ValueError("--iou must be in (0, 1).")

    if args.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but PyTorch cannot access a CUDA GPU."
            )
        if args.device.isdigit() and int(args.device) >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {args.device} is not available.")

    if args.name is None:
        args.name = f"{args.model.parent.parent.name}_{args.split}_evaluation"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_class_metrics(
    output_dir: Path, model: YOLO, box: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_indices = list(getattr(box, "ap_class_index", []))
    precision = getattr(box, "p", [])
    recall = getattr(box, "r", [])
    ap50 = getattr(box, "ap50", [])
    ap = getattr(box, "ap", [])

    for position, class_index in enumerate(class_indices):
        class_index = int(class_index)
        class_name = model.names.get(class_index, str(class_index))

        def value_at(values: Any) -> float | None:
            try:
                return to_float(values[position])
            except (IndexError, TypeError):
                return None

        try:
            ap_mean = to_float(ap[position].mean())
        except (IndexError, TypeError, AttributeError):
            ap_mean = None

        rows.append({
            "class_id": class_index,
            "class_name": class_name,
            "precision": value_at(precision),
            "recall": value_at(recall),
            "map50": value_at(ap50),
            "map50_95": ap_mean,
        })

    with (output_dir / "class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = ["class_id", "class_name", "precision", "recall", "map50", "map50_95"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    print("=" * 72)
    print("YOLO OBJECT DETECTION EVALUATION")
    print("=" * 72)
    print(f"Started:       {started_at}")
    print(f"Checkpoint:    {args.model}")
    print(f"Dataset:       {args.data}")
    print(f"Split:         {args.split}")
    print(f"Image size:    {args.imgsz}")
    print(f"Batch size:    {args.batch}")
    print(f"Workers:       {args.workers}")
    print(f"Device:        {args.device}")
    print(f"Confidence:    {args.conf}")
    print(f"NMS IoU:       {args.iou}")

    if args.device != "cpu":
        device_index = int(args.device) if args.device.isdigit() else 0
        print(f"GPU:           {torch.cuda.get_device_name(device_index)}")

    print("=" * 72)

    model = YOLO(str(args.model))
    if model.task != "detect":
        raise ValueError(
            f"Expected a detection checkpoint, got task={model.task!r}."
        )

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=args.plots == "true",
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    output_dir = Path(metrics.save_dir).resolve()
    box = metrics.box

    overall = {
        "precision": to_float(getattr(box, "mp", None)),
        "recall": to_float(getattr(box, "mr", None)),
        "map50": to_float(getattr(box, "map50", None)),
        "map50_95": to_float(getattr(box, "map", None)),
    }

    class_rows = write_class_metrics(output_dir, model, box)

    speed = getattr(metrics, "speed", {}) or {}
    speed_ms = {key: to_float(value) for key, value in speed.items()}

    summary = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(args.model),
        "dataset_yaml": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "metrics": overall,
        "speed_ms_per_image": speed_ms,
        "per_class": class_rows,
        "results_directory": str(output_dir),
    }

    json_path = output_dir / "evaluation_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "YOLO OBJECT DETECTION EVALUATION",
        "=" * 40,
        f"Checkpoint: {args.model}",
        f"Dataset:    {args.data}",
        f"Split:      {args.split}",
        f"Image size: {args.imgsz}",
        "",
        "Overall metrics",
        f"Precision:  {overall['precision']}",
        f"Recall:     {overall['recall']}",
        f"mAP50:      {overall['map50']}",
        f"mAP50-95:   {overall['map50_95']}",
        "",
        "Speed (ms/image)",
    ]

    for key, value in speed_ms.items():
        lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "Per-class metrics",
        "Class ID | Class | Precision | Recall | mAP50 | mAP50-95",
    ])

    for row in class_rows:
        lines.append(
            f"{row['class_id']} | {row['class_name']} | "
            f"{row['precision']} | {row['recall']} | "
            f"{row['map50']} | {row['map50_95']}"
        )

    txt_path = output_dir / "evaluation_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nEvaluation completed.")
    for label, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("mAP50", "map50"),
        ("mAP50-95", "map50_95"),
    ]:
        value = overall[key]
        print(
            f"{label + ':':16}{value:.4f}"
            if value is not None
            else f"{label + ':':16}N/A"
        )

    print(f"Results:         {output_dir}")
    print(f"Summary JSON:    {json_path}")
    print(f"Summary TXT:     {txt_path}")
    print(f"Class CSV:       {output_dir / 'class_metrics.csv'}")


if __name__ == "__main__":
    main()
