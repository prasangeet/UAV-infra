from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "cubit_det" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the original, non-tiled three-class CUBIT-Det dataset."
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)

    # Conservative defaults for a local RTX 2050 with 4 GB VRAM.
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", choices=["true", "false"], default="true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)

    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)

    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="false")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="cubit_det_yolo11n_1024")
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume <project>/<name>/weights/last.pt with its saved optimizer and arguments.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("--epochs and --batch must be positive; --imgsz must be >= 32.")
    if args.workers < 0 or args.patience < 0 or args.warmup_epochs < 0:
        raise ValueError("--workers, --patience and --warmup-epochs cannot be negative.")
    if args.lr0 <= 0 or not 0 < args.lrf <= 1 or args.weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay.")
    if args.close_mosaic < 0 or args.save_period == 0 or args.save_period < -1:
        raise ValueError("Invalid close-mosaic or save-period value.")

    for key in ("mosaic", "mixup", "fliplr", "flipud"):
        if not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"--{key} must be in [0, 1].")
    for key in ("hsv_h", "hsv_s", "hsv_v", "scale", "degrees", "translate"):
        if getattr(args, key) < 0:
            raise ValueError(f"--{key} cannot be negative.")

    args.data = args.data.expanduser().resolve()
    args.project = args.project.expanduser().resolve()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA GPU.")
        if args.device.isdigit() and int(args.device) >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {args.device} is not available.")


def training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    cache: bool | str = False if args.cache == "false" else args.cache
    return {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "amp": args.amp == "true",
        "cache": cache,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": False,
        "plots": True,
        "save": True,
        "save_period": args.save_period,
        "val": True,
        "close_mosaic": args.close_mosaic,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "warmup_epochs": args.warmup_epochs,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "scale": args.scale,
        "degrees": args.degrees,
        "translate": args.translate,
    }


def write_metadata(run_dir: Path, args: argparse.Namespace, status: str, model_source: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        "Run name": args.name,
        "Dataset": "Original CUBIT-Det (non-tiled)",
        "Dataset YAML": args.data,
        "Model/checkpoint": model_source,
        "Image size": args.imgsz,
        "Batch size": args.batch,
        "Workers": args.workers,
        "Epochs requested": args.epochs,
        "Patience": args.patience,
        "Device": args.device,
        "AMP": args.amp,
        "Optimizer": args.optimizer,
        "Initial learning rate": args.lr0,
        "Weight decay": args.weight_decay,
        "Updated": timestamp(),
        "Status": status,
    }
    with (run_dir / "run_metadata.txt").open("a", encoding="utf-8") as file:
        file.write("\n" + "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_dir = args.project / args.name
    last_checkpoint = run_dir / "weights" / "last.pt"

    print("=" * 72)
    print("ORIGINAL CUBIT-DET — YOLO TRAINING")
    print("=" * 72)
    print(f"PyTorch:      {torch.__version__}")
    print(f"Dataset YAML: {args.data}")
    print(f"Model:        {args.model}")
    print(f"Image size:   {args.imgsz}")
    print(f"Batch size:   {args.batch}")
    print(f"Workers:      {args.workers}")
    print(f"Epochs:       {args.epochs}")
    print(f"Device:       {args.device}")
    print(f"Output:       {run_dir}")
    if args.device != "cpu":
        index = int(args.device) if args.device.isdigit() else 0
        props = torch.cuda.get_device_properties(index)
        print(f"GPU:          {props.name} ({props.total_memory / 1024**3:.2f} GiB)")
    print("=" * 72)

    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
        write_metadata(run_dir, args, "resume_requested", str(last_checkpoint))
        print(f"Resuming from: {last_checkpoint}")
        print("Ultralytics restores the checkpoint's saved optimizer and training arguments.")
        results = model.train(resume=True)
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}\n"
                "Choose another --name, or use --resume to continue this run."
            )
        model = YOLO(args.model)
        if model.task != "detect":
            raise ValueError(f"Expected detection task, got {model.task!r}.")
        write_metadata(run_dir, args, "started", args.model)
        results = model.train(**training_kwargs(args))

    actual_run_dir = Path(results.save_dir)
    write_metadata(actual_run_dir, args, "training_call_finished", args.model)
    print("\nTraining call finished.")
    print(f"Results: {actual_run_dir}")
    print(f"Best:    {actual_run_dir / 'weights' / 'best.pt'}")
    print(f"Last:    {actual_run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
