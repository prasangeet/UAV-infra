#!/usr/bin/env python3
"""Visualize the CUBIT-Det tiled YOLO dataset."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "cubit_det_tiled"
DEFAULT_OUTPUT = ROOT / "reports" / "cubit_det_tiled_visualization"
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "Crack", 1: "Spalling", 2: "Moisture"}


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Sample:
    image_path: Path
    boxes: list[Box]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def read_labels(path: Path, width: int, height: int):
    boxes = []
    invalid = 0
    if not path.is_file():
        return boxes, invalid

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            raw_class = float(parts[0])
            class_id = int(raw_class)
            xc, yc, bw, bh = map(float, parts[1:])
        except (ValueError, OverflowError):
            invalid += 1
            continue

        if (
            raw_class != class_id
            or class_id not in CLASS_NAMES
            or not all(v == v and abs(v) != float("inf") for v in (xc, yc, bw, bh))
            or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1)
        ):
            invalid += 1
            continue

        x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
        x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
        if x1 < -0.001 or y1 < -0.001 or x2 > width + 0.001 or y2 > height + 0.001:
            invalid += 1
            continue
        if x2 <= x1 or y2 <= y1:
            invalid += 1
            continue

        boxes.append(Box(class_id, max(0, x1), max(0, y1),
                         min(width, x2), min(height, y2)))
    return boxes, invalid


def load_split(data_root: Path, split: str):
    image_root = data_root / "images" / split
    label_root = data_root / "labels" / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    paths = sorted(
        p for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    samples = []
    invalid = 0
    missing_labels = 0
    counts = Counter()

    for image_path in paths:
        label_path = label_root / f"{image_path.stem}.txt"
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except OSError:
            invalid += 1
            continue

        if label_path.is_file():
            boxes, bad = read_labels(label_path, width, height)
            invalid += bad
        else:
            boxes = []
            missing_labels += 1

        counts.update(box.class_id for box in boxes)
        samples.append(Sample(image_path, boxes))

    return {
        "samples": samples,
        "invalid": invalid,
        "missing_labels": missing_labels,
        "counts": counts,
    }


def plot_contact_sheet(split, samples, output, sample_count, rng, show):
    if not samples:
        print(f"No images found for {split}.")
        return

    chosen = rng.sample(samples, min(sample_count, len(samples)))
    columns = 4
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.5, rows * 4.2),
                             squeeze=False)

    for ax, sample in zip(axes.flat, chosen):
        try:
            with Image.open(sample.image_path) as im:
                ax.imshow(im.convert("RGB"))
            for box in sample.boxes:
                ax.add_patch(Rectangle(
                    (box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1,
                    fill=False, linewidth=1.5, edgecolor="lime"
                ))
                ax.text(
                    box.x1, max(0, box.y1 - 3), CLASS_NAMES[box.class_id],
                    color="black", fontsize=7,
                    bbox={"facecolor": "lime", "alpha": 0.8, "pad": 1},
                )
            ax.set_title(f"{sample.image_path.name}\n{len(sample.boxes)} boxes",
                         fontsize=8)
            ax.axis("off")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Could not open image:\n{exc}", ha="center")
            ax.axis("off")

    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")

    fig.suptitle(f"CUBIT-Det tiled dataset — {split}")
    fig.tight_layout()
    destination = output / f"{split}_contact_sheet.png"
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved contact sheet: {destination}")


def plot_distributions(results, output):
    total = Counter()
    for result in results.values():
        total.update(result["counts"])

    names = [CLASS_NAMES[i] for i in CLASS_NAMES]
    values = [total[i] for i in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, values)
    ax.set_title("CUBIT-Det tiled dataset — object instances by class")
    ax.set_ylabel("Bounding-box instances")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    split_names = list(results)
    split_values = [sum(results[s]["counts"].values()) for s in split_names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(split_names, split_values)
    ax.set_title("CUBIT-Det tiled dataset — object instances by split")
    ax.set_ylabel("Bounding-box instances")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, split_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "split_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(results, output):
    lines = [
        "# CUBIT-Det tiled dataset visualization report",
        "",
        "| Split | Tiles | Tiles with boxes | Boxes | Invalid label rows | Missing label files |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total = Counter()
    for split, result in results.items():
        samples = result["samples"]
        boxed = sum(bool(s.boxes) for s in samples)
        boxes = sum(len(s.boxes) for s in samples)
        total.update(result["counts"])
        lines.append(
            f"| {split} | {len(samples):,} | {boxed:,} | {boxes:,} "
            f"| {result['invalid']:,} | {result['missing_labels']:,} |"
        )

    lines += ["", "## Class distribution", "", "| Class | Instances |",
              "|---|---:|"]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"| {name} | {total[class_id]:,} |")
    lines += [
        "", "## Generated figures", "",
        "- `class_distribution.png`: object instances by class.",
        "- `split_distribution.png`: object instances by split.",
        "- `<split>_contact_sheet.png`: sampled tiles with bounding boxes.",
        "", "This is an exploratory visualization, not a complete annotation audit.",
        "Inspect clipped boxes and tile boundaries before training.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1.")
    if not data_root.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_root}")
    output.mkdir(parents=True, exist_ok=True)

    splits = SPLITS if args.split == "all" else (args.split,)
    results = {}
    rng = random.Random(args.seed)

    print("=" * 68)
    print("CUBIT-DET TILED DATASET VISUALIZATION")
    print("=" * 68)
    print(f"Dataset: {data_root}")
    print(f"Output:  {output}")

    for split in splits:
        print(f"\nInspecting {split}...")
        result = load_split(data_root, split)
        results[split] = result
        samples = result["samples"]
        total_boxes = sum(len(s.boxes) for s in samples)
        with_boxes = sum(bool(s.boxes) for s in samples)

        print(f"  Tiles:               {len(samples):,}")
        print(f"  Tiles with boxes:    {with_boxes:,}")
        print(f"  Background tiles:    {len(samples) - with_boxes:,}")
        print(f"  Valid boxes:         {total_boxes:,}")
        print(f"  Invalid label rows:  {result['invalid']:,}")
        print(f"  Missing label files: {result['missing_labels']:,}")
        for class_id, class_name in CLASS_NAMES.items():
            print(f"  {class_name:<12}          {result['counts'][class_id]:,}")

        plot_contact_sheet(split, samples, output, args.samples, rng, args.show)

    plot_distributions(results, output)
    write_report(results, output)
    print("\nVisualization complete.")
    print(f"Report: {output / 'report.md'}")
    print(f"Figures: {output}")


if __name__ == "__main__":
    main()
