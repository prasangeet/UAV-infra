from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "cubit_det"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cubit_det_tiled"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = {0: "Crack", 1: "Spalling", 2: "Moisture"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a YOLO detection dataset from native-resolution, "
            "overlapping 640x640 tiles."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument(
        "--min-visible", type=float, default=0.30,
        help="Keep an object in a tile if at least this fraction of its area is visible."
    )
    parser.add_argument(
        "--negative-per-image", type=int, default=2,
        help="Maximum number of randomly selected background tiles per source image."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete and recreate the output directory if it already exists."
    )
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def find_split_dirs(source: Path, split: str) -> tuple[Path, Path]:
    image_dir = source / "images" / f"{split}2017" / f"{split}2017"
    label_dir = source / "labels" / f"{split}2017" / f"{split}2017"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image and label directories for '{split}':\n"
            f"  {image_dir}\n  {label_dir}"
        )
    return image_dir, label_dir


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    objects = []
    if not path.exists():
        return objects

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            continue

        x1 = max(0.0, (xc - bw / 2) * width)
        y1 = max(0.0, (yc - bh / 2) * height)
        x2 = min(float(width), (xc + bw / 2) * width)
        y2 = min(float(height), (yc + bh / 2) * height)
        if x2 > x1 and y2 > y1:
            objects.append((class_id, x1, y1, x2, y2))

    return objects


def labels_for_tile(
    objects: list[tuple[int, float, float, float, float]],
    left: int,
    top: int,
    tile_width: int,
    tile_height: int,
    min_visible: float,
) -> list[tuple[int, float, float, float, float]]:
    tile_right = left + tile_width
    tile_bottom = top + tile_height
    tile_labels = []

    for class_id, x1, y1, x2, y2 in objects:
        ix1 = max(x1, left)
        iy1 = max(y1, top)
        ix2 = min(x2, tile_right)
        iy2 = min(y2, tile_bottom)

        intersection_width = max(0.0, ix2 - ix1)
        intersection_height = max(0.0, iy2 - iy1)
        intersection_area = intersection_width * intersection_height
        original_area = (x2 - x1) * (y2 - y1)

        if original_area <= 0 or intersection_area / original_area < min_visible:
            continue

        # Coordinates are normalized relative to the tile, not the source image.
        xc = ((ix1 + ix2) / 2 - left) / tile_width
        yc = ((iy1 + iy2) / 2 - top) / tile_height
        bw = intersection_width / tile_width
        bh = intersection_height / tile_height
        tile_labels.append((class_id, xc, yc, bw, bh))

    return tile_labels


def write_labels(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for class_id, xc, yc, bw, bh in labels:
            file.write(
                f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n"
            )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if args.tile_size < 64:
        raise ValueError("--tile-size must be at least 64.")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1).")
    if not 0 < args.min_visible <= 1:
        raise ValueError("--min-visible must be in (0, 1].")
    if args.negative_per_image < 0:
        raise ValueError("--negative-per-image cannot be negative.")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Use --overwrite only if you intentionally want to rebuild it."
            )
        import shutil
        shutil.rmtree(output)

    stride = max(1, int(round(args.tile_size * (1 - args.overlap))))
    rng = random.Random(args.seed)

    print("=" * 72)
    print("CUBIT-DET NATIVE-RESOLUTION TILING")
    print("=" * 72)
    print(f"Source:           {source}")
    print(f"Output:           {output}")
    print(f"Tile size:        {args.tile_size} x {args.tile_size}")
    print(f"Overlap:          {args.overlap:.0%}")
    print(f"Stride:           {stride} px")
    print(f"Minimum visible:  {args.min_visible:.0%}")
    print(f"Negative tiles:   up to {args.negative_per_image} per source image")
    print("=" * 72)

    totals = {
        split: {
            "source_images": 0,
            "tiles": 0,
            "positive_tiles": 0,
            "negative_tiles": 0,
            "objects": Counter(),
            "discarded_images_without_tiles": 0,
        }
        for split in SPLITS
    }

    for split in SPLITS:
        image_dir, label_dir = find_split_dirs(source, split)
        out_images = output / "images" / split
        out_labels = output / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        stats = totals[split]
        stats["source_images"] = len(image_paths)

        print(f"\n[{split.upper()}] Processing {len(image_paths)} source images...")

        for image_number, image_path in enumerate(image_paths, 1):
            label_path = label_dir / f"{image_path.stem}.txt"

            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
                width, height = image.size
                objects = read_yolo_labels(label_path, width, height)

                xs = tile_starts(width, args.tile_size, stride)
                ys = tile_starts(height, args.tile_size, stride)

                positive_tiles = []
                negative_tiles = []

                for top in ys:
                    for left in xs:
                        tile_width = min(args.tile_size, width - left)
                        tile_height = min(args.tile_size, height - top)

                        # Skip undersized edge crops; edge-anchored starts normally
                        # ensure full-sized tiles for images larger than tile_size.
                        if tile_width != args.tile_size or tile_height != args.tile_size:
                            continue

                        tile_labels = labels_for_tile(
                            objects, left, top, tile_width, tile_height,
                            args.min_visible
                        )
                        record = (left, top, tile_labels)
                        if tile_labels:
                            positive_tiles.append(record)
                        else:
                            negative_tiles.append(record)

                if args.negative_per_image and negative_tiles:
                    selected_negatives = rng.sample(
                        negative_tiles,
                        k=min(args.negative_per_image, len(negative_tiles))
                    )
                else:
                    selected_negatives = []

                selected_tiles = positive_tiles + selected_negatives
                if not selected_tiles:
                    stats["discarded_images_without_tiles"] += 1
                    continue

                for left, top, tile_labels in selected_tiles:
                    tile_name = f"{image_path.stem}__x{left:05d}_y{top:05d}"
                    tile_image_path = out_images / f"{tile_name}.jpg"
                    tile_label_path = out_labels / f"{tile_name}.txt"

                    tile = image.crop((
                        left, top, left + args.tile_size, top + args.tile_size
                    ))
                    tile.save(tile_image_path, format="JPEG", quality=95)
                    write_labels(tile_label_path, tile_labels)

                    stats["tiles"] += 1
                    if tile_labels:
                        stats["positive_tiles"] += 1
                        for label in tile_labels:
                            stats["objects"][label[0]] += 1
                    else:
                        stats["negative_tiles"] += 1

            if image_number % 250 == 0 or image_number == len(image_paths):
                print(f"  {image_number}/{len(image_paths)} images processed")

        print(
            f"  Tiles: {stats['tiles']} "
            f"(positive={stats['positive_tiles']}, "
            f"negative={stats['negative_tiles']})"
        )
        print(f"  Retained objects: {sum(stats['objects'].values())}")

    yaml_lines = [
        f"path: {output.as_posix()}",
        "",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 3",
        "names:",
        "  0: Crack",
        "  1: Spalling",
        "  2: Moisture",
        "",
    ]
    (output / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    report_lines = [
        "CUBIT-DET TILING REPORT",
        "=" * 48,
        f"Source: {source}",
        f"Tile size: {args.tile_size}",
        f"Overlap: {args.overlap:.0%}",
        f"Stride: {stride}",
        f"Minimum visible fraction: {args.min_visible:.0%}",
        f"Negative tiles per source image (maximum): {args.negative_per_image}",
        "",
    ]

    for split in SPLITS:
        stats = totals[split]
        report_lines.extend([
            f"[{split.upper()}]",
            f"Source images: {stats['source_images']}",
            f"Total tiles: {stats['tiles']}",
            f"Positive tiles: {stats['positive_tiles']}",
            f"Negative tiles: {stats['negative_tiles']}",
            f"Retained object instances: {sum(stats['objects'].values())}",
        ])
        for class_id, class_name in CLASS_NAMES.items():
            report_lines.append(
                f"  {class_name}: {stats['objects'].get(class_id, 0)}"
            )
        report_lines.append("")

    (output / "tiling_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\nTiling complete.")
    print(f"Dataset YAML: {output / 'data.yaml'}")
    print(f"Report:       {output / 'tiling_report.txt'}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()
