#!/usr/bin/env python3
"""
Download the official CUBIT-Det dataset from the authors' Google Drive folder.

Run from the UAV-II project root:
    uv run --with gdown python scripts/dataset_download.py

Optional:
    uv run --with gdown python scripts/dataset_download.py --output data/cubit_det
    uv run --with gdown python scripts/dataset_download.py --dry-run

This script only downloads and inventories the source dataset. It does not
rewrite annotations, split data, or modify the existing DACL10K datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_URL = (
    "https://drive.google.com/drive/folders/"
    "1-20JAwuyyd1-Fbpa8f8jHUwtI5B0Frx5?usp=sharing"
)

DEFAULT_OUTPUT = Path("data/cubit_det_raw")

SOURCE_NOTE = """# CUBIT-Det source dataset

- Official source: https://github.com/BenyunZhao/CUBIT
- Google Drive: https://drive.google.com/drive/folders/1-20JAwuyyd1-Fbpa8f8jHUwtI5B0Frx5
- Paper: Benyun Zhao et al., "High-resolution infrastructure defect detection
  dataset sourced by unmanned systems and validated with deep learning",
  Automation in Construction, 163 (2024), 105405.
- DOI: https://doi.org/10.1016/j.autcon.2024.105405
- Dataset: 5,527 UAV images; building, pavement and bridge scenes;
  crack, spalling and moisture; bounding-box annotations.
- License stated by the authors: Creative Commons Attribution 4.0
  International (CC BY 4.0).

Keep the original downloaded files unchanged. Inspect the annotation format,
class IDs, and official train/validation/test split before converting anything
to the project's YOLO training layout. Do not mix this dataset with DACL10K
without reconciling class definitions and split provenance.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official CUBIT-Det UAV defect dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Download directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help="Google Drive shared-folder URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download target without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow downloading into a non-empty output directory.",
    )
    return parser.parse_args()


def inventory(root: Path) -> None:
    files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    print("\nDownloaded file inventory")
    print("-------------------------")
    print(f"Files: {len(files):,}")
    print(f"Total size: {total_bytes / (1024 ** 3):.2f} GiB")

    if not files:
        print("No files found. Google Drive may have blocked the download.")
        return

    print("\nTop-level contents:")
    for item in sorted(root.iterdir()):
        if item.is_dir():
            count = sum(1 for p in item.rglob("*") if p.is_file())
            print(f"  [DIR]  {item.name}/ ({count:,} files)")
        else:
            size = item.stat().st_size
            print(f"  [FILE] {item.name} ({size / (1024 ** 2):.1f} MiB)")

    archives = [
        p for p in files
        if p.suffix.lower() in {".zip", ".7z", ".rar", ".tar", ".gz"}
    ]
    if archives:
        print("\nArchives found (not automatically extracted):")
        for archive in archives:
            print(f"  {archive.relative_to(root)}")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()

    print("CUBIT-Det official dataset downloader")
    print(f"Source: {args.url}")
    print(f"Target: {output}")

    if args.dry_run:
        print("Dry run: no files downloaded.")
        return 0

    if output.exists() and any(output.iterdir()) and not args.force:
        print(
            "\nERROR: The output directory already exists and is not empty:\n"
            f"  {output}\n"
            "Choose another --output directory or pass --force if you "
            "intentionally want to download into it.",
            file=sys.stderr,
        )
        return 2

    try:
        import gdown
    except ImportError:
        print(
            "\nMissing dependency: gdown\n"
            "Run this script with:\n"
            "  uv run --with gdown python scripts/dataset_download.py",
            file=sys.stderr,
        )
        return 2

    output.mkdir(parents=True, exist_ok=True)

    print("\nStarting Google Drive folder download...")
    try:
        downloaded = gdown.download_folder(
            url=args.url,
            output=str(output),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "\nYou can also open the authors' official Google Drive folder "
            "in a browser:\n"
            f"  {args.url}",
            file=sys.stderr,
        )
        return 1

    if not downloaded:
        print(
            "\nGoogle Drive did not return downloadable files. "
            "The folder may require browser access or Google may be "
            "rate-limiting the request.",
            file=sys.stderr,
        )
        print(f"Open the official folder manually: {args.url}")
        return 1

    note_path = output / "DATASET_SOURCE.md"
    note_path.write_text(SOURCE_NOTE, encoding="utf-8")

    inventory(output)
    print(f"\nSource and citation notes saved to: {note_path}")
    print(
        "\nDownload step complete. Do not start training yet; first inspect "
        "the downloaded annotation files and official split structure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
