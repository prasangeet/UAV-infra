#!/usr/bin/env python3
"""Safely verify and extract the six CUBIT-Det ZIP archives.

Expected source: data/cubit_det_raw/{images,labels}/{train2017,val2017,test2017}.zip
Default output: data/cubit_det/

Run from project root:
    uv run python scripts/extract_cubit_det.py

Options:
    --verify-only   Test archive integrity without extracting.
    --dry-run       Show extraction targets without writing files.
    --overwrite     Permit replacing existing extracted files.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "cubit_det_raw"
DEFAULT_OUTPUT = ROOT / "data" / "cubit_det"
GROUPS = ("images", "labels")
SPLITS = ("train2017", "val2017", "test2017")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_path(root: Path, name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    target = (root / Path(*member.parts)).resolve()
    base = root.resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"ZIP path escapes destination: {name!r}")
    return target


def verify_zip(path: Path):
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise zipfile.BadZipFile(f"CRC failure in member {bad!r}")
            for item in archive.infolist():
                safe_path(Path("/tmp/cubit-det-validation"), item.filename)
            count = len(archive.infolist())
            unpacked = sum(item.file_size for item in archive.infolist())
            return count, unpacked
    except (OSError, zipfile.BadZipFile, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"{path.name}: {exc}") from exc


def extract_zip(path: Path, destination: Path, overwrite: bool):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            target = safe_path(destination, item.filename)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                if target.stat().st_size == item.file_size:
                    continue
                raise FileExistsError(
                    f"Existing file has a different size: {target}\n"
                    "Use --overwrite only if replacement is intended."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, target.open("wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    print("CUBIT-Det archive extractor")
    print(f"Source: {source}")
    print(f"Output: {output}")

    if not source.is_dir():
        print(f"ERROR: source directory not found: {source}", file=sys.stderr)
        return 2

    jobs = []
    for group in GROUPS:
        for split in SPLITS:
            archive = source / group / f"{split}.zip"
            if not archive.is_file():
                print(f"ERROR: missing archive: {archive}", file=sys.stderr)
                return 2
            jobs.append((group, split, archive, output / group / split))

    if output.exists() and any(output.iterdir()) and not (
        args.overwrite or args.verify_only or args.dry_run
    ):
        print(
            "ERROR: output directory is non-empty. Choose another --output "
            "or pass --overwrite intentionally.",
            file=sys.stderr,
        )
        return 2

    print("\nVerifying all six ZIP files before extraction...")
    verified = []
    for group, split, archive, destination in jobs:
        print(f"\n[{group}/{split}] {archive.stat().st_size / 1024**3:.2f} GiB")
        try:
            count, unpacked = verify_zip(archive)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print("No extraction has been started.")
            return 1
        print(f"PASS: {count:,} entries; {unpacked / 1024**3:.2f} GiB uncompressed")
        verified.append((group, split, archive, destination))

    if args.verify_only:
        print("\nAll archives passed. Verification only; nothing extracted.")
        return 0

    if args.dry_run:
        print("\nDry run; planned targets:")
        for group, split, archive, destination in verified:
            print(f"  {archive} -> {destination}")
        return 0

    for group, split, archive, destination in verified:
        print(f"\nExtracting {group}/{split} -> {destination}")
        try:
            extract_zip(archive, destination, args.overwrite)
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            print(f"Extraction stopped: {exc}", file=sys.stderr)
            print("Previously extracted files have been retained.")
            return 1

    print("\nExtraction complete. Original ZIP files were not modified.")
    print(f"Dataset directory: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
