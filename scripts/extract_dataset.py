"""
Extract valid ISL dataset zip archives into dataset/extracted/.

Usage:
    python scripts/extract_dataset.py
    python scripts/extract_dataset.py --zip Greetings_1of2.zip
    python scripts/extract_dataset.py --list
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataset.zip_utils import list_archives, valid_archives


def extract_archive(zip_path: Path, output_dir: Path, force: bool = False) -> int:
    marker = output_dir / f".extracted_{zip_path.stem}"
    if marker.exists() and not force:
        print(f"Skip (already extracted): {zip_path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} ...")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    marker.write_text("ok", encoding="utf-8")
    videos = sum(
        1
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in config.VIDEO_EXTENSIONS
    )
    print(f"  Done -> {zip_path.name} ({videos} video files visible under extracted/)")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ISL dataset zip files")
    parser.add_argument("--zip", help="Extract only this zip file name")
    parser.add_argument("--force", action="store_true", help="Re-extract even if marker exists")
    parser.add_argument("--list", action="store_true", help="List archive status only")
    args = parser.parse_args()

    archives = list_archives(config.DATASET_DIR)
    if args.list:
        print("\n=== Dataset Archive Status ===")
        for info in archives:
            status = "OK" if info.valid else f"CORRUPT ({info.error})"
            print(
                f"{info.path.name:28} {status:30} "
                f"videos={info.video_count:4} categories={','.join(info.categories)}"
            )
        valid = [a for a in archives if a.valid]
        corrupt = [a for a in archives if not a.valid]
        print(f"\nValid: {len(valid)} | Corrupt/incomplete: {len(corrupt)}")
        if corrupt:
            print("Corrupt files (re-download needed):")
            for item in corrupt:
                print(f"  - {item.path.name}")
        return

    targets = valid_archives(config.DATASET_DIR)
    if args.zip:
        targets = [info for info in targets if info.path.name == args.zip]
        if not targets:
            raise SystemExit(f"Zip not found or invalid: {args.zip}")

    if not targets:
        raise SystemExit("No valid zip archives found in dataset/")

    extracted_count = 0
    for info in targets:
        extracted_count += extract_archive(info.path, config.EXTRACTED_DIR, force=args.force)

    print(f"\nExtracted {extracted_count} archive(s) to {config.EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
