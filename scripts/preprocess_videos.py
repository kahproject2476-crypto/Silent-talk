"""
Extract MediaPipe hand landmarks from extracted ISL videos.

Usage:
    python scripts/preprocess_videos.py
    python scripts/preprocess_videos.py --category Greetings --max-videos 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataset.label_utils import normalize_label, parse_gesture_label
from src.gesture.landmarks import HandLandmarkExtractor


def iter_videos(root: Path, category: str | None = None) -> list[tuple[Path, str, str]]:
    videos: list[tuple[Path, str, str]] = []
    if not root.exists():
        return videos

    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if category and category_dir.name.lower() != category.lower():
            continue

        for label_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            label = normalize_label(parse_gesture_label(label_dir.name))
            for video_path in sorted(label_dir.iterdir()):
                if video_path.suffix.lower() in config.VIDEO_EXTENSIONS:
                    videos.append((video_path, label, category_dir.name))
    return videos


def sample_frames(cap: cv2.VideoCapture, target: int) -> list:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return []

    if total <= target:
        indices = list(range(total))
    else:
        step = max(total // target, 1)
        indices = list(range(0, total, step))[:target]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def preprocess(
    extracted_dir: Path,
    output_csv: Path,
    category: str | None = None,
    frames_per_video: int = config.FRAMES_PER_VIDEO,
    max_videos_per_class: int = config.MAX_VIDEOS_PER_CLASS,
) -> dict:
    videos = iter_videos(extracted_dir, category=category)
    if not videos:
        raise FileNotFoundError(
            f"No videos found in {extracted_dir}. Run: python scripts/extract_dataset.py"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_header = [f"f{i}" for i in range(config.FEATURE_DIM)]
    fieldnames = feature_header + ["label", "category", "video"]

    per_class_count: dict[str, int] = {}
    rows_written = 0
    skipped_videos = 0

    extractor = HandLandmarkExtractor()
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for video_path, label, cat in videos:
            if max_videos_per_class > 0:
                count = per_class_count.get(label, 0)
                if count >= max_videos_per_class:
                    continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped_videos += 1
                continue

            frames = sample_frames(cap, frames_per_video)
            cap.release()

            frame_hits = 0
            for frame in frames:
                features, _ = extractor.extract(frame)
                if features is None:
                    continue

                row = {f"f{i}": float(features[i]) for i in range(config.FEATURE_DIM)}
                row["label"] = label
                row["category"] = cat
                row["video"] = video_path.name
                writer.writerow(row)
                rows_written += 1
                frame_hits += 1

            if frame_hits:
                per_class_count[label] = per_class_count.get(label, 0) + 1
            else:
                skipped_videos += 1

    extractor.close()
    return {
        "rows": rows_written,
        "classes": len(per_class_count),
        "videos_used": sum(per_class_count.values()),
        "skipped_videos": skipped_videos,
        "output": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ISL videos to landmarks")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.EXTRACTED_DIR,
        help="Extracted dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.ISL_LANDMARKS_CSV,
        help="Output CSV path",
    )
    parser.add_argument("--category", help="Process only one category (e.g. Greetings)")
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=config.FRAMES_PER_VIDEO,
        help="Number of frames sampled per video",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=config.MAX_VIDEOS_PER_CLASS,
        help="Limit videos per class (0 = all)",
    )
    args = parser.parse_args()

    stats = preprocess(
        extracted_dir=args.input_dir,
        output_csv=args.output,
        category=args.category,
        frames_per_video=args.frames_per_video,
        max_videos_per_class=args.max_videos_per_class,
    )

    print("\n=== Preprocessing Complete ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
