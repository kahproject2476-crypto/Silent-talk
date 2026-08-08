"""
Collect ISL gesture landmark data from webcam for training.

Usage:
    python scripts/collect_gestures.py --label A
    python scripts/collect_gestures.py --label HELLO --samples 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.landmarks import HandLandmarkExtractor


def collect(label: str, samples: int = 30, delay: float = 0.1) -> Path:
    label = label.upper().strip()
    csv_path = config.LANDMARKS_DIR / f"{label}.csv"
    file_exists = csv_path.exists()

    feature_header = [f"f{i}" for i in range(config.FEATURE_DIM)]
    fieldnames = feature_header + ["label"]

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    collected = 0
    print(f"Collecting {samples} samples for label '{label}'. Press 'q' to quit early.")

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        while collected < samples:
            ret, frame = cap.read()
            if not ret:
                break

            features, annotated = extractor.extract(frame)
            cv2.putText(
                annotated,
                f"Label: {label} | Sample: {collected + 1}/{samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SilentTalk - Data Collection (q to quit)", annotated)

            if features is not None:
                row = {f"f{i}": float(features[i]) for i in range(config.FEATURE_DIM)}
                row["label"] = label
                writer.writerow(row)
                collected += 1
                time.sleep(delay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Saved {collected} samples to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ISL gesture landmark data")
    parser.add_argument("--label", required=True, help="Gesture label (e.g. A, HELLO)")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples")
    parser.add_argument("--delay", type=float, default=0.08, help="Delay between samples")
    args = parser.parse_args()
    collect(args.label, args.samples, args.delay)


if __name__ == "__main__":
    main()
