"""
Train Random Forest gesture classifier from collected landmark CSV files.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --dataset-csv data/landmarks/isl_landmarks.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.model import RandomForestGestureClassifier


def merge_landmark_csvs(data_dir: Path) -> Path:
    csv_files = [
        path
        for path in data_dir.glob("*.csv")
        if path.name not in {"merged_dataset.csv", "isl_landmarks.csv"}
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in {data_dir}. Run: python scripts/collect_gestures.py --label A"
        )

    frames = [pd.read_csv(f) for f in csv_files]
    merged = pd.concat(frames, ignore_index=True)
    merged_path = data_dir / "merged_dataset.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Merged {len(csv_files)} files -> {len(merged)} samples -> {merged_path}")
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISL gesture classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.LANDMARKS_DIR,
        help="Directory containing landmark CSV files",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        help="Use a single preprocessed dataset CSV (from preprocess_videos.py)",
    )
    args = parser.parse_args()

    if args.dataset_csv:
        train_path = args.dataset_csv
        if not train_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {train_path}")
    elif config.ISL_LANDMARKS_CSV.exists():
        train_path = config.ISL_LANDMARKS_CSV
        print(f"Using preprocessed dataset: {train_path}")
    else:
        train_path = merge_landmark_csvs(args.data_dir)

    classifier = RandomForestGestureClassifier()
    metrics = classifier.train_from_csv(train_path)

    print("\n=== Training Complete ===")
    print(f"Samples: {metrics['samples']}")
    print(f"Classes: {len(classifier.labels)}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"\n{metrics['report']}")
    print(f"\nModel saved to: {config.RF_MODEL_PATH}")


if __name__ == "__main__":
    main()
