"""
Generate synthetic demo training data for quick testing.

Creates a small Random Forest model with simulated landmark variations
for letters A, B, C so the app can run without manual data collection.

Usage:
    python scripts/generate_demo_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.model import RandomForestGestureClassifier


def generate_demo_dataset(labels: list[str], samples_per_label: int = 40) -> Path:
    rows = []
    rng = np.random.default_rng(42)

    for idx, label in enumerate(labels):
        base = rng.uniform(0.2, 0.8, config.FEATURE_DIM)
        base[idx % config.FEATURE_DIM] += 0.3

        for _ in range(samples_per_label):
            noise = rng.normal(0, 0.02, config.FEATURE_DIM)
            sample = np.clip(base + noise, 0, 1)
            row = {f"f{i}": float(sample[i]) for i in range(config.FEATURE_DIM)}
            row["label"] = label
            rows.append(row)

    df = pd.DataFrame(rows)
    path = config.LANDMARKS_DIR / "demo_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    labels = ["A", "B", "C", "HELLO", "THANKYOU", "YES", "NO"]
    print("Generating demo dataset...")
    csv_path = generate_demo_dataset(labels)
    print(f"Demo data: {csv_path}")

    classifier = RandomForestGestureClassifier()
    metrics = classifier.train_from_csv(csv_path)
    print(f"Demo model accuracy: {metrics['accuracy']:.2%}")
    print(f"Model saved: {config.RF_MODEL_PATH}")
    print("\nRun the app: streamlit run app.py")


if __name__ == "__main__":
    main()
