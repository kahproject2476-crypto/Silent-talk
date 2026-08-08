"""
Train CNN-LSTM model for dynamic ISL gesture recognition.

The CNN-LSTM architecture processes sequences of hand landmarks (N frames per
video), making it suitable for motion-based gestures unlike the frame-level
Random Forest classifier.

Usage:
    python scripts/train_cnn_lstm.py
    python scripts/train_cnn_lstm.py --sequence-length 30 --epochs 50
    python scripts/train_cnn_lstm.py --dataset-csv data/landmarks/isl_landmarks.csv

Pipeline:
    isl_landmarks.csv  ->  group rows by (label, video)  ->  pad/trim to fixed
    sequence length  ->  train CNN-LSTM  ->  models/isl_cnn_lstm.keras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.model import build_cnn_lstm_model


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_sequences(
    csv_path: Path,
    sequence_length: int = config.SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load landmark CSV and convert to fixed-length sequences.

    Each unique (video, label) pair becomes one sequence. Sequences shorter
    than *sequence_length* are zero-padded; longer ones are uniformly sampled.

    Returns:
        X: float32 array of shape (N, sequence_length, FEATURE_DIM)
        y: int32 label indices of shape (N,)
        classes: sorted list of class names
    """
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in {"label", "category", "video"}]

    if "video" not in df.columns:
        # No video column — treat each consecutive block of rows as one sequence
        df["video"] = (df["label"] != df["label"].shift()).cumsum().astype(str)

    encoder = LabelEncoder()
    encoder.fit(df["label"].unique())
    classes: list[str] = list(encoder.classes_)

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for (label, video), group in df.groupby(["label", "video"], sort=False):
        frames = group[feature_cols].values.astype(np.float32)
        seq = _pad_or_sample(frames, sequence_length)
        sequences.append(seq)
        labels.append(int(encoder.transform([label])[0]))

    X = np.stack(sequences, axis=0)          # (N, seq_len, feat_dim)
    y = np.array(labels, dtype=np.int32)     # (N,)

    print(
        f"Sequences: {len(X)}  |  Classes: {len(classes)}  |  "
        f"Shape: {X.shape}  |  Labels: {classes}"
    )
    return X, y, classes


def _pad_or_sample(frames: np.ndarray, target: int) -> np.ndarray:
    """Return an array of exactly *target* frames."""
    n = len(frames)
    if n == target:
        return frames
    if n > target:
        # Uniform sampling
        indices = np.linspace(0, n - 1, target, dtype=int)
        return frames[indices]
    # Zero-pad at the end
    pad = np.zeros((target - n, frames.shape[1]), dtype=np.float32)
    return np.vstack([frames, pad])


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    csv_path: Path,
    sequence_length: int = config.SEQUENCE_LENGTH,
    epochs: int = 50,
    batch_size: int = 32,
    test_size: float = 0.2,
    patience: int = 10,
) -> dict:
    X, y, classes = load_sequences(csv_path, sequence_length)

    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes, found: {classes}")

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    # Build & summarise model
    model = build_cnn_lstm_model(
        num_classes=len(classes),
        sequence_length=sequence_length,
    )
    model.summary()

    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(config.CNN_LSTM_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on val set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Save labels alongside the model
    labels_path = config.MODELS_DIR / "cnn_lstm_labels.json"
    labels_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    return {
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "classes": len(classes),
        "sequences": len(X),
        "epochs_run": len(history.history["loss"]),
        "model_path": str(config.CNN_LSTM_MODEL_PATH),
        "labels_path": str(labels_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM model for dynamic ISL gesture recognition"
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=config.ISL_LANDMARKS_CSV,
        help="Path to landmark CSV (default: data/landmarks/isl_landmarks.csv)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=config.SEQUENCE_LENGTH,
        help=f"Frames per sequence (default: {config.SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    args = parser.parse_args()

    if not args.dataset_csv.exists():
        raise SystemExit(
            f"Dataset CSV not found: {args.dataset_csv}\n"
            "Run first:  python scripts/preprocess_videos.py"
        )

    print(f"\n=== CNN-LSTM Training ===")
    print(f"Dataset : {args.dataset_csv}")
    print(f"Seq len : {args.sequence_length}")
    print(f"Epochs  : {args.epochs}")
    print(f"Batch   : {args.batch_size}\n")

    results = train(
        csv_path=args.dataset_csv,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    print("\n=== Training Complete ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
