"""
Test the trained ISL gesture model.

Modes:
  1. Webcam  — live real-time prediction from your camera
  2. Image   — predict gesture from a saved image file
  3. Report  — show model stats and per-class accuracy from the landmark CSV

Usage:
    python scripts/test_model.py --mode webcam
    python scripts/test_model.py --mode image  --image path/to/hand.jpg
    python scripts/test_model.py --mode report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.landmarks import HandLandmarkExtractor
from src.gesture.model import RandomForestGestureClassifier


# ── Shared: load model ────────────────────────────────────────────────────────

def load_model() -> RandomForestGestureClassifier:
    clf = RandomForestGestureClassifier()
    if not clf.load():
        raise SystemExit(
            "No trained model found at models/isl_random_forest.joblib\n"
            "Run: python scripts/train_model.py"
        )
    print(f"Model loaded — {len(clf.labels)} classes: {clf.labels}")
    return clf


# ── Mode 1: Webcam ────────────────────────────────────────────────────────────

def test_webcam(top_k: int = 3) -> None:
    """
    Open webcam and show live predictions.
    Press Q to quit, SPACE to freeze/unfreeze frame.
    """
    clf = load_model()
    extractor = HandLandmarkExtractor()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise SystemExit("Cannot open webcam.")

    print("\nWebcam test started. Press Q to quit.")
    frozen = False
    frozen_frame = None

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = frozen_frame.copy()

        features, annotated = extractor.extract(frame)

        if features is not None and clf.model is not None:
            proba = clf.model.predict_proba(features.reshape(1, -1))[0]
            top_indices = np.argsort(proba)[::-1][:top_k]

            y_offset = 30
            for rank, idx in enumerate(top_indices):
                label = clf.encoder.inverse_transform([idx])[0]
                conf = proba[idx]
                color = (0, 220, 80) if rank == 0 else (180, 180, 180)
                text = f"#{rank+1} {label}: {conf:.0%}"
                cv2.putText(
                    annotated, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
                y_offset += 30

            if frozen:
                cv2.putText(
                    annotated, "FROZEN (SPACE to unfreeze)", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
        else:
            cv2.putText(
                annotated, "No hand detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2
            )

        cv2.imshow("SilentTalk - ISL Test (Q=quit, SPACE=freeze)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


# ── Mode 2: Single image ──────────────────────────────────────────────────────

def test_image(image_path: Path, top_k: int = 5) -> None:
    """Predict gesture from a single image file and print top-K predictions."""
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    clf = load_model()
    extractor = HandLandmarkExtractor()

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    features, annotated = extractor.extract(frame)

    if features is None:
        print("No hand detected in the image.")
    else:
        proba = clf.model.predict_proba(features.reshape(1, -1))[0]
        top_indices = np.argsort(proba)[::-1][:top_k]

        print(f"\nTop-{top_k} predictions for {image_path.name}:")
        print("-" * 35)
        for rank, idx in enumerate(top_indices):
            label = clf.encoder.inverse_transform([idx])[0]
            bar = "█" * int(proba[idx] * 30)
            print(f"  #{rank+1}  {label:<18} {proba[idx]:.1%}  {bar}")

    # Save annotated image
    out_path = config.OUTPUT_DIR / f"test_{image_path.stem}_annotated.jpg"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated image saved: {out_path}")

    extractor.close()


# ── Mode 3: Accuracy report from CSV ─────────────────────────────────────────

def test_report() -> None:
    """Run the model on the landmark CSV and print per-class accuracy."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    if not config.ISL_LANDMARKS_CSV.exists():
        raise SystemExit(f"Landmark CSV not found: {config.ISL_LANDMARKS_CSV}")

    clf = load_model()

    df = pd.read_csv(config.ISL_LANDMARKS_CSV)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_true = df["label"].to_numpy()

    print(f"\nDataset: {len(df)} samples, {df['label'].nunique()} classes")

    y_pred = clf.model.predict(X)
    overall_acc = (y_pred == y_true).mean()

    print(f"\nOverall accuracy: {overall_acc:.2%}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Worst and best classes
    from sklearn.metrics import accuracy_score
    classes = sorted(df["label"].unique())
    per_class = {}
    for cls in classes:
        mask = y_true == cls
        per_class[cls] = accuracy_score(y_true[mask], y_pred[mask])

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
    print("\nWorst 5 classes (need more data):")
    for cls, acc in sorted_classes[:5]:
        n = (y_true == cls).sum()
        print(f"  {cls:<20} {acc:.0%}  ({n} samples)")

    print("\nBest 5 classes:")
    for cls, acc in sorted_classes[-5:]:
        n = (y_true == cls).sum()
        print(f"  {cls:<20} {acc:.0%}  ({n} samples)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test ISL gesture model")
    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "report"],
        default="report",
        help="Test mode: webcam (live), image (single file), report (CSV accuracy)",
    )
    parser.add_argument("--image", type=Path, help="Image path for --mode image")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-K predictions")
    args = parser.parse_args()

    if args.mode == "webcam":
        test_webcam(top_k=args.top_k)
    elif args.mode == "image":
        if not args.image:
            raise SystemExit("--image is required for image mode")
        test_image(args.image, top_k=args.top_k)
    elif args.mode == "report":
        test_report()


if __name__ == "__main__":
    main()
