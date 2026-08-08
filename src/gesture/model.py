"""Gesture classification models: Random Forest and CNN-LSTM."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import config


def save_labels(labels: list[str], path: Path | None = None) -> None:
    path = path or config.LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path: Path | None = None) -> list[str]:
    path = path or config.LABELS_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return config.DEFAULT_LABELS.copy()


class RandomForestGestureClassifier:
    """Lightweight real-time ISL gesture classifier (Phase-II baseline)."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.encoder = LabelEncoder()
        self.labels: list[str] = load_labels()
        self.encoder.fit(self.labels)

    def train_from_csv(self, csv_path: Path) -> dict:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column")

        feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in {"label", "category", "video"}]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy()

        self.encoder.fit(y)
        self.labels = list(self.encoder.classes_)
        save_labels(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "encoder": self.encoder, "labels": self.labels},
            config.RF_MODEL_PATH,
        )

        return {"accuracy": acc, "report": report, "samples": len(df)}

    def load(self, path: Path | None = None) -> bool:
        path = path or config.RF_MODEL_PATH
        if not path.exists():
            return False
        payload = joblib.load(path)
        self.model = payload["model"]
        self.encoder = payload["encoder"]
        self.labels = payload.get("labels", load_labels())
        return True

    def predict(self, features: np.ndarray) -> tuple[str | None, float]:
        if self.model is None:
            return None, 0.0
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        if confidence < config.CONFIDENCE_THRESHOLD:
            return None, confidence
        label = self.encoder.inverse_transform([idx])[0]
        return str(label), confidence


def build_cnn_lstm_model(num_classes: int, sequence_length: int = config.SEQUENCE_LENGTH):
    """Build CNN-LSTM hybrid architecture per project specification."""
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, config.FEATURE_DIM)),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
