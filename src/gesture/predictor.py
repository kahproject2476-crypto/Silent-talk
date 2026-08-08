"""Real-time gesture prediction with smoothing."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

import config
from src.gesture.landmarks import HandLandmarkExtractor
from src.gesture.model import RandomForestGestureClassifier


class GesturePredictor:
    """Pipeline: webcam frame -> landmarks -> classified gesture."""

    def __init__(self) -> None:
        self.extractor = HandLandmarkExtractor()
        self.classifier = RandomForestGestureClassifier()
        self._history: deque[str] = deque(maxlen=config.PREDICTION_SMOOTHING)
        self.model_loaded = self.classifier.load()

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        features, annotated = self.extractor.extract(frame_bgr)
        result = {
            "annotated_frame": annotated,
            "gesture": None,
            "confidence": 0.0,
            "features": features,
            "model_loaded": self.model_loaded,
        }

        if features is None or not self.model_loaded:
            return result

        label, confidence = self.classifier.predict(features)
        if label:
            self._history.append(label)
            smoothed = Counter(self._history).most_common(1)[0][0]
            result["gesture"] = smoothed
            result["confidence"] = confidence

        return result

    def reset_history(self) -> None:
        self._history.clear()

    def close(self) -> None:
        self.extractor.close()
