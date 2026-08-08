"""Hand landmark extraction using MediaPipe Tasks API (mediapipe >= 1.0)."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import config

# Tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _get_model_path() -> str:
    """Return path to hand_landmarker.task, downloading it if absent."""
    model_path: Path = config.HAND_LANDMARKER_MODEL
    if not model_path.exists():
        import urllib.request

        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print(f"Downloading hand_landmarker.task from {url} ...")
        urllib.request.urlretrieve(url, str(model_path))
        print("Download complete.")
    return str(model_path)


# ── Drawing helpers ──────────────────────────────────────────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm
]


def _draw_landmarks(frame_bgr: np.ndarray, landmarks) -> np.ndarray:
    """Draw 21-point skeleton onto a BGR frame."""
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[start], pts[end], (0, 220, 80), 2)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 4, (255, 48, 48), -1)
    return frame_bgr


class HandLandmarkExtractor:
    """Extract normalised 21-point hand landmarks from video frames.

    Public interface is identical to the old mp.solutions version:
        features, annotated_frame = extractor.extract(bgr_frame)
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_model_path()),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """Return (63-d feature vector | None, annotated BGR frame)."""
        annotated = frame_bgr.copy()

        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, annotated

        hand = result.hand_landmarks[0]  # first detected hand
        _draw_landmarks(annotated, hand)

        coords: list[float] = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])

        feature = np.array(coords, dtype=np.float32)
        if feature.shape[0] != config.FEATURE_DIM:
            return None, annotated

        return feature, annotated

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
