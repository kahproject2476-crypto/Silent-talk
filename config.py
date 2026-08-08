"""SilentTalk configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
EXTRACTED_DIR = DATASET_DIR / "extracted"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LANDMARKS_DIR = DATA_DIR / "landmarks"
ISL_LANDMARKS_CSV = LANDMARKS_DIR / "isl_landmarks.csv"
AUDIO_DIR = OUTPUT_DIR / "audio"

# Video preprocessing defaults
FRAMES_PER_VIDEO = 12
MAX_VIDEOS_PER_CLASS = 0  # 0 = no limit
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}

# Gesture recognition
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # x, y, z per landmark
SEQUENCE_LENGTH = 30  # frames for LSTM dynamic gestures
CONFIDENCE_THRESHOLD = 0.65
PREDICTION_SMOOTHING = 5  # majority vote over last N predictions

# Default ISL alphabet labels (extend via data collection)
DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Model paths
RF_MODEL_PATH = MODELS_DIR / "isl_random_forest.joblib"
LABELS_PATH = MODELS_DIR / "labels.json"
CNN_LSTM_MODEL_PATH = MODELS_DIR / "isl_cnn_lstm.keras"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"

# Translation: English (recognized ISL text) -> Tulu
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tcy_Telu"
INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"

# Speech
WHISPER_MODEL = "base"
TTS_LANG = "te"  # Telugu/Tulu region TTS via gTTS (closest supported)

# Webcam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 25

# Ensure directories exist
for directory in (
    DATA_DIR,
    DATASET_DIR,
    EXTRACTED_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    LANDMARKS_DIR,
    AUDIO_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
