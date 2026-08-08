# SilentTalk

**Two-Way Communication Using Indian Sign Language Recognition**

An AI-powered real-time communication system that recognizes Indian Sign Language (ISL) gestures via webcam, translates them to Tulu text/speech, and supports two-way communication through speech-to-text.

**Team:** Akshitha, Hardhika V Shetty, Kavya B  
**Guide:** Mr. Kiran Ankalakoti  
**Institution:** Canara Engineering College, Mangaluru | AY 2025-26

---

## Features

| Module | Technology | Description |
|--------|-----------|-------------|
| Gesture Recognition | MediaPipe + Random Forest / CNN-LSTM | Real-time ISL hand gesture detection |
| Translation | IndicTrans2 | English → Tulu text translation |
| Text-to-Speech | gTTS | Tulu speech output |
| Speech-to-Text | OpenAI Whisper | Two-way communication for hearing users |
| UI | Streamlit | Live webcam feed and controls |

## Project Structure

```
silenttalk/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── src/
│   ├── gesture/            # Landmark extraction & classification
│   ├── translation/        # IndicTrans2 Tulu translation
│   ├── speech/             # TTS (gTTS) & STT (Whisper)
│   └── utils/              # Helpers
├── scripts/
│   ├── collect_gestures.py # Webcam data collection
│   ├── train_model.py      # Train Random Forest model
│   └── generate_demo_model.py
├── data/landmarks/         # Collected gesture CSV data
├── models/                 # Trained model files
└── output/                 # Generated audio & logs
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Quick start (demo model)

```bash
python scripts/generate_demo_model.py
streamlit run app.py
```

## Dataset (ISL Videos)

The `dataset/` folder contains ISL video archives:

| Archive | Status |
|---------|--------|
| Days_and_Time_1of3, 2of3 | Valid |
| Greetings_1of2 | Valid |
| People_1of5, 2of5, 4of5, 5of5 | Valid |
| Pronouns_1of2 | Valid |
| Days_and_Time_3of3, Greetings_2of2, People_3of5, Pronouns_2of2 | Corrupt — re-download |

Categories include **Greetings**, **People**, **Pronouns**, and **Days_and_Time** with labels like `Hello`, `Good Morning`, `Mother`, etc.

### Full build from dataset zips

```bash
python scripts/build_project.py
```

Or step by step:

```bash
python scripts/extract_dataset.py --list
python scripts/extract_dataset.py
python scripts/preprocess_videos.py
python scripts/train_model.py
streamlit run app.py
```

## Usage

1. **Sign → Text → Tulu → Speech**
   - Start the camera and show ISL gestures
   - Click **Add to Sentence** to build text
   - Click **Translate to Tulu & Speak** for translation and audio

2. **Speech → Text (Whisper)**
   - Upload an audio file
   - Click **Transcribe with Whisper** for speech-to-text

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Webcam    │───▶│  MediaPipe   │───▶│ CNN/LSTM or │───▶│ English  │
│   Input     │    │  Landmarks   │    │ RandomForest│    │   Text   │
└─────────────┘    └──────────────┘    └─────────────┘    └────┬─────┘
                                                               │
                    ┌──────────────┐    ┌─────────────┐        │
                    │    gTTS      │◀───│ IndicTrans2 │◀───────┘
                    │  Tulu Speech │    │  EN → Tulu  │
                    └──────────────┘    └─────────────┘

┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Microphone  │───▶│   Whisper    │───▶│    Text     │
│   Audio     │    │     STT      │    │   Output    │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Requirements

- Python 3.10+
- Webcam
- Windows/Linux
- ~4 GB RAM (Whisper base model)
- GPU optional (recommended for IndicTrans2)

## Future Work (Phase-II)

- Expand ISL gesture vocabulary with INCLUDE dataset
- Train CNN-LSTM for dynamic gestures
- Sentence-level NLP and grammar correction
- Mobile app and cloud deployment

## References

See Phase-I report for full literature survey and SRS documentation.
