# -*- coding: utf-8 -*-
"""
SilentTalk — FastAPI backend
Endpoints:
  GET  /                        → serves frontend/index.html
  WS   /ws/gesture              → webcam frame → landmarks → prediction
  POST /api/translate           → English text → Tulu text + audio URL
  POST /api/transcribe          → audio file  → transcript (Whisper)
  GET  /api/status              → model / system status
  GET  /audio/{filename}        → serve generated audio files
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.predictor import GesturePredictor
from src.speech.tts import TextToSpeech
from src.speech.stt import SpeechToText
from src.translation.indic_translator import TuluTranslator
from src.utils.helpers import append_conversation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("silenttalk")

# Thread pool for blocking ML calls
_executor = ThreadPoolExecutor(max_workers=2)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SilentTalk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Serve generated audio
config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(config.AUDIO_DIR)), name="audio")

# ── Lazy singletons ───────────────────────────────────────────────────────────
_predictor: GesturePredictor | None = None
_translator: TuluTranslator | None = None
_tts: TextToSpeech | None = None
_stt: SpeechToText | None = None


def predictor() -> GesturePredictor:
    global _predictor
    if _predictor is None:
        _predictor = GesturePredictor()
    return _predictor


def translator() -> TuluTranslator:
    global _translator
    if _translator is None:
        _translator = TuluTranslator()
    return _translator


def tts() -> TextToSpeech:
    global _tts
    if _tts is None:
        _tts = TextToSpeech()
    return _tts


def stt() -> SpeechToText:
    global _stt
    if _stt is None:
        _stt = SpeechToText()
    return _stt


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/status")
async def status():
    p = predictor()
    labels: list[str] = []
    if config.LABELS_PATH.exists():
        labels = json.loads(config.LABELS_PATH.read_text(encoding="utf-8"))
    return {
        "model_loaded": p.model_loaded,
        "labels": labels,
        "label_count": len(labels),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
    }


# ── WebSocket: real-time gesture recognition ──────────────────────────────────
@app.websocket("/ws/gesture")
async def ws_gesture(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connection opened")
    p = predictor()
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                b64 = msg["data"].split(",", 1)[-1]
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame_bgr is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "bad frame"}))
                    continue

                # Run blocking ML in thread pool — keeps event loop free
                result = await loop.run_in_executor(
                    _executor, p.process_frame, frame_bgr
                )

                _, buf = cv2.imencode(".jpg", result["annotated_frame"],
                                      [cv2.IMWRITE_JPEG_QUALITY, 72])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                await ws.send_text(json.dumps({
                    "type": "result",
                    "annotated": annotated_b64,
                    "gesture": result["gesture"],
                    "confidence": result["confidence"],
                }))

            elif msg.get("type") == "reset":
                p.reset_history()
                await ws.send_text(json.dumps({"type": "reset_ok"}))

            elif msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception:
        log.error(traceback.format_exc())


# ── REST: translate + speak ───────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    tulu_text = translator().translate(text)
    audio_path = tts().synthesize(tulu_text)

    audio_url = None
    if audio_path and Path(audio_path).exists():
        audio_url = f"/audio/{Path(audio_path).name}"

    append_conversation({
        "mode": "sign_to_speech",
        "english": text,
        "tulu": tulu_text,
    })

    return {"english": text, "tulu": tulu_text, "audio_url": audio_url}


# ── REST: speech-to-text ──────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)

    try:
        text = stt().transcribe(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    append_conversation({"mode": "speech_to_text", "transcript": text})
    return {"transcript": text}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
