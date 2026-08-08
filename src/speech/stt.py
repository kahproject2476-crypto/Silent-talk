"""Speech-to-Text using OpenAI Whisper."""

from __future__ import annotations

from pathlib import Path

import config


class SpeechToText:
    """Convert spoken language to text for two-way communication."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.WHISPER_MODEL
        self._model = None

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import whisper

            self._model = whisper.load_model(self.model_name)
            return True
        except Exception:
            return False

    def transcribe(self, audio_path: Path | str, language: str | None = "en") -> str:
        if not self.load():
            return ""

        try:
            result = self._model.transcribe(
                str(audio_path),
                language=language,
                fp16=False,
            )
            return result.get("text", "").strip()
        except Exception:
            return ""
