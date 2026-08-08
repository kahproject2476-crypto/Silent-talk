"""Text-to-Speech using gTTS."""

from __future__ import annotations

import uuid
from pathlib import Path

import config


class TextToSpeech:
    """Generate speech audio from translated text."""

    def __init__(self, lang: str | None = None) -> None:
        self.lang = lang or config.TTS_LANG

    def synthesize(self, text: str) -> Path | None:
        text = text.strip()
        if not text:
            return None

        config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.AUDIO_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3"

        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(str(output_path))
            return output_path
        except Exception:
            return None
