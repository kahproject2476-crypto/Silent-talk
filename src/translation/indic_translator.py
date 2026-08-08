"""English to Tulu translation using IndicTrans2."""

from __future__ import annotations

import config


class TuluTranslator:
    """Translate recognized ISL text (English) to Tulu."""

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(
                config.INDICTRANS_MODEL, trust_remote_code=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.INDICTRANS_MODEL, trust_remote_code=True
            )
            self._pipeline = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=config.SOURCE_LANG,
                tgt_lang=config.TARGET_LANG,
                trust_remote_code=True,
            )
            self._loaded = True
            return True
        except Exception:
            self._loaded = False
            return False

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if self.load() and self._pipeline is not None:
            try:
                output = self._pipeline(text)
                if output and isinstance(output, list):
                    return output[0].get("translation_text", text)
            except Exception:
                pass

        # Fallback: transliteration-style placeholder when model unavailable
        return self._fallback_translate(text)

    @staticmethod
    def _fallback_translate(text: str) -> str:
        """Simple fallback when IndicTrans2 is not downloaded."""
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate

            return transliterate(text, sanscript.ITRANS, sanscript.KANNADA)
        except Exception:
            return f"[Tulu translation pending] {text}"
