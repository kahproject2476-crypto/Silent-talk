"""Shared utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config


def append_conversation(entry: dict, path: Path | None = None) -> None:
    """Append a conversation turn to a JSON log."""
    path = path or config.OUTPUT_DIR / "conversation_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
    else:
        history = []

    entry["timestamp"] = datetime.now().isoformat()
    history.append(entry)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
