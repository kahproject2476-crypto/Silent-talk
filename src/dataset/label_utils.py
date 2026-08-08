"""Label parsing helpers for ISL dataset folders."""

from __future__ import annotations

import re


def parse_gesture_label(folder_name: str) -> str:
    """
    Convert dataset folder names like '48. Hello' to 'Hello'.
    """
    name = folder_name.strip()
    match = re.match(r"^\d+\.\s*(.+)$", name)
    if match:
        return match.group(1).strip()
    return name


def normalize_label(label: str) -> str:
    """Normalize label for model training."""
    return parse_gesture_label(label).upper().replace(" ", "_")
