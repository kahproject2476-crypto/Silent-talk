"""Discover and validate ISL dataset zip archives."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}


@dataclass
class ArchiveInfo:
    path: Path
    valid: bool
    entry_count: int = 0
    video_count: int = 0
    categories: tuple[str, ...] = ()
    error: str = ""


def inspect_archive(zip_path: Path) -> ArchiveInfo:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            videos = [n for n in names if Path(n).suffix.lower() in VIDEO_EXTENSIONS]
            categories = sorted(
                {
                    parts[0]
                    for n in names
                    if (parts := n.split("/")) and len(parts) >= 2 and parts[1]
                }
            )
            return ArchiveInfo(
                path=zip_path,
                valid=True,
                entry_count=len(names),
                video_count=len(videos),
                categories=tuple(categories),
            )
    except Exception as exc:
        return ArchiveInfo(path=zip_path, valid=False, error=str(exc))


def list_archives(dataset_dir: Path) -> list[ArchiveInfo]:
    return [inspect_archive(path) for path in sorted(dataset_dir.glob("*.zip"))]


def valid_archives(dataset_dir: Path) -> list[ArchiveInfo]:
    return [info for info in list_archives(dataset_dir) if info.valid]
