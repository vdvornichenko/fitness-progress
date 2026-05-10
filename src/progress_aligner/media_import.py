"""
Image and video discovery with chronological sorting.

Sort priority:
  1. EXIF DateTimeOriginal / DateTimeDigitized / DateTime
  2. Date pattern in filename  (e.g. 20211015_091727)
  3. File modification time
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v"})

# EXIF tag IDs
_TAG_DATETIME_ORIGINAL  = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME           = 306


def get_exif_date(path: Path) -> Optional[datetime]:
    """Return capture datetime from EXIF tags, or None."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            for tag_id in (_TAG_DATETIME_ORIGINAL, _TAG_DATETIME_DIGITIZED, _TAG_DATETIME):
                value = exif.get(tag_id)
                if isinstance(value, str) and value.strip():
                    try:
                        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
    except Exception:
        pass
    return None


def get_filename_date(path: Path) -> Optional[datetime]:
    """Try to parse a date from the filename stem."""
    stem = path.stem
    # Full datetime: 20211015_091727 or 20211015-091727
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})", stem)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        except ValueError:
            pass
    # Date only: 2021-10-15, 2021_10_15, or 20211015
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", stem)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def get_capture_date(path: Path) -> Optional[datetime]:
    """Best-effort capture date: EXIF > filename pattern > mtime."""
    return (
        get_exif_date(path)
        or get_filename_date(path)
        or datetime.fromtimestamp(path.stat().st_mtime)
    )


def _sort_key(path: Path) -> tuple:
    return (get_capture_date(path), path.name)


def collect_images(folder: Path) -> list[Path]:
    """Return all supported images in folder, sorted chronologically."""
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=_sort_key)


def collect_videos(folder: Path, extensions: Optional[frozenset[str]] = None) -> list[Path]:
    """Return all supported video files in folder, sorted chronologically."""
    exts = extensions if extensions is not None else VIDEO_EXTENSIONS
    videos = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(videos, key=_sort_key)


def get_video_capture_date(path: Path) -> Optional[datetime]:
    """Best-effort capture date for a video: filename pattern > mtime."""
    return get_filename_date(path) or datetime.fromtimestamp(path.stat().st_mtime)
