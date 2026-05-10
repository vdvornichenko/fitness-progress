"""
Extract candidate frames from a source video at a fixed time interval.

Uses OpenCV VideoCapture directly — no extra dependencies beyond what the
project already requires.

Rotation handling
-----------------
Smartphones record portrait video with the sensor in landscape orientation and
embed a rotation tag in the container metadata (typically 90° or 270°).
OpenCV does NOT apply that rotation automatically, so raw frames arrive
sideways.  This module reads ``cv2.CAP_PROP_ORIENTATION_META`` and rotates
every decoded frame accordingly before returning it.  The corrected pixel
dimensions are reflected in ``VideoInfo.width`` / ``VideoInfo.height``.

HDR handling
------------
HDR10+ / PQ-encoded videos (common on iPhone and modern Android) use the
SMPTE ST 2084 or HLG transfer function instead of standard sRGB gamma.
When OpenCV reads these videos the raw pixel values are PQ-encoded, so they
appear washed-out / pale when displayed as sRGB.

This module uses ``ffprobe`` (if available on PATH) to detect HDR colour
metadata, and applies a Reinhard tone map to every extracted frame so that
shoulder detection and preview thumbnails look natural.  If ``ffprobe`` is not
installed the detection falls back to a heuristic based on average luminance.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    duration_seconds: float
    fps: float
    frame_count: int
    width: int           # after rotation correction
    height: int          # after rotation correction
    rotation_degrees: int  # raw metadata value (0 / 90 / 180 / 270)
    is_hdr: bool = False
    hdr_transfer: str = ""  # e.g. "smpte2084", "hlg", "arib-std-b67"


@dataclass
class CandidateFrame:
    timestamp_seconds: float
    frame_index: int       # 0-based index within the *sampled* candidate list
    rgb: np.ndarray        # HxWx3 uint8, already rotation-corrected


# ── Rotation helpers ───────────────────────────────────────────────────────

def _read_rotation(cap: cv2.VideoCapture) -> int:
    """
    Return the container rotation tag in degrees (0, 90, 180, or 270).

    ``CAP_PROP_ORIENTATION_META`` was added in OpenCV 4.5.  Falls back to 0
    if unavailable or if the backend does not expose it.
    """
    try:
        val = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if val is None or val < 0:
            return 0
        return int(val) % 360
    except Exception:
        return 0


def _rotate_frame(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """
    Rotate *frame* so that it displays correctly.

    The metadata value means "rotate this many degrees clockwise to view
    correctly", so we apply the matching OpenCV rotation.
    """
    if rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── HDR detection & tone mapping ───────────────────────────────────────────

# Transfer-function names that indicate HDR content
_HDR_TRANSFERS: frozenset[str] = frozenset({
    "smpte2084",      # HDR10 / HDR10+ (PQ)
    "arib-std-b67",   # HLG (Hybrid Log Gamma)
    "hlg",            # alternate ffprobe label for HLG
})

# Heuristic: if ffprobe is absent, treat the video as HDR when the mean 8-bit
# luminance of any sampled frame exceeds this threshold (PQ frames are pale).
_HDR_LUMINANCE_THRESHOLD = 180.0


def _probe_hdr(path: Path) -> tuple[bool, str]:
    """
    Query ffprobe for the video stream's colour-transfer metadata.

    Returns ``(is_hdr, transfer_name)``.  Falls back to ``(False, '')`` if
    ffprobe is not installed or the query fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=color_transfer,color_primaries",
                "-of", "default=noprint_wrappers=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=8,
        )
        output = result.stdout.lower()
        for tag in _HDR_TRANSFERS:
            if tag in output:
                return True, tag
        return False, ""
    except FileNotFoundError:
        return False, ""   # ffprobe not on PATH
    except Exception:
        return False, ""


def _hdr_heuristic(bgr: np.ndarray) -> bool:
    """Return True if the frame looks washed-out (likely PQ-encoded HDR)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > _HDR_LUMINANCE_THRESHOLD


def _tonemap_sdr(bgr: np.ndarray) -> np.ndarray:
    """
    Tone-map a washed-out HDR-sourced 8-bit frame to a natural SDR appearance.

    HDR10/PQ frames read by OpenCV appear pale because the PQ transfer curve
    maps content to the upper half of the 8-bit range.  We apply a
    luminance-preserving Reinhard operator: the compression scale is computed
    from the per-pixel maximum channel (a hue-safe luminance proxy) and then
    applied uniformly to all three channels, which keeps hue and saturation
    intact.  Finally we stretch to the full [0, 255] range and apply sRGB
    gamma encoding.
    """
    f = bgr.astype(np.float32) / 255.0
    # Per-pixel luminance proxy: maximum of the three channels
    lum = f.max(axis=2, keepdims=True).clip(min=1e-6)
    # Reinhard on luminance only: lum_out = lum / (1 + lum)
    lum_out = lum / (1.0 + lum)
    # Apply the same scale to all channels to preserve hue/saturation
    f = f * (lum_out / lum)
    # Stretch so the brightest pixel reaches 1.0
    peak = f.max()
    if peak > 1e-6:
        f /= peak
    # Apply sRGB display gamma (~2.2)
    f = np.power(np.clip(f, 0.0, 1.0), 1.0 / 2.2)
    return (f * 255.0).astype(np.uint8)


def get_video_info(path: Path) -> VideoInfo:
    """Read basic metadata from a video file without decoding frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation    = _read_rotation(cap)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    # After a 90° or 270° rotation the width and height axes swap
    if rotation in (90, 270):
        display_w, display_h = raw_h, raw_w
    else:
        display_w, display_h = raw_w, raw_h

    is_hdr, hdr_transfer = _probe_hdr(path)

    return VideoInfo(
        path=path,
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=display_w,
        height=display_h,
        rotation_degrees=rotation,
        is_hdr=is_hdr,
        hdr_transfer=hdr_transfer,
    )


def sample_frames(
    path: Path,
    interval_seconds: float = 0.25,
    avoid_start: float = 0.5,
    avoid_end: float = 0.5,
) -> list[CandidateFrame]:
    """
    Sample frames from *path* every *interval_seconds*, skipping the
    first *avoid_start* and last *avoid_end* seconds.

    Each returned frame is already rotation-corrected (portrait videos are
    upright, landscape videos are unchanged) and tone-mapped to SDR (HDR10+/
    HLG videos are automatically detected and converted to natural colours).
    """
    info = get_video_info(path)
    duration = info.duration_seconds
    is_hdr   = info.is_hdr

    t_start = max(0.0, avoid_start)
    t_end   = max(t_start, duration - avoid_end)

    if t_end <= t_start:
        # Very short video — fall back to single middle frame
        t_start = 0.0
        t_end   = duration

    # Build list of timestamps to sample
    timestamps: list[float] = []
    t = t_start
    while t <= t_end + 1e-6:
        timestamps.append(min(t, t_end))
        t += interval_seconds

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    rotation = _read_rotation(cap)

    # Determine tone-mapping need from the first decoded frame.  Even when
    # ffprobe flags the video as HDR, platform decoders (e.g. VideoToolbox on
    # macOS) may already apply the HDR→SDR conversion, returning frames that
    # look perfectly normal.  Applying _tonemap_sdr on top of those produces
    # wrong/dark colours.  The luminance heuristic is the reliable ground
    # truth: only map if the frame actually looks washed-out (mean > 180).
    apply_tonemap: bool = False  # set once from the first frame

    candidates: list[CandidateFrame] = []
    for idx, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, bgr = cap.read()
        if not ret:
            continue
        bgr = _rotate_frame(bgr, rotation)
        if idx == 0:
            apply_tonemap = _hdr_heuristic(bgr)
        if apply_tonemap:
            bgr = _tonemap_sdr(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        candidates.append(CandidateFrame(
            timestamp_seconds=ts,
            frame_index=idx,
            rgb=rgb,
        ))

    cap.release()
    return candidates
