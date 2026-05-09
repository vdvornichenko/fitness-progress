"""
Phase 2 — MP4 video rendering.

Produces three videos from the pipeline output:

  progress_original.mp4    — original photos, center-cropped to the canvas
  progress_aligned.mp4     — shoulder-aligned frames
  progress_comparison.mp4  — side-by-side (original | aligned), landscape

Each image is held for config.frame_duration_seconds.
Transition modes: "hard_cut" (default) | "crossfade" (~0.3 s blend).
Date labels are drawn from project.json capture_date when show_date_label=True.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pose import load_rgb
from .transforms import center_crop_to_canvas

# ---------------------------------------------------------------------------
# VideoWriter helpers
# ---------------------------------------------------------------------------

_FOURCC_CANDIDATES = ("avc1", "H264", "mp4v")


def _open_writer(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """Try codec candidates in order; return the first that opens successfully."""
    for fourcc_str in _FOURCC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open VideoWriter for {path}. "
        "Ensure OpenCV was built with video-write support."
    )


def _write_hold(writer: cv2.VideoWriter, frame: np.ndarray, n: int) -> None:
    for _ in range(n):
        writer.write(frame)


def _write_crossfade(
    writer: cv2.VideoWriter,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    hold_frames: int,
    fade_frames: int,
) -> None:
    """Hold frame_a, then crossfade into frame_b over fade_frames."""
    hold = max(hold_frames - fade_frames, 1)
    _write_hold(writer, frame_a, hold)
    for i in range(fade_frames):
        alpha = (i + 1) / (fade_frames + 1)
        blended = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0)
        writer.write(blended)


def _write_frames(
    writer: cv2.VideoWriter,
    frames: list[np.ndarray],
    hold_frames: int,
    fade_frames: int,
) -> None:
    """Write all frames with the configured transition strategy."""
    for i, frame in enumerate(frames):
        if fade_frames > 0 and i < len(frames) - 1:
            _write_crossfade(writer, frame, frames[i + 1], hold_frames, fade_frames)
        else:
            _write_hold(writer, frame, hold_frames)


# ---------------------------------------------------------------------------
# Date label overlay
# ---------------------------------------------------------------------------

def _parse_date(date_str: Optional[str]) -> str:
    if not date_str:
        return ""
    try:
        return datetime.fromisoformat(date_str).strftime("%Y-%m-%d")
    except ValueError:
        return date_str[:10]


def _draw_date_label(frame: np.ndarray, label: str) -> np.ndarray:
    if not label:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = max(w / 1080, 0.5) * 1.1
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 10
    x = (w - tw) // 2
    y = h - pad - baseline
    # Semi-transparent backing rectangle
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _panel_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Small top-left corner label (ORIGINAL / ALIGNED)."""
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (160, 32), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Frame loaders
# ---------------------------------------------------------------------------

def _load_aligned(path: Path, out_w: int, out_h: int) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    if bgr.shape[:2] != (out_h, out_w):
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return bgr


def _load_original_cropped(source_path: Path, out_w: int, out_h: int) -> Optional[np.ndarray]:
    if not source_path.exists():
        return None
    try:
        rgb = load_rgb(source_path)
    except Exception:
        return None
    cropped = center_crop_to_canvas(rgb, out_w, out_h)
    return cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)


def _blank(out_w: int, out_h: int) -> np.ndarray:
    return np.zeros((out_h, out_w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_aligned(
    project_json: Path,
    output_path: Path,
    transition: str = "crossfade",
    show_date_label: bool = True,
    frame_duration_override: Optional[float] = None,
    skip_unreviewed: bool = True,
) -> int:
    """Render aligned frames → output_path.  Returns number of frames written."""
    proj     = json.loads(project_json.read_text())
    settings = proj["settings"]
    fps      = settings.get("fps", 30)
    dur      = frame_duration_override if frame_duration_override is not None else settings.get("frame_duration_seconds", 0.8)
    out_w    = settings["output_width"]
    out_h    = settings["output_height"]
    items    = [
        it for it in proj["items"]
        if not skip_unreviewed or it.get("status") == "approved"
    ]

    hold  = max(int(fps * dur), 1)
    fade  = (min(fps // 3, hold // 2)) if transition == "crossfade" else 0

    frames: list[np.ndarray] = []
    for item in items:
        aln = _load_aligned(Path(item["outputs"]["aligned_frame"]), out_w, out_h)
        if aln is None:
            aln = _blank(out_w, out_h)
        if show_date_label:
            aln = _draw_date_label(aln, _parse_date(item.get("capture_date")))
        frames.append(aln)

    if not frames:
        return 0

    writer = _open_writer(output_path, out_w, out_h, fps)
    _write_frames(writer, frames, hold, fade)
    writer.release()
    return len(frames)


def render_original(
    project_json: Path,
    output_path: Path,
    transition: str = "crossfade",
    show_date_label: bool = True,
    frame_duration_override: Optional[float] = None,
    skip_unreviewed: bool = True,
) -> int:
    """Render original source photos (center-cropped) → output_path."""
    proj     = json.loads(project_json.read_text())
    settings = proj["settings"]
    fps      = settings.get("fps", 30)
    dur      = frame_duration_override if frame_duration_override is not None else settings.get("frame_duration_seconds", 0.8)
    out_w    = settings["output_width"]
    out_h    = settings["output_height"]
    items    = [
        it for it in proj["items"]
        if not skip_unreviewed or it.get("status") == "approved"
    ]

    hold = max(int(fps * dur), 1)
    fade = (min(fps // 3, hold // 2)) if transition == "crossfade" else 0

    frames: list[np.ndarray] = []
    for item in items:
        bgr = _load_original_cropped(Path(item["source_path"]), out_w, out_h)
        if bgr is None:
            bgr = _blank(out_w, out_h)
        if show_date_label:
            bgr = _draw_date_label(bgr, _parse_date(item.get("capture_date")))
        frames.append(bgr)

    if not frames:
        return 0

    writer = _open_writer(output_path, out_w, out_h, fps)
    _write_frames(writer, frames, hold, fade)
    writer.release()
    return len(frames)


def render_comparison(
    project_json: Path,
    output_path: Path,
    transition: str = "crossfade",
    show_date_label: bool = True,
    frame_duration_override: Optional[float] = None,
    skip_unreviewed: bool = True,
) -> int:
    """Render side-by-side comparison (original | aligned).

    Output canvas: 1080 × 960 landscape
      left  panel  = 540 × 960  original center-cropped
      right panel  = 540 × 960  shoulder-aligned
    """
    proj     = json.loads(project_json.read_text())
    settings = proj["settings"]
    fps      = settings.get("fps", 30)
    dur      = frame_duration_override if frame_duration_override is not None else settings.get("frame_duration_seconds", 0.8)
    out_w    = settings["output_width"]
    out_h    = settings["output_height"]
    items    = [
        it for it in proj["items"]
        if not skip_unreviewed or it.get("status") == "approved"
    ]

    panel_w  = out_w // 2        # 540
    panel_h  = out_h // 2        # 960
    comp_w   = out_w             # 1080
    comp_h   = panel_h           # 960

    hold = max(int(fps * dur), 1)
    fade = (min(fps // 3, hold // 2)) if transition == "crossfade" else 0

    frames: list[np.ndarray] = []
    for item in items:
        _orig = _load_original_cropped(Path(item["source_path"]), out_w, out_h)
        orig_bgr = _orig if _orig is not None else _blank(out_w, out_h)
        _aln = _load_aligned(Path(item["outputs"]["aligned_frame"]), out_w, out_h)
        aln_bgr  = _aln if _aln is not None else _blank(out_w, out_h)

        orig_panel = cv2.resize(orig_bgr, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
        aln_panel  = cv2.resize(aln_bgr,  (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)

        orig_panel = _panel_label(orig_panel, "ORIGINAL")
        aln_panel  = _panel_label(aln_panel,  "ALIGNED")

        comp = np.hstack([orig_panel, aln_panel])

        # Divider line
        mid = comp_w // 2
        cv2.line(comp, (mid, 0), (mid, comp_h), (80, 80, 80), 1)

        if show_date_label:
            comp = _draw_date_label(comp, _parse_date(item.get("capture_date")))

        frames.append(comp)

    if not frames:
        return 0

    writer = _open_writer(output_path, comp_w, comp_h, fps)
    _write_frames(writer, frames, hold, fade)
    writer.release()
    return len(frames)
