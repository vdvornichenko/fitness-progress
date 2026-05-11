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

# Heuristic thresholds for detecting washed-out HDR frames.
#
# IMPORTANT — macOS / VideoToolbox note:
# cv2.VideoCapture uses VideoToolbox on macOS, which ALWAYS converts HDR
# content to SDR (BT.709 / sRGB) before handing bytes to OpenCV.  Applying
# the PQ EOTF to already-SDR data produces near-black output, making the
# frame look grey.  Therefore _HDR_LUMINANCE_THRESHOLD_UNKNOWN is set very
# high (210) so the heuristic never fires when ffprobe is absent.
#
# The lower confirmed threshold (125) is reserved for a future setup where
# ffmpeg (not VideoCapture) is used as decoder and actually delivers raw
# PQ-encoded bytes — in that case the mean of a PQ frame sits above 150.
_HDR_LUMINANCE_THRESHOLD_CONFIRMED = 125.0   # ffprobe said HDR + raw PQ decoder
_HDR_LUMINANCE_THRESHOLD_UNKNOWN   = 210.0   # ffprobe absent — VideoToolbox SDR output


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


def _hdr_heuristic(bgr: np.ndarray, threshold: float = 210.0) -> bool:
    """Return True if the frame looks like washed-out PQ-encoded HDR.

    Only the mean-brightness test is used.  On macOS with VideoToolbox
    (the OpenCV backend) this heuristic is intentionally set to a very high
    threshold (210) so it never fires — VideoToolbox already converts HDR to
    SDR, and applying the PQ pipeline again would produce grey output.
    When a raw-PQ decoder such as ffmpeg is used the mean of an HDR10 frame
    sits above ~150, so a threshold of 125 is appropriate in that context.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(gray.mean()) > threshold


# ── PQ (ST 2084) EOTF constants ───────────────────────────────────────────
_PQ_C1 = 0.8359375        # 3424 / 4096
_PQ_C2 = 18.8515625       # 2413 / 4096 × 32
_PQ_C3 = 18.6875          # 2415 / 4096 × 32
_PQ_M1 = 0.1593017578125  # 2610 / (4096 × 4)
_PQ_M2 = 78.84375         # 2523 / (4096 × 32)


def _pq_eotf(v: np.ndarray) -> np.ndarray:
    """PQ (ST 2084) EOTF: signal [0, 1] → linear scene light [0, 1]."""
    v    = v.clip(0.0, 1.0)
    v_m2 = np.power(v, 1.0 / _PQ_M2)
    num  = np.maximum(v_m2 - _PQ_C1, 0.0)
    den  = np.maximum(_PQ_C2 - _PQ_C3 * v_m2, 1e-10)
    return np.power(num / den, 1.0 / _PQ_M1)


def _hable(v: np.ndarray) -> np.ndarray:
    """Uncharted 2 / Hable tone-map curve.

    Scale factor 150 matches the fast-hdr reference implementation
    (``gen/hdr_sdr_gen.cpp``).  Applied per channel independently.
    """
    v = v * 150.0
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F


def _srgb_oetf(v: np.ndarray) -> np.ndarray:
    """Proper piecewise sRGB OETF (linear → gamma-encoded display signal)."""
    return np.where(
        v <= 0.0031308,
        v * 12.92,
        1.055 * np.power(v.clip(min=0.0031308), 1.0 / 2.4) - 0.055,
    )


# BT.2020 → BT.709 colour-primary matrix (applied in linear light).
# Source: ITU-R BT.2087-0, Table 2.
# Row order: R, G, B.  Input is BGR so we apply to a RGB view and convert back.
_BT2020_TO_BT709 = np.array(
    [
        [ 1.6605, -0.5876, -0.0728],
        [-0.1246,  1.1329, -0.0083],
        [-0.0182, -0.1006,  1.1187],
    ],
    dtype=np.float32,
)


def _bt2020_to_bt709(linear_bgr: np.ndarray) -> np.ndarray:
    """Convert linear BT.2020 RGB to linear BT.709 RGB.

    Input / output are both in BGR channel order (as used by OpenCV).
    Applied before tone-mapping so out-of-gamut primaries are compressed
    into the BT.709 gamut, restoring colour saturation that VideoToolbox
    loses when it decodes an HDR10/BT.2020 stream without gamut mapping.
    """
    # Work in RGB order to match the matrix row convention
    rgb = linear_bgr[:, :, ::-1]                     # BGR → RGB view
    h, w = rgb.shape[:2]
    rgb_flat = rgb.reshape(-1, 3) @ _BT2020_TO_BT709.T
    rgb_out  = rgb_flat.reshape(h, w, 3).clip(0.0, None)
    return rgb_out[:, :, ::-1]                        # RGB → BGR


def _tonemap_sdr(bgr: np.ndarray) -> np.ndarray:
    """Convert a washed-out PQ-encoded 8-bit frame to a natural SDR appearance.

    Pipeline:
      1. Normalise 8-bit → [0, 1] signal space
      2. PQ (ST 2084) EOTF      → linear BT.2020 scene light  (per channel)
      3. BT.2020 → BT.709 gamut matrix (linear light) — restores saturation
      4. Hable "Uncharted 2" tone-map × 150 scale      (per channel)
      5. sRGB OETF               → display-ready sRGB   (per channel)

    Input: 8-bit BGR as delivered by OpenCV / VideoToolbox from an HDR10 video.
    Output: 8-bit BGR ready for normal sRGB display.
    """
    f      = bgr.astype(np.float32) / 255.0
    lin    = _pq_eotf(f)                  # → linear BT.2020
    lin709 = _bt2020_to_bt709(lin)        # → linear BT.709
    tone   = _hable(lin709)               # Hable tone-map
    sdr    = _srgb_oetf(tone)             # → sRGB
    return (sdr.clip(0.0, 1.0) * 255.0).astype(np.uint8)


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

    # Tone-mapping strategy:
    # • ffprobe confirmed HDR (is_hdr=True): ALWAYS apply _tonemap_sdr().
    #   VideoToolbox decodes the PQ/BT.2020 stream but does not fully convert
    #   to sRGB; our pipeline (PQ EOTF → BT.2020→BT.709 → Hable → sRGB)
    #   restores correct colours.
    # • Unknown/SDR (is_hdr=False): use a luminance heuristic as a safety net
    #   for mislabelled content — only fires if mean > 210 (essentially never
    #   for normally-decoded SDR footage).
    if is_hdr:
        apply_tonemap = True
    else:
        apply_tonemap = False   # may be set to True by the heuristic below
    _probe_frames_remaining: int = 2  # used only for the SDR heuristic path

    candidates: list[CandidateFrame] = []
    for idx, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, bgr = cap.read()
        if not ret:
            continue
        bgr = _rotate_frame(bgr, rotation)
        if not is_hdr and _probe_frames_remaining > 0:
            _probe_frames_remaining -= 1
            if _hdr_heuristic(bgr, _HDR_LUMINANCE_THRESHOLD_UNKNOWN):
                apply_tonemap = True
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
