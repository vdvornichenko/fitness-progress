#!/usr/bin/env python3
"""
Diagnostic: extract one frame from a video and save it with five different
HDR→SDR conversion approaches so you can visually compare them.

Usage
-----
    cd fitness-progress-aligner
    .venv/bin/python tools/debug_hdr_frame.py  /path/to/video.mp4  [seconds]

Output
------
A folder called  hdr_debug/  with:
  0_raw.png          – exactly what OpenCV decoded (no processing)
  1_reinhard.png     – luminance-preserving Reinhard (current approach)
  2_fasthdr.png      – fast-hdr: PQ EOTF → Hable × 150 → sRGB OETF
  3_clahe_lab.png    – CLAHE on L channel of LAB (contrast only, no tone-map)
  4_hist_stretch.png – simple p1–p99 histogram stretch per channel

Plus a printout of per-channel statistics so you can see the raw values.
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import cv2
import numpy as np


# ── PQ (ST 2084) constants ─────────────────────────────────────────────────
_PQ_C1 = 0.8359375        # 3424 / 4096
_PQ_C2 = 18.8515625       # 2413 / 4096 × 32
_PQ_C3 = 18.6875          # 2415 / 4096 × 32
_PQ_M1 = 0.1593017578125  # 2610 / (4096 × 4)
_PQ_M2 = 78.84375         # 2523 / (4096 × 32)


def _pq_eotf(v: np.ndarray) -> np.ndarray:
    """PQ (ST 2084) EOTF: signal [0, 1] → linear scene light [0, 1]."""
    v = v.clip(0.0, 1.0)
    v_m2 = np.power(v, 1.0 / _PQ_M2)
    num  = np.maximum(v_m2 - _PQ_C1, 0.0)
    den  = np.maximum(_PQ_C2 - _PQ_C3 * v_m2, 1e-10)
    return np.power(num / den, 1.0 / _PQ_M1)


def _hable(v: np.ndarray) -> np.ndarray:
    """Uncharted 2 / Hable tone-map curve (fast-hdr uses scale × 150)."""
    v = v * 150.0
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F


def _srgb_oetf(v: np.ndarray) -> np.ndarray:
    """Proper piecewise sRGB OETF (linear → gamma-encoded)."""
    return np.where(
        v <= 0.0031308,
        v * 12.92,
        1.055 * np.power(v.clip(min=0.0031308), 1.0 / 2.4) - 0.055,
    )


# ── Approaches ────────────────────────────────────────────────────────────

def approach_reinhard(bgr: np.ndarray) -> np.ndarray:
    """Luminance-preserving Reinhard (current project approach)."""
    f   = bgr.astype(np.float32) / 255.0
    lum = f.max(axis=2, keepdims=True).clip(min=1e-6)
    f   = f * (lum / (1.0 + lum)) / lum
    peak = f.max()
    if peak > 1e-6:
        f /= peak
    f = np.power(f.clip(0.0, 1.0), 1.0 / 2.2)
    return (f * 255.0).astype(np.uint8)


# BT.2020 → BT.709 matrix (linear light, ITU-R BT.2087-0 Table 2)
_BT2020_TO_BT709 = np.array(
    [[ 1.6605, -0.5876, -0.0728],
     [-0.1246,  1.1329, -0.0083],
     [-0.0182, -0.1006,  1.1187]],
    dtype=np.float32,
)


def _bt2020_to_bt709(linear_bgr: np.ndarray) -> np.ndarray:
    rgb      = linear_bgr[:, :, ::-1]
    h, w     = rgb.shape[:2]
    rgb_out  = (rgb.reshape(-1, 3) @ _BT2020_TO_BT709.T).reshape(h, w, 3).clip(0.0, None)
    return rgb_out[:, :, ::-1]


def approach_fasthdr(bgr: np.ndarray) -> np.ndarray:
    """fast-hdr pipeline WITHOUT gamut matrix: PQ EOTF → Hable × 150 → sRGB."""
    f    = bgr.astype(np.float32) / 255.0
    lin  = _pq_eotf(f)
    tone = _hable(lin)
    sdr  = _srgb_oetf(tone)
    return (sdr.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def approach_fasthdr_gamut(bgr: np.ndarray) -> np.ndarray:
    """fast-hdr + BT.2020→BT.709 gamut matrix: PQ EOTF → gamut → Hable → sRGB."""
    f      = bgr.astype(np.float32) / 255.0
    lin    = _pq_eotf(f)
    lin709 = _bt2020_to_bt709(lin)
    tone   = _hable(lin709)
    sdr    = _srgb_oetf(tone)
    return (sdr.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def approach_clahe_lab(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel only — hue-safe contrast normalisation."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def approach_hist_stretch(bgr: np.ndarray) -> np.ndarray:
    """Simple per-channel p1–p99 histogram stretch."""
    out = np.empty_like(bgr)
    for ch in range(3):
        c = bgr[:, :, ch].astype(np.float32)
        lo = np.percentile(c, 1)
        hi = np.percentile(c, 99)
        out[:, :, ch] = np.clip((c - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


# ── Statistics ────────────────────────────────────────────────────────────

def print_stats(bgr: np.ndarray, label: str) -> None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ps   = [1, 5, 25, 50, 75, 95, 99]
    pcts = [np.percentile(gray, p) for p in ps]
    dark = float((gray < 25).mean()) * 100
    bright = float((gray > 230).mean()) * 100
    print(f"\n{label}")
    print(f"  mean={gray.mean():.1f}  "
          f"p1={pcts[0]:.0f}  p5={pcts[1]:.0f}  p25={pcts[2]:.0f}  "
          f"p50={pcts[3]:.0f}  p75={pcts[4]:.0f}  p95={pcts[5]:.0f}  "
          f"p99={pcts[6]:.0f}")
    print(f"  dark(<25)={dark:.1f}%   bright(>230)={bright:.1f}%")
    for ch, name in [(0, "B"), (1, "G"), (2, "R")]:
        c = bgr[:, :, ch].astype(np.float32)
        print(f"  {name}: mean={c.mean():.1f}  "
              f"p5={np.percentile(c,5):.0f}  p50={np.percentile(c,50):.0f}  "
              f"p95={np.percentile(c,95):.0f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python tools/debug_hdr_frame.py VIDEO.mp4 [timestamp_sec]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    timestamp  = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    out_dir = Path("hdr_debug")
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        sys.exit(1)

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    seek_ts  = min(timestamp, max(0.0, duration - 0.5))
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ts * 1000.0)
    ok, bgr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_ts * 1000.0)
        ok, bgr = cap.read()
    cap.release()

    if not ok:
        print(f"Could not read frame at {timestamp}s")
        sys.exit(1)

    print(f"\nVideo : {video_path.name}")
    print(f"Frame : {bgr.shape[1]}×{bgr.shape[0]} at {seek_ts:.2f}s")
    print(f"Output: {out_dir.resolve()}/")

    steps = [
        ("0_raw.png",              bgr,                           "0. RAW  (no processing)"),
        ("1_reinhard.png",         approach_reinhard(bgr),        "1. Reinhard (old)"),
        ("2_fasthdr.png",          approach_fasthdr(bgr),         "2. fast-hdr  PQ EOTF → Hable × 150 → sRGB  (no gamut)"),
        ("2b_fasthdr_gamut.png",   approach_fasthdr_gamut(bgr),   "2b. fast-hdr + BT.2020→BT.709 gamut matrix  ← NEW"),
        ("3_clahe_lab.png",        approach_clahe_lab(bgr),       "3. CLAHE on LAB-L"),
        ("4_hist_stretch.png",     approach_hist_stretch(bgr),    "4. Histogram stretch p1–p99"),
    ]

    for fname, img, label in steps:
        cv2.imwrite(str(out_dir / fname), img)
        print_stats(img, label)

    print(f"\nDone. Open  {out_dir.resolve()}  and compare the PNG files.")
    print("Share the stats above so we can pick the right approach.")


if __name__ == "__main__":
    main()
