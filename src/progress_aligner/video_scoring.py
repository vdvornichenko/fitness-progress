"""
Score candidate frames extracted from a source video.

Each frame receives a composite score in [0, 1] based on:
  - pose_score          — were shoulders detected with good confidence?
  - shoulders_score     — visibility of both shoulder landmarks
  - blur_score          — sharpness (variance of Laplacian)
  - centering_score     — how close the shoulder midpoint is to the canvas centre
  - brightness_score    — avoids under/over-exposed frames

The final score is a weighted sum of the five components.  Weights come from
the Config object and should sum to ~1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import mediapipe as mp
import numpy as np

from .pose import ShoulderDetection, detect_shoulders
from .video_sampling import CandidateFrame

if TYPE_CHECKING:
    from .config import Config


@dataclass
class ScoredFrame:
    candidate: CandidateFrame
    detection: ShoulderDetection
    score: float
    # Component scores (each in [0, 1])
    pose_score: float
    shoulders_score: float
    blur_score: float
    centering_score: float
    brightness_score: float
    # Human-readable reason string for the best-frame selection log
    reason: str


# ── Component scorers ──────────────────────────────────────────────────────

def _blur_score(rgb: np.ndarray) -> float:
    """Variance of Laplacian on the luma channel, normalised to [0, 1]."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Empirically: blurry ≈ 0–50, sharp ≈ 200+.  Clamp at 500.
    return float(min(var / 500.0, 1.0))


def _brightness_score(rgb: np.ndarray) -> float:
    """
    Penalise very dark (< 30) and very bright (> 225) frames.
    Optimal brightness is around 100–160.
    """
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean  = float(gray.mean())
    # Gaussian centred at 128, half-width ~80
    score = math.exp(-((mean - 128) ** 2) / (2 * 80 ** 2))
    return score


def _centering_score(
    detection: ShoulderDetection,
    canvas_cx: int,
    canvas_cy: int,
    canvas_w: int,
    canvas_h: int,
) -> float:
    """
    Distance of the shoulder midpoint from the target canvas centre,
    normalised to [0, 1].  Computed in normalised [0, 1] image space.
    """
    if not detection.detected:
        return 0.0
    h_img, w_img = 1, 1   # we work in relative coords
    # Convert pixel midpoint to [0, 1] relative to original frame size
    mx, my = detection.midpoint_px
    # We don't have the original frame dims here — use canvas dims as proxy
    rel_x = mx / canvas_w
    rel_y = my / canvas_h
    target_x = canvas_cx / canvas_w
    target_y = canvas_cy / canvas_h
    dist = math.sqrt((rel_x - target_x) ** 2 + (rel_y - target_y) ** 2)
    # Max possible distance in normalised space ≈ sqrt(2) ≈ 1.41
    return float(max(0.0, 1.0 - dist / 0.5))


def _shoulders_score(detection: ShoulderDetection) -> float:
    """Average visibility of both shoulders."""
    if not detection.detected:
        return 0.0
    return float((detection.left_visibility + detection.right_visibility) / 2.0)


# ── Main scorer ────────────────────────────────────────────────────────────

def score_frames(
    candidates: list[CandidateFrame],
    pose_model: mp.solutions.pose.Pose,
    config: "Config",
) -> list[ScoredFrame]:
    """
    Detect shoulders and compute a composite score for every candidate frame.
    Returns ScoredFrame objects in the same order as *candidates*.
    """
    results: list[ScoredFrame] = []

    for cand in candidates:
        rgb = cand.rgb
        detection = detect_shoulders(rgb, pose_model, config)

        pose_score     = 1.0 if detection.detected else 0.0
        sh_score       = _shoulders_score(detection)
        blur           = _blur_score(rgb)
        brightness     = _brightness_score(rgb)
        centering      = _centering_score(
            detection,
            config.target_shoulder_midpoint_x,
            config.target_shoulder_midpoint_y,
            config.output_width,
            config.output_height,
        )

        composite = (
            config.video_weight_pose        * pose_score
            + config.video_weight_shoulders * sh_score
            + config.video_weight_blur      * blur
            + config.video_weight_centering * centering
            + config.video_weight_brightness * brightness
        )

        parts: list[str] = []
        if detection.detected:
            parts.append("shoulders detected")
        else:
            parts.append(f"no shoulders ({detection.fail_reason})")
        parts.append(f"blur={blur:.2f}")
        parts.append(f"brightness={brightness:.2f}")
        parts.append(f"centering={centering:.2f}")

        results.append(ScoredFrame(
            candidate      = cand,
            detection      = detection,
            score          = round(composite, 4),
            pose_score     = pose_score,
            shoulders_score= sh_score,
            blur_score     = blur,
            centering_score= centering,
            brightness_score= brightness,
            reason         = ", ".join(parts),
        ))

    return results
