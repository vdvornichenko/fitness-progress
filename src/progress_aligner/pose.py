"""
MediaPipe shoulder detection with validation.

Key fix: angle normalisation to [-90, 90]°.
MediaPipe's LEFT_SHOULDER (index 11) is the body-left shoulder, which in a
mirror selfie appears on the RIGHT side of the frame.  The raw atan2 angle of
the left→right vector is therefore near ±180°, not near 0°.  Normalising to
[-90, 90] gives the true geometric tilt regardless of left/right assignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from .config import Config

LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12


@dataclass
class ShoulderDetection:
    detected: bool
    fail_reason: str = ""
    left_px: tuple[int, int] = (0, 0)
    right_px: tuple[int, int] = (0, 0)
    midpoint_px: tuple[int, int] = (0, 0)
    shoulder_width_px: float = 0.0
    angle_degrees: float = 0.0
    left_visibility: float = 0.0
    right_visibility: float = 0.0


def load_rgb(path: Path) -> np.ndarray:
    """Load image with Pillow, apply EXIF orientation, return numpy RGB uint8 array."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img)


def _pixel(lm, w: int, h: int) -> tuple[int, int]:
    return (int(lm.x * w), int(lm.y * h))


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _tilt_angle(left: tuple[int, int], right: tuple[int, int]) -> float:
    """Geometric tilt of the shoulder line normalised to [-90, 90]°."""
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def detect_shoulders(
    rgb: np.ndarray,
    pose_model: mp.solutions.pose.Pose,
    config: "Config",
) -> ShoulderDetection:
    """Run MediaPipe pose detection and return a validated ShoulderDetection."""
    h, w = rgb.shape[:2]
    result = pose_model.process(rgb)

    if result.pose_landmarks is None:
        return ShoulderDetection(detected=False, fail_reason="no pose landmarks detected")

    lm  = result.pose_landmarks.landmark
    ls  = lm[LEFT_SHOULDER_IDX]
    rs  = lm[RIGHT_SHOULDER_IDX]
    lv  = float(ls.visibility)
    rv  = float(rs.visibility)

    if lv < config.min_landmark_visibility:
        return ShoulderDetection(
            detected=False,
            fail_reason=f"left_shoulder visibility {lv:.2f} < {config.min_landmark_visibility}",
            left_visibility=lv, right_visibility=rv,
        )
    if rv < config.min_landmark_visibility:
        return ShoulderDetection(
            detected=False,
            fail_reason=f"right_shoulder visibility {rv:.2f} < {config.min_landmark_visibility}",
            left_visibility=lv, right_visibility=rv,
        )

    lp = _pixel(ls, w, h)
    rp = _pixel(rs, w, h)

    width = _dist(lp, rp)
    if width < config.min_shoulder_width_px:
        return ShoulderDetection(
            detected=False,
            fail_reason=f"shoulder_width {width:.0f}px < {config.min_shoulder_width_px:.0f}px",
            left_visibility=lv, right_visibility=rv,
        )

    angle = _tilt_angle(lp, rp)
    if abs(angle) > config.max_shoulder_angle_abs_degrees:
        return ShoulderDetection(
            detected=False,
            fail_reason=f"shoulder_angle {angle:.1f}° > {config.max_shoulder_angle_abs_degrees}°",
            left_visibility=lv, right_visibility=rv,
        )

    margin = config.min_margin_from_edge_px
    for name, (x, y) in [("left_shoulder", lp), ("right_shoulder", rp)]:
        if x < margin or x > w - margin or y < margin or y > h - margin:
            return ShoulderDetection(
                detected=False,
                fail_reason=f"{name} too close to image edge",
                left_visibility=lv, right_visibility=rv,
            )

    mid = _midpoint(lp, rp)
    return ShoulderDetection(
        detected=True,
        left_px=lp,
        right_px=rp,
        midpoint_px=mid,
        shoulder_width_px=round(width, 1),
        angle_degrees=round(angle, 3),
        left_visibility=round(lv, 4),
        right_visibility=round(rv, 4),
    )
