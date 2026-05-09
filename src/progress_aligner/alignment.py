"""
Compute alignment transform parameters from a shoulder detection.

Transform sequence (executed in transforms.py via OpenCV):
  1. Rotate around shoulder_midpoint by -shoulder_angle  (level the shoulders)
  2. Scale around shoulder_midpoint by target_width / detected_width
  3. Translate shoulder_midpoint → config.target_shoulder_midpoint

All corrections are clamped to the limits in Config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .pose import ShoulderDetection


@dataclass
class Transform:
    rotation_degrees: float
    scale: float
    translate_x: float      # pixels to shift midpoint → target in X
    translate_y: float      # pixels to shift midpoint → target in Y
    clamped_rotation: bool
    clamped_scale: bool


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def compute_transform(
    detection: "ShoulderDetection",
    config: "Config",
    target_shoulder_width: float,
) -> Transform:
    """Return the Transform that aligns this detection to the target canvas.

    translate_x/y store (target_midpoint - detected_midpoint).
    cv2.getRotationMatrix2D(center=midpoint) keeps the midpoint fixed, so
    adding this offset moves it exactly to the target position.
    """
    rotation, clamped_rot = _clamp(
        -detection.angle_degrees,
        -config.max_rotation_degrees,
        config.max_rotation_degrees,
    )

    raw_scale = target_shoulder_width / detection.shoulder_width_px
    scale, clamped_scale = _clamp(raw_scale, config.min_scale, config.max_scale)

    mx, my = detection.midpoint_px
    tx, ty = config.target_shoulder_midpoint

    return Transform(
        rotation_degrees=round(rotation, 4),
        scale=round(scale, 6),
        translate_x=round(float(tx - mx), 2),
        translate_y=round(float(ty - my), 2),
        clamped_rotation=clamped_rot,
        clamped_scale=clamped_scale,
    )
