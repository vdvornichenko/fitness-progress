"""
OpenCV affine transform application and fallback crop.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_affine_matrix(
    midpoint: tuple[int, int],
    rotation_degrees: float,
    scale: float,
    translate_x: float,
    translate_y: float,
) -> np.ndarray:
    """Build a 2×3 OpenCV affine matrix.

    cv2.getRotationMatrix2D(center, angle, scale) rotates+scales around
    `center` keeping it fixed.  Adding translate_x/y moves the (now-fixed)
    midpoint to the target canvas position.
    """
    M = cv2.getRotationMatrix2D(
        center=(float(midpoint[0]), float(midpoint[1])),
        angle=rotation_degrees,
        scale=scale,
    )
    M[0, 2] += translate_x
    M[1, 2] += translate_y
    return M


def apply_affine(
    rgb: np.ndarray,
    M: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Apply affine transform and crop to output canvas size.

    BORDER_REPLICATE fills any uncovered canvas edges with replicated border
    pixels rather than black bars.
    """
    return cv2.warpAffine(
        rgb,
        M,
        (output_width, output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def center_crop_to_canvas(
    rgb: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Fallback for undetected images: scale to cover the canvas, center-crop."""
    h, w = rgb.shape[:2]
    scale = max(output_width / w, output_height / h)
    new_w = max(int(w * scale), output_width)
    new_h = max(int(h * scale), output_height)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x = (new_w - output_width) // 2
    y = (new_h - output_height) // 2
    return resized[y : y + output_height, x : x + output_width]
