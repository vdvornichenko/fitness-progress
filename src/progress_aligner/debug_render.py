"""
Debug overlay renderer.

Draws on the already-aligned (or fallback) frame so you can verify that
shoulders landed where expected on the output canvas.

For detected images shows:
  - Green circle  = left shoulder (body-left, appears on right of frame)
  - Orange circle = right shoulder (body-right, appears on left of frame)
  - Grey line     = shoulder line
  - Cyan dot      = detected shoulder midpoint (should be near the blue cross)
  - Blue cross    = target shoulder midpoint position
  - Transform params in the banner area

For undetected images shows the fallback frame with a NEEDS REVIEW banner.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .alignment import Transform
    from .config import Config
    from .pose import ShoulderDetection

# BGR colours
_GREEN  = (0, 200, 0)
_ORANGE = (0, 140, 255)
_CYAN   = (220, 220, 0)
_GREY   = (180, 180, 180)
_BLUE   = (255, 100, 0)
_RED    = (0, 60, 220)
_WHITE  = (255, 255, 255)
_DARK   = (30, 30, 30)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_H = 72


def _tx_point(pt: tuple[int, int], M: np.ndarray) -> tuple[int, int]:
    """Transform a point from original image space to output canvas space."""
    x, y = pt
    return (
        int(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
        int(M[1, 0] * x + M[1, 1] * y + M[1, 2]),
    )


def _text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw text with a dark shadow for readability on any background."""
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), _FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def render_debug(
    aligned_rgb: np.ndarray,
    detection: "ShoulderDetection",
    transform: Optional["Transform"],
    affine_matrix: Optional[np.ndarray],
    config: "Config",
    source_filename: str,
    index: int,
) -> np.ndarray:
    """Return a BGR debug image the same size as the aligned frame."""
    canvas = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    if detection.detected and affine_matrix is not None and transform is not None:
        lp  = _tx_point(detection.left_px,    affine_matrix)
        rp  = _tx_point(detection.right_px,   affine_matrix)
        mid = _tx_point(detection.midpoint_px, affine_matrix)
        tgt = config.target_shoulder_midpoint

        # Shoulder line
        cv2.line(canvas, lp, rp, _GREY, 2, cv2.LINE_AA)

        # Shoulder circles
        cv2.circle(canvas, lp,  14, _GREEN,  -1)
        cv2.circle(canvas, lp,  14, _WHITE,   1)
        cv2.circle(canvas, rp,  14, _ORANGE, -1)
        cv2.circle(canvas, rp,  14, _WHITE,   1)

        # Detected midpoint
        cv2.circle(canvas, mid, 8, _CYAN, -1)

        # Target crosshair
        cv2.drawMarker(canvas, tgt, _BLUE, cv2.MARKER_CROSS, 44, 2, cv2.LINE_AA)

        # Landmark visibility labels (only if on-canvas)
        if 0 <= lp[0] < w and 0 <= lp[1] < h:
            _text(canvas, f"L {detection.left_visibility:.2f}",
                  (max(lp[0] + 16, 0), max(lp[1] - 8, 12)), _GREEN)
        if 0 <= rp[0] < w and 0 <= rp[1] < h:
            _text(canvas, f"R {detection.right_visibility:.2f}",
                  (max(rp[0] + 16, 0), max(rp[1] - 8, 12)), _ORANGE)

        # Transform detail lines below the top banner
        clamp = ""
        if transform.clamped_rotation:
            clamp += " [rot clamped]"
        if transform.clamped_scale:
            clamp += " [scale clamped]"

        info_lines = [
            f"rot={transform.rotation_degrees:.2f}°  scale={transform.scale:.3f}"
            f"  tx={transform.translate_x:.0f}  ty={transform.translate_y:.0f}{clamp}",
            f"w={detection.shoulder_width_px:.0f}px  tilt={detection.angle_degrees:.2f}°",
        ]
        for i, line in enumerate(info_lines):
            _text(canvas, line, (10, _BANNER_H + 18 + i * 22), _WHITE, scale=0.50)

        status_text  = "OK"
        status_color = _GREEN
    else:
        status_text  = f"NEEDS REVIEW: {detection.fail_reason}"
        status_color = _RED

    # Top banner
    cv2.rectangle(canvas, (0, 0), (w, _BANNER_H), _DARK, -1)
    banner = f"[{index:04d}] {source_filename}"
    _text(canvas, banner,      (10, 26), _WHITE,       scale=0.62, thickness=1)
    _text(canvas, status_text, (10, 54), status_color, scale=0.58, thickness=1)

    return canvas
