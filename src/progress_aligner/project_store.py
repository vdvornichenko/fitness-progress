"""
JSON project state — write project.json.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .alignment import Transform
    from .config import Config
    from .pose import ShoulderDetection


def _pose_dict(d: "ShoulderDetection") -> dict:
    return {
        "detected": d.detected,
        "fail_reason": d.fail_reason if not d.detected else None,
        "left_visibility": d.left_visibility,
        "right_visibility": d.right_visibility,
        "left_shoulder": list(d.left_px),
        "right_shoulder": list(d.right_px),
        "shoulder_midpoint": list(d.midpoint_px),
        "shoulder_width_px": d.shoulder_width_px,
        "shoulder_angle_degrees": d.angle_degrees,
    }


def _transform_dict(t: Optional["Transform"]) -> dict:
    if t is None:
        return {"rotation_degrees": 0.0, "scale": 1.0, "translate_x": 0.0, "translate_y": 0.0}
    return {
        "rotation_degrees": t.rotation_degrees,
        "scale": t.scale,
        "translate_x": t.translate_x,
        "translate_y": t.translate_y,
        "clamped_rotation": t.clamped_rotation,
        "clamped_scale": t.clamped_scale,
    }


def build_item(
    index: int,
    source_path: Path,
    output_base: Path,
    capture_date: Optional[datetime],
    detection: "ShoulderDetection",
    transform: Optional["Transform"],
) -> dict:
    idx = f"{index:04d}"
    return {
        "id": idx,
        "source_path": str(source_path),
        "media_type": "photo",
        "capture_date": capture_date.isoformat() if capture_date else None,
        "status": "approved" if detection.detected else "needs_manual_review",
        "pose": _pose_dict(detection),
        "auto_transform": _transform_dict(transform),
        "manual_adjustment": {
            "rotation_degrees": 0.0,
            "scale": 1.0,
            "translate_x": 0.0,
            "translate_y": 0.0,
        },
        "outputs": {
            "aligned_frame": str(output_base / "aligned_frames" / f"{idx}.png"),
            "debug_frame":   str(output_base / "debug"          / f"{idx}_debug.jpg"),
        },
    }


def save_project(
    path: Path,
    config: "Config",
    items: list[dict],
    effective_target_shoulder_width: float,
    created_at: Optional[datetime] = None,
) -> None:
    project = {
        "project_name": path.parent.name,
        "created_at": (created_at or datetime.now()).isoformat(),
        "settings": {
            "output_width": config.output_width,
            "output_height": config.output_height,
            "target_shoulder_midpoint": list(config.target_shoulder_midpoint),
            "target_shoulder_width_configured": config.target_shoulder_width,
            "target_shoulder_width_used": round(effective_target_shoulder_width, 2),
            "max_rotation_degrees": config.max_rotation_degrees,
            "min_scale": config.min_scale,
            "max_scale": config.max_scale,
            "frame_duration_seconds": config.frame_duration_seconds,
            "fps": config.fps,
        },
        "items": items,
    }
    with open(path, "w") as fh:
        json.dump(project, fh, indent=2)
