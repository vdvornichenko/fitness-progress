"""
Configuration dataclass and YAML loader.

target_shoulder_width = None  →  auto-compute from median shoulder width across
all detected images (recommended default).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclasses.dataclass
class Config:
    # Output canvas
    output_width: int = 1080
    output_height: int = 1920

    # Alignment targets (output canvas pixels)
    target_shoulder_midpoint_x: int = 540
    target_shoulder_midpoint_y: int = 620
    # None = auto-compute from median across detected images
    target_shoulder_width: Optional[float] = None

    # Transform clamping
    max_rotation_degrees: float = 8.0
    min_scale: float = 0.75
    max_scale: float = 1.35

    # Video rendering (Phase 2)
    frame_duration_seconds: float = 0.8
    fps: int = 30

    # Detection quality thresholds
    min_landmark_visibility: float = 0.5
    min_shoulder_width_px: float = 80.0
    max_shoulder_angle_abs_degrees: float = 30.0
    min_margin_from_edge_px: int = 20

    # ── Video import (Phase 4) ─────────────────────────────────────────────
    video_enabled: bool = True
    # Space-separated or comma-separated list is also accepted via YAML
    video_extensions: dataclasses.field(default_factory=list) = dataclasses.field(
        default_factory=lambda: [".mp4", ".mov", ".m4v"]
    )
    # How often to sample a candidate frame (seconds)
    video_sample_interval: float = 0.25
    # Avoid frames this close to video start/end (seconds)
    video_avoid_start_seconds: float = 0.5
    video_avoid_end_seconds: float = 0.5
    # Reject automatically-selected frames below this score (0–1)
    video_min_score: float = 0.5
    # Scoring weights (must sum ≈ 1.0)
    video_weight_pose: float = 0.35
    video_weight_shoulders: float = 0.25
    video_weight_blur: float = 0.15
    video_weight_centering: float = 0.15
    video_weight_brightness: float = 0.10

    @property
    def target_shoulder_midpoint(self) -> tuple[int, int]:
        return (self.target_shoulder_midpoint_x, self.target_shoulder_midpoint_y)


_KNOWN_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(Config))


def load_config(path: Optional[Path] = None) -> Config:
    """Load Config from a YAML file; fall back to defaults if absent or None."""
    if path is None or not path.exists():
        return Config()
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required to load a config file. Run: pip install PyYAML"
        )
    with open(path) as fh:
        raw = _yaml.safe_load(fh) or {}

    # Flatten the optional nested `video:` section into top-level keys
    video_section = raw.pop("video", None)
    if isinstance(video_section, dict):
        for k, v in video_section.items():
            flat_key = f"video_{k}"
            if flat_key in _KNOWN_FIELDS:
                raw[flat_key] = v
        # Handle sub-sections: best_frame / scoring
        if "best_frame" in video_section:
            for k, v in video_section["best_frame"].items():
                flat_key = f"video_{k}"
                if flat_key in _KNOWN_FIELDS:
                    raw[flat_key] = v
        if "scoring" in video_section:
            for k, v in video_section["scoring"].items():
                flat_key = f"video_weight_{k.removesuffix('_weight')}"
                if flat_key in _KNOWN_FIELDS:
                    raw[flat_key] = v

    known = {k: v for k, v in raw.items() if k in _KNOWN_FIELDS}
    return Config(**known)
