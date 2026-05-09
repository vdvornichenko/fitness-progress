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
    known = {k: v for k, v in raw.items() if k in _KNOWN_FIELDS}
    return Config(**known)
