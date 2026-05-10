"""
Select the best still frame (or best clip anchor) from scored video candidates.

Returns a PickedMoment describing what was chosen, or None if no candidate
reached the minimum quality threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .video_scoring import ScoredFrame

if TYPE_CHECKING:
    from .config import Config


@dataclass
class PickedMoment:
    """The result of automatic best-frame selection for one source video."""
    source_path: Path
    timestamp_seconds: float
    score: float
    reason: str
    scored_frame: ScoredFrame       # full scored frame for downstream use
    # All candidates, sorted best-first (useful for manual override UI)
    all_scored: list[ScoredFrame]


def pick_best_frame(
    scored_frames: list[ScoredFrame],
    source_path: Path,
    config: "Config",
) -> Optional[PickedMoment]:
    """
    Return the highest-scoring frame that meets the minimum quality bar.

    If *no* candidate has a detected pose we still return the highest
    general-quality frame (blur / brightness) so the item appears in the
    review UI for manual intervention, rather than silently disappearing.
    """
    if not scored_frames:
        return None

    # Sort best-first
    ranked = sorted(scored_frames, key=lambda sf: sf.score, reverse=True)
    best   = ranked[0]

    if best.score < config.video_min_score:
        # Still return it (marked as needing review) rather than dropping it
        reason = (
            f"best score {best.score:.2f} < threshold {config.video_min_score:.2f}; "
            f"needs manual review. {best.reason}"
        )
    else:
        reason = f"score={best.score:.2f}. {best.reason}"

    return PickedMoment(
        source_path       = source_path,
        timestamp_seconds = best.candidate.timestamp_seconds,
        score             = best.score,
        reason            = reason,
        scored_frame      = best,
        all_scored        = ranked,
    )
