"""Provisional baseline. Replace signatures only after reading the organizer kit."""

from pathlib import Path


def detect_events(video_path: str | Path) -> list:
    """Return no event candidates until an official interface is available."""
    return []


class RiskEstimator:
    """Zero-risk placeholder; its final input/output contract is not yet known."""

    def step(self, *args: object, **kwargs: object) -> float:
        return 0.0
