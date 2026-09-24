"""Internal evidence-rich interfaces, independent of the official output schema."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    label: str
    start: float
    end: float
    confidence: float
    track_ids: tuple[int, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSample:
    time: float
    value: float
    evidence: dict[str, Any] = field(default_factory=dict)
