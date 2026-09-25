"""Internal evidence-rich interfaces, independent of the official output schema."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import math
from numbers import Integral
import numpy as np


OBJECT_CLASSES = ("person", "bicycle", "motorcycle", "car", "bus", "truck")
EVENT_LABELS = frozenset((
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
))


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Values must be finite")


def _probability(value: float) -> None:
    _finite(value)
    if not 0 <= value <= 1:
        raise ValueError("Probability must be in [0, 1]")


def _index(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("IDs and frame indices must be nonnegative integers")


@dataclass
class Detections:
    """One frame, pixel xyxy boxes, explicitly remapped internal class IDs."""

    xyxy: np.ndarray
    cls: np.ndarray
    score: np.ndarray

    def __post_init__(self) -> None:
        boxes, classes, scores = map(np.asarray, (self.xyxy, self.cls, self.score))
        n = len(boxes) if boxes.ndim else 0
        if boxes.shape != (n, 4) or classes.shape != (n,) or scores.shape != (n,):
            raise ValueError("Expected xyxy (N,4), cls (N,), score (N,)")
        if classes.dtype.kind not in "iu":
            raise ValueError("Class IDs must be integers")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("Detections must be finite")
        if np.any(boxes < 0) or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Boxes require 0 <= x1 < x2 and 0 <= y1 < y2")
        if np.any((classes < 0) | (classes >= len(OBJECT_CLASSES))):
            raise ValueError("Remap detector classes to the six internal IDs")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Detection confidence must be in [0, 1]")
        self.xyxy = boxes.astype(np.float32, copy=True)
        self.cls = classes.astype(np.int64, copy=True)
        self.score = scores.astype(np.float32, copy=True)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.empty((0, 4)), np.empty(0, dtype=np.int64), np.empty(0))


@dataclass(frozen=True)
class TrackState:
    """A tracker observation; t is always frame_idx / actual video fps."""

    track_id: int
    frame_idx: int
    t: float
    cls: int
    score: float
    xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _finite(self.t, *self.xyxy)
        _probability(self.score)
        for index in (self.track_id, self.frame_idx, self.cls):
            _index(index)
        if self.track_id < 0 or self.frame_idx < 0 or self.t < 0:
            raise ValueError("Track IDs, frames and times must be nonnegative")
        if not 0 <= self.cls < len(OBJECT_CLASSES):
            raise ValueError("Unknown internal object class")
        x1, y1, x2, y2 = self.xyxy
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Invalid pixel box")

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)


# ARCHITECTURE.md uses the shorter name in its Tracker protocol.
Track = TrackState


@dataclass(frozen=True)
class Interaction:
    """Pair features in one consistent coordinate system, never assumed metres."""

    t: float
    id_a: int
    id_b: int
    dist: float
    closing_speed: float
    tca: float
    dmin: float
    path_conflict: bool | None = None
    lane_conflict: bool | None = None

    def __post_init__(self) -> None:
        _finite(self.t, self.dist, self.closing_speed, self.tca, self.dmin)
        _index(self.id_a)
        _index(self.id_b)
        if min(self.t, self.dist, self.tca, self.dmin, self.id_a, self.id_b) < 0:
            raise ValueError("Times, distances and IDs must be nonnegative")
        if self.id_a == self.id_b:
            raise ValueError("An interaction requires two different tracks")


class Detector(Protocol):
    def predict(self, frames: list[np.ndarray]) -> list[Detections]: ...


class Tracker(Protocol):
    def update(self, dets: Detections, frame_idx: int) -> list[TrackState]: ...
    def reset(self) -> None: ...


class RiskModel(Protocol):
    """Implementations may consume only features available at the current time."""

    def reset(self) -> None: ...
    def step(self, feats_t: dict[str, float]) -> float: ...


@dataclass(frozen=True)
class Event:
    label: str
    start: float
    end: float
    confidence: float
    track_ids: tuple[int, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.start, self.end)
        _probability(self.confidence)
        if self.label not in EVENT_LABELS or not 0 <= self.start < self.end:
            raise ValueError("Event requires an official label and 0 <= start < end")
        for track_id in self.track_ids:
            _index(track_id)


@dataclass(frozen=True)
class RiskSample:
    time: float
    value: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.time)
        _probability(self.value)
        if self.time < 0:
            raise ValueError("Risk timestamp must be nonnegative")
