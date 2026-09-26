"""Shared machinery for event engines: condition-to-segment state machine, signal timeline,
and the per-video context engines replay from (cache tables + scene), never re-running perception."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.atlas.flow import Atlas
from src.scene.geometry import Scene, side_of_line
from src.scene.signal import smooth_states

VEHICLES = (2, 3, 4, 5)  # motorcycle, car, bus, truck; bicycles are ambiguous as "vehicles"
PERSON = 0


def condition_segments(t: np.ndarray, cond: np.ndarray, min_on_s: float, max_gap_s: float,
                       step_s: float) -> list[tuple[float, float, int, int]]:
    """Idle -> candidate -> active -> ending, over one track's samples.

    A run of True becomes active once it has lasted min_on_s (persistence); False gaps up to
    max_gap_s do not end it (hysteresis). Start is backdated to the first True sample of the
    run; end is the first False sample after it, or last sample + step_s if the track ends.
    Returns (start, end, first_index, last_true_index).
    """
    t = np.asarray(t, float)
    cond = np.asarray(cond, bool)
    out = []
    i, n = 0, len(t)
    while i < n:
        if not cond[i]:
            i += 1
            continue
        start_i, last_true = i, i
        j = i + 1
        while j < n:
            if cond[j]:
                last_true = j
            elif t[j] - t[last_true] > max_gap_s:
                break
            j += 1
        end = t[last_true + 1] if last_true + 1 < n else t[last_true] + step_s
        if t[last_true] - t[start_i] + step_s >= min_on_s:
            out.append((float(t[start_i]), float(end), start_i, last_true))
        i = last_true + 1
    return out


@dataclass
class SignalTimeline:
    t: np.ndarray
    state: np.ndarray

    @classmethod
    def from_series(cls, df: pd.DataFrame, signal_id: str) -> "SignalTimeline":
        g = df[df["signal"] == signal_id].sort_values("t")
        return cls(g["t"].to_numpy(float), np.asarray(smooth_states(g["state"].tolist()), dtype=object))

    def at(self, times) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.t, np.asarray(times, float), side="right") - 1, 0, len(self.t) - 1)
        return self.state[idx]

    def next_change_to(self, target: str, after: float) -> float | None:
        """First sample time after `after` showing `target`, or None if it never does."""
        m = (self.t > after) & (self.state == target)
        return float(self.t[int(np.argmax(m))]) if m.any() else None


@dataclass
class VideoContext:
    video_id: str
    fps: float
    width: int
    height: int
    duration: float
    stride: int
    samples: pd.DataFrame          # world kinematics joined with box geometry, one row per track sample
    scene: Scene
    signals: dict[str, SignalTimeline]
    atlas: Atlas | None = None

    @property
    def step_s(self) -> float:
        return self.stride / self.fps

    def approach_direction(self, approach_id: str) -> np.ndarray:
        """Unit travel direction in pixels (camera.yaml stores it in normalized coordinates)."""
        d = np.asarray(self.scene.approaches[approach_id]["direction"], float) * [self.width, self.height]
        return d / np.linalg.norm(d)

    def pixel_velocity(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[["vx", "vy"]].to_numpy() * [self.width, self.height]

    def foot_px(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[["gx", "gy"]].to_numpy() * [self.width, self.height]

    @classmethod
    def from_cache(cls, cache: Path, camera: Path, atlas: Path | None = None) -> "VideoContext":
        import json

        meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        world = pd.read_parquet(cache / "world.parquet")
        world_meta = cache / "world_meta.json"
        source = (json.loads(world_meta.read_text(encoding="utf-8"))["tracks_source"] if world_meta.exists()
                  else "tracks.parquet")
        boxes = pd.read_parquet(cache / source, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        samples = world.merge(boxes, on=["frame", "track_id"], how="left", validate="one_to_one")
        scene = Scene.load(camera, meta["width"], meta["height"])
        signals = {}
        series = cache / "signal_series.parquet"
        if series.exists():
            df = pd.read_parquet(series)
            signals = {sid: SignalTimeline.from_series(df, sid) for sid in df["signal"].unique()}
        return cls(meta["video_id"], meta["fps"], meta["width"], meta["height"], meta["duration"],
                   meta["request"]["stride"], samples, scene, signals,
                   Atlas.load(atlas) if atlas is not None and Path(atlas).exists() else None)


def front_point(x1, y1, x2, y2, direction) -> np.ndarray:
    """Bottom-edge corner furthest along the travel direction: the front-bumper ground proxy."""
    left = np.stack([x1, y2], axis=-1)
    right = np.stack([x2, y2], axis=-1)
    d = np.asarray(direction, float)
    return np.where(((right - left) @ d)[..., None] >= 0, right, left)


def signed_line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, direction) -> np.ndarray:
    """Distance to the infinite line a-b, positive on the side the travel direction points to."""
    d = b - a
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if normal @ np.asarray(direction, float) < 0:
        normal = -normal
    return (np.atleast_2d(points) - a) @ normal


def along_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """True where the projection onto a-b falls within the segment (with a fractional margin)."""
    d = b - a
    s = ((np.atleast_2d(points) - a) @ d) / (d @ d)
    return (s >= -margin) & (s <= 1 + margin)


def heading_alignment(vx, vy, direction) -> np.ndarray:
    v = np.stack([vx, vy], axis=-1)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    speed = np.linalg.norm(v, axis=-1)
    return np.where(speed > 0, (v @ d) / np.maximum(speed, 1e-12), 0.0)


__all__ = ["VEHICLES", "PERSON", "condition_segments", "SignalTimeline", "VideoContext",
           "front_point", "signed_line_distance", "along_segment", "heading_alignment", "side_of_line"]
