"""Hand-built VideoContexts for engine tests: a 1000x1000 scene with a horizontal stop line."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.events.base import SignalTimeline, VideoContext
from src.scene.geometry import Region, Scene

W = H = 1000
FPS = 10.0


def scene() -> Scene:
    s = Scene(W, H)
    s.road = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], float)
    s.stop_lines["stop"] = np.array([[100, 500], [900, 500]], float)
    s.stop_line_meta["stop"] = {"approach": "down", "signal": "sig"}
    s.approaches["down"] = {"direction": [0.0, 1.0]}
    s.regions.append(Region("xing", "crosswalks", np.array([[100, 510], [900, 510], [900, 560], [100, 560]], float)))
    s.regions.append(Region("refuge", "islands", np.array([[450, 200], [550, 200], [550, 260], [450, 260]], float)))
    s.regions.append(Region("walk", "sidewalks", np.array([[0, 0], [60, 0], [60, 1000], [0, 1000]], float)))
    s.intersection = np.array([[100, 570], [900, 570], [900, 895], [100, 895]], float)
    return s


def signal(phases: list[tuple[float, str]], end: float = 60.0) -> SignalTimeline:
    t = np.arange(0, end, 1 / FPS)
    state = np.empty(len(t), dtype=object)
    for start, st in phases:
        state[t >= start - 1e-9] = st
    return SignalTimeline(t, state)


def track(track_id: int, cls: int, times, xs, ys, w=40.0, h=40.0) -> pd.DataFrame:
    """Rows in the VideoContext.samples schema; foot point (x, y) is the bottom centre of the box."""
    times, xs, ys = (np.asarray(v, float) for v in (times, xs, ys))
    vx = np.gradient(xs, times) if len(times) > 1 else np.zeros_like(xs)
    vy = np.gradient(ys, times) if len(times) > 1 else np.zeros_like(ys)
    speed_rel = np.hypot(vx, vy) / h
    still = speed_rel < 0.15
    stationary = np.zeros_like(times)
    for i in range(1, len(times)):
        stationary[i] = stationary[i - 1] + (times[i] - times[i - 1]) if still[i] and still[i - 1] else 0.0
    return pd.DataFrame({
        "frame": np.round(times * FPS).astype(int), "t": times, "track_id": track_id, "cls": cls,
        "cls_major": cls, "gx": xs / W, "gy": ys / H, "vx": vx / W, "vy": vy / H,
        "speed_rel": speed_rel, "stationary_s": stationary, "box_h": h / H,
        "x1": xs - w / 2, "y1": ys - h, "x2": xs + w / 2, "y2": ys,
    })


def context(tracks: list[pd.DataFrame], phases, duration: float = 60.0) -> VideoContext:
    return VideoContext("test.mp4", FPS, W, H, duration, 1, pd.concat(tracks, ignore_index=True),
                        scene(), {"sig": signal(phases, duration)})
