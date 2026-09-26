"""Synthetic road users for risk-model tests (a plain 4K scene, 10 Hz updates)."""

from __future__ import annotations

import numpy as np

from src.anticipation.risk import RiskConfig, RiskModel
from src.contracts import TrackState
from src.scene.geometry import Region, Scene

W, H = 3840, 2160
DT = 0.1


def scene(with_signal: bool = False) -> Scene:
    s = Scene(W, H)
    s.road = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    s.regions.append(Region("walk", "sidewalks", np.array([[0, 0], [150, 0], [150, H], [0, H]], float)))
    if with_signal:
        s.stop_lines["stop"] = np.array([[1000, 1000], [2400, 1000]], float)
        s.stop_line_meta["stop"] = {"approach": "down", "signal": "sig"}
        s.approaches["down"] = {"direction": [0.0, 1.0]}
    return s


def mover(track_id: int, cls: int, p0, v, h: float, w: float, t0: float = 0.0, t1: float = 1e9, stop_at=None):
    """Callable t -> TrackState or None; foot point p0 + v*(t - t0); optional stop point (x, y)."""
    p0, v = np.asarray(p0, float), np.asarray(v, float)

    def at(t: float):
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            return None
        p = p0 + v * (t - t0)
        if stop_at is not None:
            travel = np.linalg.norm(np.asarray(stop_at, float) - p0)
            if np.linalg.norm(p - p0) >= travel:
                p = np.asarray(stop_at, float)
        return _state(track_id, cls, t, p, h, w)
    return at


def _state(track_id, cls, t, p, h, w):
    """None when the box is not fully inside the frame (a detector would not report it whole)."""
    x, y = p
    box = (x - w / 2, y - h, x + w / 2, y)
    if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
        return None
    return TrackState(track_id, int(round(t * 30)), t, cls, 0.9, box)


def braking_mover(track_id: int, cls: int, p0, direction, speed: float, decel: float, t_brake: float,
                  h: float, w: float):
    """Constant speed until t_brake, then constant deceleration to a stop."""
    p0, d = np.asarray(p0, float), np.asarray(direction, float) / np.linalg.norm(direction)

    def at(t: float):
        if t <= t_brake:
            s = speed * t
        else:
            tb = min(t - t_brake, speed / decel)
            s = speed * t_brake + speed * tb - 0.5 * decel * tb * tb
        return _state(track_id, cls, t, p0 + d * s, h, w)
    return at


def run(movers, t_end: float, signal=None, cfg: RiskConfig | None = None, with_signal=False, model=None):
    """Returns (times, risks, evidence). `signal` maps t -> state for signal id 'sig'."""
    model = model or RiskModel(scene(with_signal), None, cfg)
    model.reset({"fps": 29.97, "width": W, "height": H})
    times = np.round(np.arange(0.0, t_end + 1e-9, DT), 3)
    risks, evidence = [], []
    for t in times:
        states = [s for s in (m(t) for m in movers) if s is not None]
        sig = {"sig": signal(t)} if signal else None
        risks.append(model.update(states, float(t), sig))
        evidence.append(model.last_evidence)
    return times, np.array(risks), evidence
