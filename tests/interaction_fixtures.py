"""Road users with scripted speed profiles, for the near_miss and accident engines."""

from __future__ import annotations

import numpy as np

from event_fixtures import track


def moving(track_id, cls, start, direction, speed_of_t, t0=0.0, t1=8.0, w=40.0, h=40.0):
    """Foot point starts at `start` and moves along `direction` with speed_of_t(t) px/s (10 Hz)."""
    t = np.round(np.arange(t0, t1 + 1e-9, 0.1), 3)
    v = np.maximum(np.asarray(speed_of_t(t), float), 0.0)
    s = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    xy = np.asarray(start, float) + s[:, None] * d
    return track(track_id, cls, t, xy[:, 0], xy[:, 1], w=w, h=h)


def cruise(v0):
    return lambda t: np.full_like(t, v0)


def brake(v0, t_brake, decel):
    """Constant speed v0 until t_brake, then constant deceleration (px/s^2) to a stop."""
    return lambda t: np.where(t < t_brake, v0, np.maximum(v0 - decel * (t - t_brake), 0.0))
