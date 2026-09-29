"""Tracks that drive straight, turn along a circular arc, and drive on, for the turn engines."""

from __future__ import annotations

import numpy as np

from event_fixtures import track


def arc_path(start, heading_deg, turn_deg, radius, before=300.0, after=300.0, speed=100.0, t0=0.0,
             track_id=1, cls=3):
    """Straight for `before` px, a circular arc turning `turn_deg` (positive = clockwise on screen,
    a right turn), straight for `after` px. Returns (track rows, arc start time, arc end time)."""
    h0 = np.radians(heading_deg)
    p = np.asarray(start, float) + before * np.array([np.cos(h0), np.sin(h0)])
    pts = [np.asarray(start, float), p]
    n = 60
    sign = np.sign(turn_deg)
    centre = p + radius * np.array([np.cos(h0 + sign * np.pi / 2), np.sin(h0 + sign * np.pi / 2)])
    a0 = np.arctan2(p[1] - centre[1], p[0] - centre[0])
    for k in range(1, n + 1):
        a = a0 + sign * np.radians(abs(turn_deg)) * k / n
        pts.append(centre + radius * np.array([np.cos(a), np.sin(a)]))
    h1 = h0 + np.radians(turn_deg)
    pts.append(pts[-1] + after * np.array([np.cos(h1), np.sin(h1)]))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.round(np.arange(t0, t0 + cum[-1] / speed, 0.1), 3)
    s = (t - t0) * speed
    x, y = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
    arc_len = radius * np.radians(abs(turn_deg))
    return track(track_id, cls, t, x, y), t0 + before / speed, t0 + (before + arc_len) / speed
