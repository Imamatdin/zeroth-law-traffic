"""Turn manoeuvres from one track's heading, shared by illegal_turn and illegal_u_turn.

A turn is a maximal run where the smoothed heading keeps changing at min_rate_deg_s or faster, the
rate taken across rate_window_s so that sample-to-sample noise does not flip its sign
(short slower gaps bridged) with a total change of at least min_turn_deg, lasting min_turn_s and
covering min_travel_box_h of path. Heading changes faster than max_rate_deg_s are not steering but
a velocity estimate flipping on a slow, jittering box: such samples neither extend nor break a run.
It starts at the run's
first sample ("vehicle starts turning") and ends one step after its last ("completes the turn").
Under the camera's perspective straight roads stay straight in the image, so a straight drive does
not accumulate heading change; the sign survives, the angle is approximate. Positive change is
clockwise on screen (y down), which is a right turn seen from above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.events.base import VideoContext


@dataclass(frozen=True)
class Turn:
    i0: int
    i1: int
    start: float
    end: float
    change_deg: float

    @property
    def direction(self) -> str:
        return "right" if self.change_deg > 0 else "left"


def _moving_median(t: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    lo = np.searchsorted(t, t - window_s / 2, side="left")
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    return np.array([np.median(y[a:b]) for a, b in zip(lo, hi)])


def find_turns(ctx: VideoContext, g: pd.DataFrame, cfg: dict) -> list[Turn]:
    """Turns of one track (rows sorted by t); only samples moving fast enough carry a heading."""
    edge = cfg["frame_edge_px"]
    moving = ((g["speed_rel"] >= cfg["min_speed_rel"]) & (g["x1"] > edge) & (g["y1"] > edge)
              & (g["x2"] < ctx.width - edge) & (g["y2"] < ctx.height - edge)).to_numpy()
    idx = np.flatnonzero(moving)
    if len(idx) < 4:
        return []
    t = g["t"].to_numpy(float)[idx]
    v = ctx.pixel_velocity(g.iloc[idx])
    heading = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
    heading = _moving_median(t, heading, cfg["heading_smooth_s"])
    half = cfg["rate_window_s"] / 2
    lo, hi = np.maximum(t - half, t[0]), np.minimum(t + half, t[-1])
    rate = (np.interp(hi, t, heading) - np.interp(lo, t, heading)) / np.maximum(hi - lo, 1e-6)
    turning = np.abs(rate) >= cfg["min_rate_deg_s"]
    valid = np.abs(rate) <= cfg["max_rate_deg_s"]
    foot = ctx.foot_px(g.iloc[idx])
    box_h = (g["y2"] - g["y1"]).to_numpy(float)[idx]
    out = []
    k = 0
    while k < len(t):
        if not (turning[k] and valid[k]):
            k += 1
            continue
        a = b = k
        sign = np.sign(rate[k])
        j = k + 1
        while j < len(t) and t[j] - t[b] <= cfg["max_gap_s"]:
            if turning[j] and valid[j] and np.sign(rate[j]) == sign:
                b = j
            elif turning[j] and valid[j]:
                break
            j += 1
        change = float(heading[b] - heading[a])
        path = float(np.linalg.norm(np.diff(foot[a:b + 1], axis=0), axis=1).sum())
        if (abs(change) >= cfg["min_turn_deg"] and t[b] - t[a] >= cfg["min_turn_s"]
                and path >= cfg["min_travel_box_h"] * float(np.median(box_h[a:b + 1]))):
            end = min(float(t[b]) + ctx.step_s, ctx.duration)
            out.append(Turn(int(idx[a]), int(idx[b]), float(t[a]), end, change))
        k = b + 1
    return out


__all__ = ["Turn", "find_turns"]
