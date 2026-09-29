"""solid_line_crossing: a lane change or manoeuvre across a solid marking. Start when a wheel crosses
the line; end when the vehicle is fully in the new lane (docs/task_spec.md).

Only lines in camera.yaml `solid_lines` count; they come from the markings, never from traffic. The
bottom corners of the box are the wheel proxy (ARCHITECTURE.md): the event starts at the first sample
of the manoeuvre where a corner is on the new side and ends at the first where both are. The decision
itself uses the foot point, which must settle on one side for min_side_s before and after, beyond a
dead band around the line, within the line's extent. A foot point that jumps (an ID switch between
cars in neighbouring lanes) gives no verdict.
"""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import VideoContext

LABEL = "solid_line_crossing"


def polyline_distance(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of each point to its nearest segment of the polyline (left of travel along the
    polyline is positive), and whether that nearest projection falls within the segment."""
    points = np.atleast_2d(np.asarray(points, float))
    best = np.full(len(points), np.inf)
    signed = np.zeros(len(points))
    within = np.zeros(len(points), bool)
    for a, b in zip(polyline[:-1], polyline[1:]):
        d = b - a
        s = ((points - a) @ d) / (d @ d)
        foot = a + np.clip(s, 0.0, 1.0)[:, None] * d
        dist = np.linalg.norm(points - foot, axis=1)
        cross = d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0])
        closer = dist < best
        best = np.where(closer, dist, best)
        signed = np.where(closer, cross / np.linalg.norm(d), signed)
        within = np.where(closer, (s >= 0.0) & (s <= 1.0), within)
    return signed, within


def _runs(side: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal nonzero side, zeros skipped: (side, first index, last index)."""
    out = []
    for k in np.flatnonzero(side):
        if out and out[-1][0] == side[k]:
            out[-1][2] = k
        else:
            out.append([int(side[k]), int(k), int(k)])
    return [tuple(r) for r in out]


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    if not ctx.scene.solid_lines:
        return []
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])].sort_values(["track_id", "t"])
    edge = cfg["frame_edge_px"]
    for tid, g in veh.groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        if len(t) < 3:
            continue
        x1, y1, x2, y2 = (g[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
        h = np.maximum(y2 - y1, 1.0)
        foot = ctx.foot_px(g)
        visible = (x1 > edge) & (x2 < ctx.width - edge) & (y2 < ctx.height - edge)
        step = np.r_[0.0, np.linalg.norm(np.diff(foot, axis=0), axis=1)] / h
        speed = g["speed_rel"].to_numpy()
        for line_id, line in ctx.scene.solid_lines.items():
            d, within = polyline_distance(foot, line)
            dl, _ = polyline_distance(np.stack([x1, y2], axis=1), line)
            dr, _ = polyline_distance(np.stack([x2, y2], axis=1), line)
            dead = cfg["deadband_box_h"] * h
            side = np.where(d > dead, 1, np.where(d < -dead, -1, 0)) * (within & visible)
            runs = _runs(side)
            for (s0, a0, a1), (s1, b0, b1) in zip(runs[:-1], runs[1:]):
                if s1 != -s0 or t[b0] - t[a1] > cfg["max_transit_s"]:
                    continue
                if t[a1] - t[a0] < cfg["min_side_s"] or t[b1] - t[b0] < cfg["min_side_s"]:
                    continue
                if np.median(speed[a1:b0 + 1]) < cfg["min_speed_rel"]:
                    continue
                if step[a1:b0 + 1].max() > cfg["max_jump_box_h"]:
                    continue
                new = np.stack([dl, dr], axis=1) * s1 > 0                   # corners on the new side
                lo = max(a0, int(np.searchsorted(t, t[b0] - cfg["max_backdate_s"])))
                k0 = b0
                while k0 - 1 >= lo and new[k0 - 1].any():
                    k0 -= 1
                hi = int(np.searchsorted(t, t[b0] + cfg["max_settle_s"], side="right")) - 1
                full = np.flatnonzero(new[b0:hi + 1].all(axis=1))
                k1 = b0 + int(full[0]) if len(full) else min(hi, b1)
                start = float(t[k0])
                end = max(float(t[k1]), start + ctx.step_s)
                events.append(Event(LABEL, start, min(end, ctx.duration), 0.6, (int(tid),), {
                    "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]), "line": line_id,
                    "from_side": int(s0), "foot_cross_t": round(float(t[b0]), 2),
                    "offset_before_box_h": round(float(d[a1] / h[a1]), 2),
                    "offset_after_box_h": round(float(d[b0] / h[b0]), 2),
                    "fully_across": bool(len(full)),
                }))
    return events
