"""red_light: start when the front of the vehicle crosses the stop line while its signal is red;
end when the vehicle leaves the intersection or the frame (docs/task_spec.md)."""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import (VideoContext, along_segment, front_point, heading_alignment,
                             signed_line_distance)
from src.scene.geometry import point_in_polygon

LABEL = "red_light"


def red_onset(ctx: VideoContext, signal_id: str, t: float) -> float | None:
    """Time the current red phase began (None if the signal is not red at t)."""
    tl = ctx.signals[signal_id]
    k = int(np.searchsorted(tl.t, t, side="right") - 1)
    if k < 0 or tl.state[k] != "red":
        return None
    while k > 0 and tl.state[k - 1] == "red":
        k -= 1
    return float(tl.t[k])


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    line_id = cfg["stop_line"]
    if line_id not in ctx.scene.stop_lines:
        return []
    a, b = ctx.scene.stop_lines[line_id]
    signal_id = ctx.scene.stop_line_meta[line_id]["signal"]
    if signal_id not in ctx.signals:
        return []
    direction = ctx.approach_direction(cfg["approach"])
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])]
    for track_id, g in veh.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        fp = front_point(g["x1"].to_numpy(), g["y1"].to_numpy(), g["x2"].to_numpy(), g["y2"].to_numpy(),
                         direction)
        dist = signed_line_distance(fp, a, b, direction)
        lateral = along_segment(fp, a, b, cfg["max_lateral_margin"])
        v = ctx.pixel_velocity(g)
        align = heading_alignment(v[:, 0], v[:, 1], direction)
        crossings = np.where((dist[:-1] < 0) & (dist[1:] >= 0))[0] + 1
        for k in crossings:
            if not (lateral[k] or lateral[k - 1]):
                continue
            if max(align[k - 1], align[k]) < cfg["min_alignment"]:
                continue
            if dist[k:].max() < cfg["min_past_px"]:
                continue
            frac = -dist[k - 1] / (dist[k] - dist[k - 1])
            t_cross = float(t[k - 1] + frac * (t[k] - t[k - 1]))
            state = ctx.signals[signal_id].at([t_cross])[0]
            if state != "red":
                continue
            foot = ctx.foot_px(g)
            inside = (point_in_polygon(foot, ctx.scene.intersection) if ctx.scene.intersection is not None
                      else np.zeros(len(g), bool))
            entered = np.where(inside[k:])[0]
            end = None
            if len(entered):
                first_in = k + entered[0]
                left = np.where(~inside[first_in:])[0]
                if len(left):
                    end = float(t[first_in + left[0]])
            if end is None:
                end = float(min(t[-1] + ctx.step_s, ctx.duration))
            onset = red_onset(ctx, signal_id, t_cross)
            events.append(Event(LABEL, t_cross, end, 0.9 if onset is not None and t_cross - onset >= 1.0 else 0.6,
                                (int(track_id),), {
                "track_id": int(track_id), "cls": int(g["cls_major"].iloc[0]), "signal": signal_id,
                "signal_state": str(state), "red_for_s": None if onset is None else round(t_cross - onset, 2),
                "crossing_t": round(t_cross, 3), "front_px": [round(float(x), 1) for x in fp[k]],
                "max_past_px": round(float(dist[k:].max()), 1), "entered_intersection": bool(len(entered)),
            }))
            break
    return events
