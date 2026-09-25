"""stop_line: a vehicle stops past the stop line on red without entering the intersection.
Start when it stops (past the line, on red); end when the signal turns green (docs/task_spec.md)."""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import VideoContext, along_segment, condition_segments, front_point, signed_line_distance
from src.scene.geometry import point_in_polygon

LABEL = "stop_line"


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    line_id = cfg["stop_line"]
    a, b = ctx.scene.stop_lines[line_id]
    signal_id = ctx.scene.stop_line_meta[line_id]["signal"]
    if signal_id not in ctx.signals:
        return []
    timeline = ctx.signals[signal_id]
    direction = ctx.approach_direction(cfg["approach"])
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])]
    for track_id, g in veh.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        fp = front_point(g["x1"].to_numpy(), g["y1"].to_numpy(), g["x2"].to_numpy(), g["y2"].to_numpy(),
                         direction)
        dist = signed_line_distance(fp, a, b, direction)
        foot = ctx.foot_px(g)
        inside = (point_in_polygon(foot, ctx.scene.intersection) if ctx.scene.intersection is not None
                  else np.zeros(len(g), bool))
        approached = np.minimum.accumulate(dist) < 0  # it was once behind the line: this approach
        # The box corner can sit ahead of the real bumper in this oblique view: demand a margin
        # proportional to the vehicle's own size as well as an absolute jitter guard.
        box_h = (g["y2"] - g["y1"]).to_numpy()
        past = dist >= np.maximum(cfg["min_past_px"], cfg.get("min_past_box_h", 0.0) * box_h)
        cond = ((g["speed_rel"].to_numpy() < cfg["still_rel_speed"]) & past
                & along_segment(fp, a, b, cfg["max_lateral_margin"]) & ~inside
                & (timeline.at(t) == "red") & approached)
        for start, _, i0, i1 in condition_segments(t, cond, cfg["min_stop_s"], cfg.get("max_gap_s", 0.2), ctx.step_s):
            green = timeline.next_change_to("green", start)
            end = min(green if green is not None else ctx.duration, ctx.duration)
            if end <= start:
                continue
            in_xwalk = ctx.scene.in_kind(foot[i0:i1 + 1], "crosswalks")
            events.append(Event(LABEL, start, end, 0.8, (int(track_id),), {
                "track_id": int(track_id), "cls": int(g["cls_major"].iloc[0]), "signal": signal_id,
                "stopped_from": round(start, 3), "green_at": None if green is None else round(green, 3),
                "past_line_px": round(float(np.median(dist[i0:i1 + 1])), 1),
                "on_crosswalk_fraction": round(float(in_xwalk.mean()), 2),
                "track_last_seen": round(float(t[-1]), 2),
            }))
    return events
