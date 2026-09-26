"""wrong_way: a vehicle moves against the traffic direction of its lane, including the oncoming lane.
Start when it enters the opposing flow; end when it returns to a correct lane or leaves the frame
(docs/task_spec.md).

No lane polygons are verified, so the lane direction is the flow atlas's dominant heading, used only
in cells with enough samples and a consistent direction. Mixed-flow cells (the junction, turn areas)
give no verdict, so turns and U-turns do not fire this class; a vehicle is judged only where the
normal direction is unambiguous. The atlas describes normal movement, which is what "against the
traffic direction" needs; it does not rule on legality of turns (D-003).
"""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import VideoContext, condition_segments

LABEL = "wrong_way"


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    if ctx.atlas is None:
        return []
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])]
    for track_id, g in veh.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        if t[-1] - t[0] < cfg["min_persist_s"]:
            continue
        dev = ctx.atlas.heading_deviation(g["gx"].to_numpy(), g["gy"].to_numpy(), g["heading"].to_numpy(),
                                          cfg["min_resultant"])
        # A box cut by the frame border has no true foot point: its motion gives no heading verdict.
        b = cfg.get("frame_edge_px", 8)
        inside = ((g["x1"] > b) & (g["y1"] > b) & (g["x2"] < ctx.width - b) & (g["y2"] < ctx.height - b)).to_numpy()
        moving = (g["speed_rel"].to_numpy() >= cfg["min_speed_rel"]) & inside
        against = moving & (dev >= cfg["against_deg"])            # NaN compares False: no verdict
        judged = moving & np.isfinite(dev)
        # Boundaries: the event spans the run where the vehicle is not back in agreement with the flow
        # (deviation below return_deg) nor stopped; unjudged cells inside the run do not end it.
        still_wrong = ~(judged & (dev < cfg["return_deg"])) & moving
        box_h = (g["y2"] - g["y1"]).to_numpy()
        foot = ctx.foot_px(g)
        for _, _, i0, i1 in condition_segments(t, against, cfg["min_persist_s"], cfg["max_gap_s"], ctx.step_s):
            seg_h = float(np.median(box_h[i0:i1 + 1]))
            travel = float(np.linalg.norm(foot[i1] - foot[i0]))
            if travel < cfg["min_travel_box_h"] * seg_h:
                continue                                          # jitter or a short reversing manoeuvre
            s0, s1 = i0, i1
            # Backdate to the entry into the opposing flow: over samples already clearly against it
            # (entry_deg), and at most max_unjudged_backdate_s of samples with no verdict.
            while s0 > 0 and moving[s0 - 1]:
                prev = dev[s0 - 1]
                if np.isfinite(prev):
                    if prev < cfg["entry_deg"]:
                        break
                elif t[i0] - t[s0 - 1] > cfg["max_unjudged_backdate_s"]:
                    break
                s0 -= 1
            while s1 + 1 < len(t) and still_wrong[s1 + 1]:
                s1 += 1
            end = float(t[s1 + 1]) if s1 + 1 < len(t) else min(float(t[s1]) + ctx.step_s, ctx.duration)
            events.append(Event(LABEL, float(t[s0]), end, 0.7, (int(track_id),), {
                "track_id": int(track_id), "cls": int(g["cls_major"].iloc[0]),
                "median_deviation_deg": round(float(np.nanmedian(dev[i0:i1 + 1])), 1),
                "judged_fraction": round(float(judged[s0:s1 + 1].mean()), 2),
                "travel_box_h": round(travel / max(seg_h, 1.0), 2),
                "foot_start": [round(float(v), 1) for v in foot[s0]],
                "foot_end": [round(float(v), 1) for v in foot[s1]],
                "left_frame": s1 + 1 >= len(t),
            }))
    return events
