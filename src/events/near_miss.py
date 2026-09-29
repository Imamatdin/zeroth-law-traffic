"""near_miss: sharp braking or swerving to avoid a collision, with no contact. Start at the onset of
the evasive action; end when the road users are clear of each other (docs/task_spec.md).

A pair is in conflict while its constant-velocity course would bring the two within dmin_max_h of
each other in tca_max_s, closing at closing_min_h or faster. Parallel pairs (same or opposite
direction), which in this oblique view project onto each other all the time, count only when closing
as fast as an imminent rear-end. The conflict becomes a near miss when one of the two, heading
towards the other, brakes hard or swerves around it and back onto its course (sustained
min_evasive_s, box size stable so an occlusion cannot fake it), the two come within near_dist_h of
each other, and they never touch (no footprint contact: that is accident's business). Crossing
paths conflict only while both are moving: a car braking behind a turning bus or a standing queue
is waiting, not evading. The atlas interaction percentiles are not used as thresholds: at this camera they are
dominated by queueing pairs (p1 clearance ~0), so they carry no signal about conflicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Event
from src.events.base import PERSON, VideoContext
from src.events.pairs import evasive_runs, kinematics, pair_table

LABEL = "near_miss"


def evasive_action(g: pd.DataFrame, other: pd.DataFrame, lo: float, hi: float, cfg: dict) -> tuple | None:
    """(kind, onset time, peak) of the earliest hard brake or swerve of track g in [lo, hi] while it
    heads towards `other`, or None."""
    w = g[(g["t"] >= lo) & (g["t"] <= hi)]
    if len(w) < 3:
        return None
    t = w["t"].to_numpy(float)
    others = other.set_index("frame")[["px", "py"]]
    rel = others.reindex(w["frame"]).to_numpy() - w[["px", "py"]].to_numpy()
    towards = np.nan_to_num((rel * w[["pvx", "pvy"]].to_numpy()).sum(axis=1)) > 0
    h = w["h"].to_numpy()
    stable = np.r_[0.0, np.abs(np.diff(h)) / h[1:]] <= cfg["max_box_change"]
    fast_before = pd.Series(w["speed_h"].to_numpy()).rolling(4, min_periods=1).max().to_numpy() >= cfg["min_speed_before_h"]
    found = []
    brake = (w["decel_h"].to_numpy() >= cfg["brake_h_s2"]) & towards & stable & fast_before
    for a, b in evasive_runs(t, brake, cfg["min_evasive_s"], cfg["evasive_gap_s"]):
        k = a
        decel = w["decel_h"].to_numpy()
        while k > 0 and decel[k - 1] >= cfg["onset_h_s2"]:
            k -= 1
        found.append(("brake", float(t[k]), round(float(decel[a:b + 1].max()), 2)))
    if w["cls_major"].iloc[0] != PERSON:
        yaw_signed = w["yaw_deg_s"].to_numpy()
        yaw = np.abs(yaw_signed)
        # Faster than a road vehicle can yaw: heading noise on a small or occluded box.
        swerve = (yaw >= cfg["swerve_deg_s"]) & (yaw <= cfg["swerve_max_deg_s"]) & towards & stable
        for a, b in evasive_runs(t, swerve, cfg["min_evasive_s"], cfg["evasive_gap_s"]):
            # A dodge returns to the original course; a sustained heading change is a turn.
            around = (g["t"] >= t[a] - cfg["swerve_return_s"]) & (g["t"] <= t[b] + cfg["swerve_return_s"])
            net = abs(float(np.trapezoid(g.loc[around, "yaw_deg_s"], g.loc[around, "t"])))
            if net <= cfg["swerve_max_net_deg"]:
                found.append(("swerve", float(t[a]), round(float(yaw[a:b + 1].max()), 1)))
    return min(found, key=lambda f: f[1]) if found else None


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    kin = kinematics(ctx, cfg)
    if kin.empty:
        return []
    pairs = pair_table(ctx, kin, cfg)
    if pairs.empty:
        return []
    parallel = np.abs(pairs["cos"].to_numpy()) > cfg["follow_cos"]
    conflict = ((pairs["tca"] <= cfg["tca_max_s"]) & (pairs["dmin_h"] <= cfg["dmin_max_h"])
                & (pairs["closing_h"] >= cfg["closing_min_h"])).to_numpy().copy()
    conflict &= ~parallel | (pairs["closing_h"].to_numpy() >= cfg["rear_closing_min_h"])
    tracks = {int(tid): g.sort_values("t") for tid, g in kin.groupby("track_id")}
    speed = kin.set_index(["frame", "track_id"])["speed_h"]
    sa = speed.reindex(pd.MultiIndex.from_arrays([pairs["frame"], pairs["id_a"]])).to_numpy()
    sb = speed.reindex(pd.MultiIndex.from_arrays([pairs["frame"], pairs["id_b"]])).to_numpy()
    # Crossing paths conflict only while both are moving; standing traffic is waited for, not avoided.
    conflict &= parallel | ((sa >= cfg["moving_h"]) & (sb >= cfg["moving_h"]))
    events = []
    hot = pairs[conflict]
    for (ida, idb), _ in hot.groupby(["id_a", "id_b"], sort=True):
        p = pairs[(pairs["id_a"] == ida) & (pairs["id_b"] == idb)].sort_values("t")
        t = p["t"].to_numpy(float)
        c = conflict[p.index.to_numpy()]
        for a, b in evasive_runs(t, c, 0.0, cfg["conflict_gap_s"]):
            acts = [(tid, evasive_action(tracks[tid], tracks[other], t[a] - cfg["pre_s"], t[b] + cfg["post_s"], cfg))
                    for tid, other in ((int(ida), int(idb)), (int(idb), int(ida)))]
            acts = [(tid, act) for tid, act in acts if act is not None]
            if not acts:
                continue
            actor, (kind, onset, peak) = min(acts, key=lambda x: x[1][1])
            window = p[(p["t"] >= onset) & (p["t"] <= onset + cfg["max_event_s"])]
            if window["contact"].any() or window.empty or window["dist_h"].min() > cfg["near_dist_h"]:
                continue
            after = p[p["t"] > t[a]]
            clear = after[(after["dist_h"] >= cfg["clear_dist_h"]) & (after["closing_h"] <= 0)]
            end = float(clear["t"].iloc[0]) if len(clear) else float(p["t"].iloc[-1]) + ctx.step_s
            end = min(max(end, onset + ctx.step_s), ctx.duration)
            events.append(Event(LABEL, onset, end, 0.6, (int(ida), int(idb)), {
                "pair": [int(ida), int(idb)], "cls": [int(p["cls_a"].iloc[0]), int(p["cls_b"].iloc[0])],
                "actor": int(actor), "action": kind, "peak": peak,
                "min_tca_s": round(float(p["tca"].to_numpy()[a:b + 1].min()), 2),
                "min_dmin_h": round(float(p["dmin_h"].to_numpy()[a:b + 1].min()), 2),
                "max_closing_h": round(float(p["closing_h"].to_numpy()[a:b + 1].max()), 2),
                "min_dist_h": round(float(window["dist_h"].min()), 2) if len(window) else None,
            }))
    return events
