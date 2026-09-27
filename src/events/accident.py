"""accident: contact between road users, or between a road user and a fixed object. Start at the first
frame where contact is visible; end when all involved objects stop moving or leave the frame
(docs/task_spec.md).

Candidates come from two sources. Part A runs Part B's causal risk model (src/anticipation/risk.py)
over its own tracks and signal readings; the model's attention on a pair shortly before it touches
makes that pair a candidate. Any pair that touches while closing fast is a candidate too. Part A
never reads RiskEstimator output: the harness runs Part B after Part A, and the rule is one-way.

A candidate is confirmed only by all of:
- contact: the ground footprints (lower bands of the boxes) overlap;
- impact: at contact one of the two decelerates or yaws abruptly, box size stable (not occlusion);
- post-impact stop: within post_window_s both stand still for min_post_stop_s next to each other
  (a vehicle alone, when the other road user is a person who may fall or be carried).
A single vehicle that runs into a kerb, refuge or median with the same impact and stops is a
collision with a fixed object. Queues touching at walking pace, passing boxes that overlap in the
oblique view, and near misses do not meet these together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anticipation.risk import RiskModel
from src.contracts import Event, TrackState
from src.events.base import PERSON, VideoContext
from src.events.failure_to_yield import footprint_points
from src.events.pairs import kinematics, pair_table
from src.scene.geometry import point_in_polygon

LABEL = "accident"


def risk_pass(ctx: VideoContext) -> pd.DataFrame:
    """Part B's causal risk model driven by Part A's own tracks: t, risk, and the pair it watches."""
    model = RiskModel(ctx.scene, ctx.atlas.data if ctx.atlas is not None else None)
    model.reset({"video_id": ctx.video_id, "fps": ctx.fps, "width": ctx.width, "height": ctx.height})
    s = ctx.samples.sort_values(["frame", "track_id"])
    boxes = s[["x1", "y1", "x2", "y2"]].to_numpy(float, copy=True)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ctx.width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ctx.height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    rows = []
    frames = s["frame"].to_numpy()
    tids, cls = s["track_id"].to_numpy(), s["cls"].to_numpy()
    groups = pd.Series(np.arange(len(s))).groupby(frames).indices
    for f in sorted(groups):
        idx = groups[f]
        t = f / ctx.fps
        tracks = [TrackState(int(tids[k]), int(f), t, int(cls[k]), 1.0, tuple(float(v) for v in boxes[k]))
                  for k in idx if valid[k]]
        signals = {sid: str(tl.at([t])[0]) for sid, tl in ctx.signals.items()}
        risk = model.update(tracks, t, signals)
        pair = model.last_evidence.get("pair")
        rows.append((t, risk, tuple(sorted(pair)) if pair else None))
    return pd.DataFrame(rows, columns=["t", "risk", "pair"])


def _still_from(g: pd.DataFrame, t0: float, cfg: dict) -> float | None:
    """First time at or after t0 from which the track stands still for min_post_stop_s (or its track ends
    standing), or None."""
    w = g[g["t"] >= t0]
    t, still = w["t"].to_numpy(float), (w["speed_h"] < cfg["still_h"]).to_numpy()
    for k in np.flatnonzero(still):
        run = still[k:]
        stop = int(np.argmin(run)) if not run.all() else len(run)
        if t[k + stop - 1] - t[k] >= cfg["min_post_stop_s"] or (stop == len(run) and t[-1] - t[k] >= cfg["min_end_stop_s"]):
            return float(t[k])
    return None


def _impact(g: pd.DataFrame, t0: float, cfg: dict) -> float:
    """Largest abrupt deceleration (or yaw rate, scaled) of the track around t0 on a stable box."""
    w = g[(g["t"] >= t0 - cfg["impact_before_s"]) & (g["t"] <= t0 + cfg["impact_after_s"])]
    if len(w) < 2:
        return 0.0
    h = w["h"].to_numpy()
    stable = np.r_[True, np.abs(np.diff(h)) / h[1:] <= cfg["max_box_change"]]
    decel = np.where(stable, w["decel_h"].to_numpy(), 0.0)
    yaw = np.where(stable, np.abs(w["yaw_deg_s"].to_numpy()), 0.0)
    yaw = np.where(yaw <= cfg["impact_yaw_max_deg_s"], yaw, 0.0)
    return float(max(decel.max() / cfg["impact_decel_h_s2"], yaw.max() / cfg["impact_yaw_deg_s"]))


def _end(tracks: dict, ids, t0: float, cfg: dict, ctx: VideoContext) -> float:
    ends = []
    for tid in ids:
        g = tracks[tid]
        stop = _still_from(g, t0, cfg)
        ends.append(stop if stop is not None else float(g["t"].iloc[-1]) + ctx.step_s)
    return min(max(ends), t0 + cfg["max_event_s"], ctx.duration)


def _pair_events(ctx, kin, tracks, pairs, risk, cfg) -> list[Event]:
    events = []
    touching = pairs[pairs["contact"]]
    for (ida, idb), rows in touching.groupby(["id_a", "id_b"], sort=True):
        p = pairs[(pairs["id_a"] == ida) & (pairs["id_b"] == idb)].sort_values("t")
        c0 = float(rows["t"].min())
        before = p[(p["t"] >= c0 - cfg["approach_s"]) & (p["t"] <= c0)]
        closing = float(before["closing_h"].max()) if len(before) else 0.0
        watched = risk[(risk["t"] >= c0 - cfg["risk_lookback_s"]) & (risk["t"] <= c0)
                       & (risk["pair"] == (int(ida), int(idb)))]
        risk_peak = float(watched["risk"].max()) if len(watched) else 0.0
        if closing < cfg["impact_closing_h"] and risk_peak < cfg["risk_candidate"]:
            continue
        ga, gb = tracks[int(ida)], tracks[int(idb)]
        impact = max(_impact(ga, c0, cfg), _impact(gb, c0, cfg))
        if impact < 1.0:
            continue
        people = [int(i) for i, c in ((ida, p["cls_a"].iloc[0]), (idb, p["cls_b"].iloc[0])) if c == PERSON]
        vehicles = [int(i) for i in (ida, idb) if int(i) not in people]
        stops = {tid: _still_from(tracks[tid], c0, cfg) for tid in (int(ida), int(idb))}
        must_stop = vehicles if people else [int(ida), int(idb)]
        if any(stops[tid] is None or stops[tid] > c0 + cfg["post_window_s"] for tid in must_stop):
            continue
        settled = max(stops[tid] for tid in must_stop)
        after = p[(p["t"] >= settled) & (p["t"] <= settled + cfg["min_post_stop_s"])]
        if not people and (after.empty or after["dist_h"].median() > cfg["stay_close_h"]):
            continue
        events.append(Event(LABEL, c0, _end(tracks, (int(ida), int(idb)), c0, cfg, ctx), 0.6, (int(ida), int(idb)), {
            "kind": "road_users", "pair": [int(ida), int(idb)],
            "cls": [int(p["cls_a"].iloc[0]), int(p["cls_b"].iloc[0])],
            "closing_before_contact_h": round(closing, 2), "risk_peak_before_contact": round(risk_peak, 3),
            "impact_score": round(impact, 2), "stopped_at": {str(k): None if v is None else round(v, 2)
                                                             for k, v in stops.items()},
        }))
    return events


def _fixed_object_events(ctx, tracks, cfg) -> list[Event]:
    if ctx.scene.road is None:
        return []
    events = []
    for tid, g in tracks.items():
        if g["cls_major"].iloc[0] not in cfg["vehicle_classes"] or len(g) < 3:
            continue
        pts = footprint_points(g["x1"], g["y1"], g["x2"], g["y2"], cfg["footprint_depth"])
        n, k, _ = pts.shape
        flat = pts.reshape(-1, 2)
        off = ~point_in_polygon(flat, ctx.scene.road)
        for kind in ("islands", "median", "sidewalks"):
            off |= ctx.scene.in_kind(flat, kind)
        edge = cfg["frame_edge_px"]
        inside = ((g["x1"] > edge) & (g["x2"] < ctx.width - edge) & (g["y2"] < ctx.height - edge)).to_numpy()
        hit = off.reshape(n, k).mean(axis=1) >= cfg["fixed_contact_fraction"]
        hit &= inside
        if not hit.any():
            continue
        t = g["t"].to_numpy(float)
        speed = g["speed_h"].to_numpy()
        for c in np.flatnonzero(hit & ~np.r_[False, hit[:-1]]):
            c0 = float(t[c])
            fast = speed[(t >= c0 - cfg["approach_s"]) & (t <= c0)]
            if not len(fast) or fast.max() < cfg["fixed_speed_h"]:
                continue
            if _impact(g, c0, cfg) < cfg["fixed_impact_factor"]:
                continue
            stop = _still_from(g, c0, cfg)
            if stop is None or stop > c0 + cfg["post_window_s"]:
                continue
            events.append(Event(LABEL, c0, _end(tracks, (tid,), c0, cfg, ctx), 0.5, (int(tid),), {
                "kind": "fixed_object", "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]),
                "speed_before_h": round(float(fast.max()), 2), "impact_score": round(_impact(g, c0, cfg), 2),
                "stopped_at": round(stop, 2),
            }))
            break
    return events


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    kin = kinematics(ctx, cfg)
    if kin.empty:
        return []
    tracks = {int(tid): g.sort_values("t").reset_index(drop=True) for tid, g in kin.groupby("track_id")}
    pairs = pair_table(ctx, kin, cfg)
    events = []
    if not pairs.empty and pairs["contact"].any():
        events += _pair_events(ctx, kin, tracks, pairs, risk_pass(ctx), cfg)
    events += _fixed_object_events(ctx, tracks, cfg)
    return events
