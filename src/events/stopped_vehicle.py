"""stopped_vehicle: a vehicle stationary on the carriageway for 10 s or more, not in a queue at a
signal. Start when it stops; end when it moves again or is removed (docs/task_spec.md).

Most long stops at this camera are normal: signal queues (including movements whose signals are
out of view), bus-stop dwell and kerbside drop-offs. A stationary sample is "explained" when
1. another stationary vehicle stands directly ahead of it (queue chain),
2. it waits on a signal-controlled approach during red, within reach of the stop line, or
3. other vehicles in this video also stood still at this spot (a normal stopping place), or
4. it is inside a standstill: several nearby vehicles are stopped or crawling at the same time
   (a jam, including in the junction where no single flow direction defines "ahead"),
and the event needs min_stop_s of unexplained standing. Part A may look at the whole video, so
rule 3 counts stops at any time; the vehicle's own stop never counts towards it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Event
from src.events.base import VideoContext, signed_line_distance

LABEL = "stopped_vehicle"


def dwell_runs(t: np.ndarray, foot: np.ndarray, h: np.ndarray, speed_rel: np.ndarray, cfg: dict,
               min_s: float) -> list[tuple[int, int]]:
    """Index ranges where the vehicle stays put, by position rather than speed.

    Box jitter on small or partly occluded vehicles makes instantaneous speed spike while the vehicle
    stands still. A dwell starts at a slow sample and lasts while the median-filtered foot point stays
    within dwell_radius_h box heights of its anchor (the median position just after the start).
    """
    n = len(t)
    if n == 0:
        return []
    half = cfg["dwell_median_s"] / 2
    med = np.empty_like(foot)
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    for k in range(n):
        med[k] = np.median(foot[lo[k]:hi[k]], axis=0)
    out, i = [], 0
    while i < n:
        if speed_rel[i] >= cfg["still_rel_speed"]:
            i += 1
            continue
        anchor = med[i]
        j = i
        while j + 1 < n and np.linalg.norm(med[j + 1] - anchor) <= cfg["dwell_radius_h"] * h[j + 1]:
            j += 1
        start = i
        # Backdate over earlier samples already at the stop (their speed read high from jitter).
        floor = out[-1][1] + 1 if out else 0
        while start > floor and np.linalg.norm(med[start - 1] - anchor) <= cfg["dwell_radius_h"] * h[start - 1]:
            start -= 1
        i = start
        if t[j] - t[i] + (t[1] - t[0] if n > 1 else 0.0) >= min_s:
            out.append((i, j))
        i = j + 1
    return out


def _still_runs(ctx: VideoContext, veh: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every dwell >= min_other_stop_s: track, t0, t1, mean foot, median box height."""
    rows = []
    for tid, g in veh.groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        foot = ctx.foot_px(g)
        h = (g["y2"] - g["y1"]).to_numpy()
        for i0, i1 in dwell_runs(t, foot, h, g["speed_rel"].to_numpy(), cfg, cfg["min_other_stop_s"]):
            p = foot[i0:i1 + 1].mean(axis=0)
            rows.append((int(tid), float(t[i0]), float(t[i1]), p[0], p[1], float(np.median(h[i0:i1 + 1]))))
    return pd.DataFrame(rows, columns=["track_id", "t0", "t1", "fx", "fy", "h"])


def _direction(ctx: VideoContext, g: pd.DataFrame, i: int) -> np.ndarray | None:
    """Travel direction at sample i: the atlas's dominant flow there, else the track's own recent motion."""
    if ctx.atlas is not None:
        a = ctx.atlas
        cx = int(np.clip(g["gx"].iloc[i] * a.nx, 0, a.nx - 1))
        cy = int(np.clip(g["gy"].iloc[i] * a.ny, 0, a.ny - 1))
        if a.count[cy, cx] >= a.min_samples and a.resultant[cy, cx] >= 0.8:
            ang = np.radians(a.heading[cy, cx])
            return np.array([np.cos(ang), np.sin(ang)])
    moving = g.iloc[:i + 1]
    moving = moving[moving["speed_rel"] >= 0.5]
    if moving.empty:
        return None
    v = moving[["vx", "vy"]].to_numpy()[-5:].mean(axis=0) * [ctx.width, ctx.height]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])].sort_values(["track_id", "t"])
    if veh.empty:
        return []
    runs = _still_runs(ctx, veh, cfg)
    still_all = veh[veh["speed_rel"] < cfg["still_rel_speed"]]
    still_by_frame = {f: (g["track_id"].to_numpy(), ctx.foot_px(g)) for f, g in still_all.groupby("frame")}
    slow_all = veh[veh["speed_rel"] < cfg["crawl_rel_speed"]]
    slow_by_frame = {f: (g["track_id"].to_numpy(), ctx.foot_px(g)) for f, g in slow_all.groupby("frame")}
    line_id = cfg.get("stop_line")
    line = ctx.scene.stop_lines.get(line_id) if line_id else None
    signal = ctx.signals.get(ctx.scene.stop_line_meta.get(line_id, {}).get("signal")) if line is not None else None
    approach = ctx.approach_direction(cfg["approach"]) if line is not None else None

    events = []
    for tid, g in veh.groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        if t[-1] - t[0] < cfg["min_stop_s"]:
            continue
        foot = ctx.foot_px(g)
        h = (g["y2"] - g["y1"]).to_numpy()
        on_road = ctx.scene.on_carriageway(foot)
        for i0, i1 in dwell_runs(t, foot, h, g["speed_rel"].to_numpy(), cfg, cfg["min_stop_s"]):
            if on_road[i0:i1 + 1].mean() < 0.5:
                continue
            s = float(t[i0])
            d = _direction(ctx, g, i0)
            p_stop = foot[i0:i1 + 1].mean(axis=0)
            h_stop = float(np.median(h[i0:i1 + 1]))
            # Rule 3: a normal stopping place (other tracks stood here, any time in the video).
            others = runs[(runs["track_id"] != tid)]
            near = np.hypot(others["fx"] - p_stop[0], others["fy"] - p_stop[1]) <= cfg["place_radius_h"] * h_stop
            n_place = int(others.loc[near, "track_id"].nunique())
            explained = np.zeros(i1 - i0 + 1, bool)
            if n_place >= cfg["place_min_others"]:
                explained[:] = True
            # Rule 1: queue chain, per sample.
            reason_chain = np.zeros_like(explained)
            if d is not None:
                perp = np.array([-d[1], d[0]])
                for k, i in enumerate(range(i0, i1 + 1)):
                    ids, pts = still_by_frame.get(int(g["frame"].iloc[i]), (np.empty(0), np.empty((0, 2))))
                    m = ids != tid
                    if not m.any():
                        continue
                    rel = pts[m] - foot[i]
                    along, lateral = rel @ d, np.abs(rel @ perp)
                    reason_chain[k] = bool(((along > 0.1 * h[i]) & (along < cfg["queue_gap_h"] * h[i])
                                            & (lateral < cfg["queue_lateral_h"] * h[i])).any())
            # Rule 2: waiting on the controlled approach during red, behind and near the stop line.
            reason_signal = np.zeros_like(explained)
            if line is not None and signal is not None:
                dist = signed_line_distance(foot[i0:i1 + 1], line[0], line[1], approach)
                red = signal.at(t[i0:i1 + 1]) == "red"
                reason_signal = red & (dist < cfg["min_past_box_h"] * h[i0:i1 + 1]) \
                    & (dist > -cfg["signal_reach_h"] * h[i0:i1 + 1])
            # Rule 4: standstill, per sample.
            reason_jam = np.zeros_like(explained)
            for k, i in enumerate(range(i0, i1 + 1)):
                ids, pts = slow_by_frame.get(int(g["frame"].iloc[i]), (np.empty(0), np.empty((0, 2))))
                m = ids != tid
                near_slow = np.hypot(*(pts[m] - foot[i]).T) <= cfg["jam_radius_h"] * h[i] if m.any() else []
                reason_jam[k] = int(np.sum(near_slow)) >= cfg["jam_min_neighbours"]
            explained |= reason_chain | reason_signal | reason_jam
            unexplained = ~explained
            if unexplained.sum() * ctx.step_s < cfg["min_stop_s"]:
                continue
            if cfg.get("exclude_track_end_at_border", True) and i1 == len(t) - 1:
                x1, y1, x2, y2 = (g[c].iloc[i1] for c in ("x1", "y1", "x2", "y2"))
                b = cfg.get("frame_edge_px", 8)
                if x1 <= b or y1 <= b or x2 >= ctx.width - b or y2 >= ctx.height - b:
                    continue                                   # partly out of view: stillness unreliable
            end = float(t[i1 + 1]) if i1 + 1 < len(t) else min(float(t[i1]) + ctx.step_s, ctx.duration)
            events.append(Event(LABEL, s, end, 0.7, (int(tid),), {
                "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]),
                "position": [round(float(v), 1) for v in p_stop],
                "unexplained_s": round(float(unexplained.sum() * ctx.step_s), 1),
                "queue_chain_fraction": round(float(reason_chain.mean()), 2),
                "signal_queue_fraction": round(float(np.mean(reason_signal)), 2),
                "standstill_fraction": round(float(reason_jam.mean()), 2),
                "others_stopped_here": n_place,
                "ends_with_track": bool(i1 == len(t) - 1),
            }))
    return events
