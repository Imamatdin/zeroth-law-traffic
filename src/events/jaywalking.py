"""jaywalking: a pedestrian on the carriageway outside a crossing.
Start when the pedestrian steps onto the road; end when they leave it (docs/task_spec.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Event
from src.events.base import PERSON, VideoContext, condition_segments

LABEL = "jaywalking"
TWO_WHEELERS = (1, 2)
ENCLOSING = (3, 4, 5)
SAFE_KINDS = ("crosswalks", "islands", "sidewalks", "median")


def _overlap_of_first(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of each box in a (N,4) covered by each box in b (M,4) -> (N, M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def non_pedestrian_person_rows(samples: pd.DataFrame, rider_overlap: float) -> set[tuple[int, int]]:
    """(frame, track_id) of person detections that are riders or vehicle occupants."""
    out = set()
    box_cols = ["x1", "y1", "x2", "y2"]
    for frame, g in samples.groupby("frame", sort=False):
        people = g[g["cls_major"] == PERSON]
        if people.empty:
            continue
        pb = people[box_cols].to_numpy(float)
        riders = g[g["cls_major"].isin(TWO_WHEELERS)]
        if len(riders):
            rb = riders[box_cols].to_numpy(float)
            grow = 0.25 * np.stack([rb[:, 2] - rb[:, 0], rb[:, 3] - rb[:, 1]], axis=1)
            big = np.concatenate([rb[:, :2] - grow, rb[:, 2:] + grow], axis=1)
            foot = np.stack([(pb[:, 0] + pb[:, 2]) / 2, pb[:, 3]], axis=1)
            foot_on = ((foot[:, None, 0] >= big[None, :, 0]) & (foot[:, None, 0] <= big[None, :, 2])
                       & (foot[:, None, 1] >= big[None, :, 1]) & (foot[:, None, 1] <= big[None, :, 3]))
            hit = ((_overlap_of_first(pb, rb) >= rider_overlap) | foot_on).any(axis=1)
            out.update((frame, int(t)) for t in people["track_id"].to_numpy()[hit])
        vehicles = g[g["cls_major"].isin(ENCLOSING)]
        if len(vehicles):
            vb = vehicles[box_cols].to_numpy(float)
            inside = _overlap_of_first(pb, vb) >= 0.9
            small = (pb[:, None, 3] - pb[:, None, 1]) < 0.6 * (vb[None, :, 3] - vb[None, :, 1])
            hit = (inside & small).any(axis=1)
            out.update((frame, int(t)) for t in people["track_id"].to_numpy()[hit])
    return out


def near_safe_area(ctx: VideoContext, foot: np.ndarray, margin) -> np.ndarray:
    return ctx.scene.near_safe_area(foot, margin, SAFE_KINDS)


def _bridge_walk_through(segments, t: np.ndarray, stationary_s: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Merge on-road segments separated by a short safe-area contact the pedestrian walked through
    without stopping (clipping a refuge tip mid-crossing); stopping on the refuge ends the event."""
    out: list[list[int]] = []
    for _, _, i0, i1 in segments:
        if out:
            p1 = out[-1][1]
            gap = t[i0] - t[p1]
            if (gap <= cfg.get("walk_through_gap_s", 0.0)
                    and np.max(stationary_s[p1:i0 + 1]) < cfg.get("walk_through_max_still_s", 0.0)):
                out[-1][1] = i1
                continue
        out.append([i0, i1])
    return [(a, b) for a, b in out]


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    if ctx.scene.road is None:
        return []
    excluded = non_pedestrian_person_rows(ctx.samples, cfg["rider_overlap"])
    events = []
    people = ctx.samples[ctx.samples["cls_major"] == PERSON]
    for track_id, g in people.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        foot = ctx.foot_px(g)
        pedestrian = np.array([(f, int(track_id)) not in excluded for f in g["frame"].to_numpy()])
        # A box cut by the frame border has no visible feet, so its bottom edge is not a foot point.
        border = cfg.get("frame_edge_px", 4)
        pedestrian &= ((g["y2"] < ctx.height - border) & (g["x1"] > border)
                       & (g["x2"] < ctx.width - border)).to_numpy()
        box_h = (g["y2"] - g["y1"]).to_numpy()
        on_carriageway = ctx.scene.on_carriageway(foot) & pedestrian
        # Two thresholds. The wide margin (~1 m, scaled by body height) confirms an event robustly; the
        # small one (~25 cm) sets its boundaries: stepping onto the road, reaching a crossing, refuge or
        # curb. Zero would let box jitter along a crossing edge drag the start back by seconds.
        margin = np.maximum(cfg["crosswalk_margin_px"], cfg.get("margin_box_h", 0.0) * box_h)
        confirmed = on_carriageway & ~near_safe_area(ctx, foot, margin)
        boundary = np.maximum(cfg.get("boundary_px", 0.0), cfg.get("boundary_box_h", 0.0) * box_h)
        on_road = on_carriageway & ~near_safe_area(ctx, foot, boundary)
        runs = condition_segments(t, confirmed, cfg["min_on_road_s"], cfg["max_gap_s"], ctx.step_s)
        segments = _bridge_walk_through(condition_segments(t, on_road, 0.0, cfg["max_gap_s"], ctx.step_s),
                                        t, g["stationary_s"].to_numpy(), cfg)
        for i0, i1 in segments:
            if not any(r0 <= i1 and r1 >= i0 for _, _, r0, r1 in runs):
                continue
            start = float(t[i0])
            end = min(float(t[i1 + 1]) if i1 + 1 < len(t) else float(t[i1]) + ctx.step_s, ctx.duration)
            speed = float(np.median(g["speed_rel"].to_numpy()[i0:i1 + 1]))
            if speed > cfg.get("max_walk_speed_rel", np.inf):
                continue  # faster than a running person: an unpaired rider or a misdetection
            seg = foot[i0:i1 + 1]
            events.append(Event(LABEL, start, end, 0.7, (int(track_id),), {
                "track_id": int(track_id), "on_road_fraction": round(float(on_road[i0:i1 + 1].mean()), 2),
                "foot_start": [round(float(v), 1) for v in seg[0]],
                "foot_end": [round(float(v), 1) for v in seg[-1]],
                "path_px": round(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()), 1),
                "track_span": [round(float(t[0]), 2), round(float(t[-1]), 2)],
                "median_speed_rel": round(speed, 2),
            }))
    return events
