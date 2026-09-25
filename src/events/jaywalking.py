"""jaywalking: a pedestrian on the carriageway outside a crossing.
Start when the pedestrian steps onto the road; end when they leave it (docs/task_spec.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Event
from src.events.base import PERSON, VideoContext, condition_segments
from src.scene.geometry import point_in_polygon

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
    """On, or within `margin` px (scalar or per point) of, a crossing, refuge, sidewalk, the median
    or the curb (the edge of the road polygon)."""
    m = np.broadcast_to(np.asarray(margin, float), (len(foot),))[:, None]
    safe = np.zeros(len(foot), bool)
    for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        p = foot + m * np.array([ox, oy], float)
        safe |= ~point_in_polygon(p, ctx.scene.road)
        for kind in SAFE_KINDS:
            safe |= ctx.scene.in_kind(p, kind)
    return safe


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
        edge = cfg.get("frame_edge_px", 4)
        pedestrian &= ((g["y2"] < ctx.height - edge) & (g["x1"] > edge) & (g["x2"] < ctx.width - edge)).to_numpy()
        # "On the crossing edge" is about a metre whatever the depth: scale the margin by body height.
        margin = np.maximum(cfg["crosswalk_margin_px"], cfg.get("margin_box_h", 0.0) * (g["y2"] - g["y1"]).to_numpy())
        on_road = ctx.scene.on_carriageway(foot) & ~near_safe_area(ctx, foot, margin) & pedestrian
        # Two thresholds: the margin confirms an event robustly; the exact geometry sets its boundaries
        # (step onto the road / reach a crossing, refuge or curb), as the annotation convention requires.
        strict = ctx.scene.on_carriageway(foot) & ~near_safe_area(ctx, foot, 0.0) & pedestrian
        for _, _, i0, i1 in condition_segments(t, on_road, cfg["min_on_road_s"], cfg["max_gap_s"],
                                               ctx.step_s):
            while i0 > 0 and strict[i0 - 1]:
                i0 -= 1
            while i1 + 1 < len(t) and strict[i1 + 1]:
                i1 += 1
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
