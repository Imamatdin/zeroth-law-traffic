"""failure_to_yield: a vehicle drives through a crossing while a pedestrian is on it or stepping onto
it. Start when the vehicle enters the crossing; end when it leaves it (docs/task_spec.md).

A vehicle is on a crossing while the lower part of its box (its ground footprint in this oblique view)
overlaps the crossing polygon. A pedestrian counts when they are not a rider or occupant and their foot
point is on the roadway part of the same crossing (not at its kerb or refuge end), or within
step_margin_box_h of it and moving onto it. The conflict is local: the pedestrian must be within
conflict_box_h vehicle box heights of the vehicle's bottom edge, and not behind it, while the vehicle
is moving on the crossing. Someone at the far end of a multi-lane crossing, or a
vehicle that waits and goes once the pedestrian has cleared its path, does not fire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Event
from src.events.base import PERSON, VideoContext, condition_segments
from src.events.jaywalking import non_pedestrian_person_rows
from src.scene.geometry import point_in_polygon

LABEL = "failure_to_yield"


def footprint_points(x1, y1, x2, y2, depth_frac: float) -> np.ndarray:
    """(N, 15, 2) grid over the lower depth_frac of each box: its ground footprint proxy."""
    x1, y1, x2, y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    u = np.linspace(0.0, 1.0, 5)
    w = np.linspace(0.0, 1.0, 3)
    top = y2 - depth_frac * (y2 - y1)
    xs = x1[:, None, None] + (x2 - x1)[:, None, None] * u[None, None, :]
    ys = top[:, None, None] + (y2 - top)[:, None, None] * w[None, :, None]
    xs, ys = np.broadcast_arrays(xs, ys)
    return np.stack([xs, ys], axis=-1).reshape(len(x1), -1, 2)


def footprint_in(polygon: np.ndarray, x1, y1, x2, y2, depth_frac: float) -> np.ndarray:
    pts = footprint_points(x1, y1, x2, y2, depth_frac)
    n, k, _ = pts.shape
    return point_in_polygon(pts.reshape(-1, 2), polygon).reshape(n, k).any(axis=1)


def distance_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point p (M,2) to one segment a-b (2,), vectorised."""
    d = b - a
    s = np.clip(((p - a) @ d) / max(float(d @ d), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + s[:, None] * d), axis=1)


def _pedestrians_by_frame(ctx: VideoContext, polygon: np.ndarray, excluded: set, cfg: dict) -> dict:
    """frame -> (track ids, foot points) of pedestrians on, or stepping onto, this crossing."""
    people = ctx.samples[ctx.samples["cls_major"] == PERSON]
    if people.empty:
        return {}
    foot = ctx.foot_px(people)
    h = (people["y2"] - people["y1"]).to_numpy(float)
    v = ctx.pixel_velocity(people)
    on = point_in_polygon(foot, polygon)
    ahead = foot + v * cfg["step_lookahead_s"]
    margin = cfg["step_margin_box_h"] * h
    near = np.zeros(len(foot), bool)
    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near |= point_in_polygon(foot + margin[:, None] * [ox, oy], polygon)
    stepping = near & point_in_polygon(ahead, polygon) & (people["speed_rel"].to_numpy() >= cfg["min_ped_speed_rel"])
    border = cfg["frame_edge_px"]
    visible = ((people["y2"] < ctx.height - border) & (people["x1"] > border)
               & (people["x2"] < ctx.width - border)).to_numpy()
    real = np.array([(f, int(tid)) not in excluded for f, tid in zip(people["frame"], people["track_id"])])
    # Waiting at the kerb end of a crossing is not being on it: the foot must be on the roadway part.
    kerb = ctx.scene.near_safe_area(foot, cfg["kerb_margin_box_h"] * h, ("islands", "sidewalks", "median"))
    keep = (on | stepping) & visible & real & ~kerb
    sel = people[keep]
    pts = foot[keep]
    out = {}
    for f, idx in pd.Series(np.arange(len(sel)), index=sel["frame"].to_numpy()).groupby(level=0):
        out[int(f)] = (sel["track_id"].to_numpy()[idx.to_numpy()], pts[idx.to_numpy()])
    return out


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    crossings = ctx.scene.regions_of("crosswalks")
    if not crossings:
        return []
    excluded = non_pedestrian_person_rows(ctx.samples, cfg["rider_overlap"])
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])].sort_values(["track_id", "t"])
    events = []
    for region in crossings:
        peds = _pedestrians_by_frame(ctx, region.polygon, excluded, cfg)
        if not peds:
            continue
        for tid, g in veh.groupby("track_id", sort=True):
            t = g["t"].to_numpy(float)
            x1, y1, x2, y2 = (g[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
            # A box cut by the frame bottom has no visible ground contact.
            visible = y2 < ctx.height - cfg["frame_edge_px"]
            on = footprint_in(region.polygon, x1, y1, x2, y2, cfg["footprint_depth"]) & visible
            if not on.any():
                continue
            moving = g["speed_rel"].to_numpy() >= cfg["min_vehicle_speed_rel"]
            v = ctx.pixel_velocity(g)
            heading = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            h = y2 - y1
            frames = g["frame"].to_numpy()
            conflict = np.zeros(len(g), bool)
            closest = np.full(len(g), np.inf)
            who: dict[int, set] = {}
            for k in np.flatnonzero(on & moving):
                hit = peds.get(int(frames[k]))
                if hit is None:
                    continue
                ids, pts = hit
                dist = distance_to_segments(pts, np.array([x1[k], y2[k]]), np.array([x2[k], y2[k]])) / max(h[k], 1.0)
                # A pedestrian the vehicle has already passed is behind it, not in its way.
                ahead = (pts - [(x1[k] + x2[k]) / 2, y2[k]]) @ heading[k] >= -cfg["behind_tolerance_box_h"] * h[k]
                close = (dist <= cfg["conflict_box_h"]) & ahead
                if close.any():
                    conflict[k] = True
                    closest[k] = float(dist.min())
                    who[k] = {int(i) for i in ids[close]}
            if not conflict.any():
                continue
            for s, e, i0, i1 in condition_segments(t, on, 0.0, cfg["max_gap_s"], ctx.step_s):
                inside = np.zeros(len(g), bool)
                inside[i0:i1 + 1] = True
                c = conflict & inside
                if c.sum() * ctx.step_s < cfg["min_conflict_s"]:
                    continue
                # Drives through: a queue standing on the crossing reads as moving only through box jitter.
                speed = float(np.median(g["speed_rel"].to_numpy()[i0:i1 + 1]))
                travel = float(np.hypot(x1[i1] + x2[i1] - x1[i0] - x2[i0], 2 * (y2[i1] - y2[i0])) / 2)
                if speed < cfg["min_through_speed_rel"] or travel < cfg["min_travel_box_h"] * np.median(h[i0:i1 + 1]):
                    continue
                ped_ids = sorted(set().union(*(who[k] for k in np.flatnonzero(c))))
                events.append(Event(LABEL, s, min(e, ctx.duration), 0.6, (int(tid), *ped_ids), {
                    "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]), "crosswalk": region.id,
                    "pedestrians": ped_ids, "conflict_s": round(float(c.sum() * ctx.step_s), 2),
                    "min_dist_box_h": round(float(closest[c].min()), 2),
                    "median_speed_rel": round(speed, 2),
                }))
    return events
