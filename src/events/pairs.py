"""Per-track kinematics and per-frame pair features for the interaction engines (near_miss, accident).

Part A may look ahead, so velocity and acceleration are least-squares slopes over a centred window.
Distances are in the pair's mean box height and speeds in box heights per second, as in the risk
model, so thresholds hold at every depth of the oblique view. A pair's ground contact proxy is the
overlap of the lower bands of the two boxes (their footprints); a person's footprint is the lower
band of their box too, so a pedestrian at a bumper counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.events.base import PERSON, VideoContext
from src.events.jaywalking import non_pedestrian_person_rows

ROAD_USERS = (0, 1, 2, 3, 4, 5)


def _slope(t: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """Centred least-squares slope of y (N, k) over t, at least three samples per fit."""
    out = np.zeros_like(y, dtype=float)
    lo = np.searchsorted(t, t - window_s / 2, side="left")
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    for i in range(len(t)):
        a, b = lo[i], hi[i]
        if b - a < 3:
            a, b = max(0, i - 1), min(len(t), i + 2)
        tt = t[a:b] - t[a:b].mean()
        d = float(tt @ tt)
        if d > 0:
            out[i] = (tt[:, None] * (y[a:b] - y[a:b].mean(axis=0))).sum(axis=0) / d
    return out


def kinematics(ctx: VideoContext, cfg: dict) -> pd.DataFrame:
    """samples + v (px/s), speed_h, decel_h (h/s^2 along the motion), yaw_deg_s and box height h."""
    parts = []
    for _, g in ctx.samples.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        foot = ctx.foot_px(g)
        h = np.maximum((g["y2"] - g["y1"]).to_numpy(float), 1.0)
        v = _slope(t, foot, cfg["vel_window_s"])
        acc = _slope(t, v, cfg["vel_window_s"])
        speed = np.linalg.norm(v, axis=1)
        ref_h = float(np.median(h))
        unit = np.where(speed[:, None] > 0, v / np.maximum(speed[:, None], 1e-9), 0.0)
        heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        yaw = np.degrees(_slope(t, heading[:, None], cfg["vel_window_s"])[:, 0])
        parts.append(g.assign(px=foot[:, 0], py=foot[:, 1], pvx=v[:, 0], pvy=v[:, 1], h=h,
                              speed_h=speed / ref_h, decel_h=-(acc * unit).sum(axis=1) / ref_h,
                              yaw_deg_s=np.where(speed / ref_h >= cfg["min_heading_speed_h"], yaw, 0.0)))
    return pd.concat(parts, ignore_index=True) if parts else ctx.samples.iloc[:0]


def pair_table(ctx: VideoContext, kin: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per frame and pair of road users within max_pair_dist_h of each other."""
    edge = cfg["frame_edge_px"]
    excluded = non_pedestrian_person_rows(ctx.samples, cfg["rider_overlap"])
    k = kin[kin["cls_major"].isin(ROAD_USERS)
            & (kin["x1"] > edge) & (kin["y1"] > edge) & (kin["x2"] < ctx.width - edge)
            & (kin["y2"] < ctx.height - edge)]
    person = (k["cls_major"] == PERSON).to_numpy()
    rider = np.array([(f, int(t)) in excluded for f, t in zip(k["frame"], k["track_id"])]) & person
    kerb = np.zeros(len(k), bool)
    if ctx.scene.road is not None and person.any():
        pts = k.loc[person, ["px", "py"]].to_numpy()
        kerb[person] = ctx.scene.near_safe_area(pts, cfg["ped_margin_box_h"] * k.loc[person, "h"].to_numpy(),
                                                ("islands", "sidewalks", "median"))
    k = k[~rider & ~kerb]
    cols = {c: k[c].to_numpy() for c in ("frame", "t", "track_id", "cls_major", "px", "py", "pvx", "pvy", "h",
                                          "speed_h", "x1", "x2", "y2")}
    order = np.argsort(cols["frame"], kind="stable")
    bounds = np.flatnonzero(np.diff(cols["frame"][order])) + 1
    rows = []
    for idx in np.split(order, bounds):
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        both_people = (cols["cls_major"][a] == PERSON) & (cols["cls_major"][b] == PERSON)
        scale = (cols["h"][a] + cols["h"][b]) / 2
        r = np.stack([cols["px"][b] - cols["px"][a], cols["py"][b] - cols["py"][a]], axis=1)
        dist = np.linalg.norm(r, axis=1)
        keep = (dist / scale <= cfg["max_pair_dist_h"]) & ~both_people
        if not keep.any():
            continue
        a, b, r, dist, scale = a[keep], b[keep], r[keep], dist[keep], scale[keep]
        dv = np.stack([cols["pvx"][b] - cols["pvx"][a], cols["pvy"][b] - cols["pvy"][a]], axis=1)
        vv = (dv * dv).sum(axis=1)
        tca = np.where(vv > 1e-9, np.maximum(0.0, -(r * dv).sum(axis=1) / np.maximum(vv, 1e-9)), np.inf)
        dmin = np.linalg.norm(r + np.where(np.isfinite(tca), tca, 0.0)[:, None] * dv, axis=1)
        closing = -(r * dv).sum(axis=1) / np.maximum(dist, 1e-9) / scale
        va = np.stack([cols["pvx"][a], cols["pvy"][a]], axis=1)
        vb = np.stack([cols["pvx"][b], cols["pvy"][b]], axis=1)
        cos = (va * vb).sum(axis=1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-9)
        depth = cfg["footprint_depth"]
        top_a, top_b = cols["y2"][a] - depth * cols["h"][a], cols["y2"][b] - depth * cols["h"][b]
        ix = np.minimum(cols["x2"][a], cols["x2"][b]) - np.maximum(cols["x1"][a], cols["x1"][b])
        iy = np.minimum(cols["y2"][a], cols["y2"][b]) - np.maximum(top_a, top_b)
        inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
        area = np.minimum((cols["x2"][a] - cols["x1"][a]) * depth * cols["h"][a],
                          (cols["x2"][b] - cols["x1"][b]) * depth * cols["h"][b])
        swap = cols["track_id"][a] > cols["track_id"][b]
        ida, idb = np.where(swap, cols["track_id"][b], cols["track_id"][a]), np.where(swap, cols["track_id"][a], cols["track_id"][b])
        rows.append(pd.DataFrame({
            "frame": cols["frame"][a], "t": cols["t"][a], "id_a": ida, "id_b": idb,
            "cls_a": np.where(swap, cols["cls_major"][b], cols["cls_major"][a]),
            "cls_b": np.where(swap, cols["cls_major"][a], cols["cls_major"][b]),
            "dist_h": dist / scale, "closing_h": closing, "tca": tca, "dmin_h": dmin / scale, "cos": cos,
            "contact": inter / np.maximum(area, 1e-9) >= cfg["contact_overlap"],
        }))
    cols_out = ["frame", "t", "id_a", "id_b", "cls_a", "cls_b", "dist_h", "closing_h", "tca", "dmin_h", "cos", "contact"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols_out)


def evasive_runs(t: np.ndarray, flag: np.ndarray, min_s: float, max_gap_s: float) -> list[tuple[int, int]]:
    """Index runs of a boolean series lasting min_s, gaps up to max_gap_s bridged."""
    out = []
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return out
    a = b = idx[0]
    for k in idx[1:]:
        if t[k] - t[b] <= max_gap_s:
            b = k
        else:
            out.append((a, b))
            a = b = k
    out.append((a, b))
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return [(a, b) for a, b in out if t[b] - t[a] + step >= min_s]


__all__ = ["kinematics", "pair_table", "evasive_runs", "ROAD_USERS"]
