"""Per-track kinematics from cached tracks (Stage 4.3).

Positions are foot points in normalized image coordinates (no homography: no real
correspondences exist). Because image speed shrinks with depth, `speed_rel` divides
pixel speed by the object's own box height, giving an approximately depth-invariant
"body heights per second" used for stop/move thresholds.

`mode="centered"` may look ahead and is for Part A only; `mode="causal"` uses samples
at or before t and is what Part B must use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scene.geometry import Scene, point_in_polygon

VEHICLE_CLASSES = (1, 2, 3, 4, 5)


def _local_slope(t: np.ndarray, y: np.ndarray, window_s: float, causal: bool) -> np.ndarray:
    """Least-squares slope of y(t) in a window around (or ending at) each sample."""
    out = np.zeros_like(y, dtype=float)
    for i, ti in enumerate(t):
        lo, hi = (ti - window_s, ti) if causal else (ti - window_s / 2, ti + window_s / 2)
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        tt, yy = t[m], y[m]
        if len(tt) < 2:
            continue
        tc = tt - tt.mean()
        denom = (tc * tc).sum()
        if denom > 0:
            out[i] = (tc * (yy - yy.mean())).sum() / denom
    return out


def _stationary_run(t: np.ndarray, still: np.ndarray) -> np.ndarray:
    """Seconds the object has been continuously still up to each sample (causal by construction)."""
    out = np.zeros(len(t))
    start = None
    for i in range(len(t)):
        if still[i]:
            start = t[i] if start is None else start
            out[i] = t[i] - start
        else:
            start = None
    return out


def compute_world(tracks: pd.DataFrame, width: int, height: int, mode: str = "centered",
                  window_s: float = 0.6, still_rel_speed: float = 0.15,
                  scene: Scene | None = None) -> pd.DataFrame:
    if mode not in ("centered", "causal"):
        raise ValueError("mode must be 'centered' or 'causal'")
    causal = mode == "causal"
    parts = []
    for track_id, g in tracks.sort_values(["track_id", "frame"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        fx, fy = g["fx"].to_numpy(float), g["fy"].to_numpy(float)
        box_h = (g["y2"] - g["y1"]).to_numpy(float)
        vx_px = _local_slope(t, fx, window_s, causal)
        vy_px = _local_slope(t, fy, window_s, causal)
        ax_px = _local_slope(t, vx_px, window_s, causal)
        ay_px = _local_slope(t, vy_px, window_s, causal)
        speed_px = np.hypot(vx_px, vy_px)
        if causal:
            ref_h = pd.Series(box_h).expanding().median().to_numpy()
        else:
            ref_h = np.full_like(box_h, np.median(box_h))
        speed_rel = speed_px / np.maximum(ref_h, 1.0)
        counts = g["cls"].value_counts()
        if causal:
            cls_major = g["cls"].expanding().apply(lambda s: s.mode().iloc[0], raw=False).astype(int)
        else:
            cls_major = int(counts.index[0])
        parts.append(pd.DataFrame({
            "frame": g["frame"].to_numpy(), "t": t, "track_id": track_id,
            "cls": g["cls"].to_numpy(), "cls_major": cls_major,
            "gx": fx / width, "gy": fy / height,
            "vx": vx_px / width, "vy": vy_px / height,
            "ax": ax_px / width, "ay": ay_px / height,
            "speed": np.hypot(vx_px / width, vy_px / height),
            "speed_rel": speed_rel,
            "heading": np.degrees(np.arctan2(vy_px, vx_px)),
            "box_h": box_h / height,
            "stationary_s": _stationary_run(t, speed_rel < still_rel_speed),
            "age_s": t - t[0],
        }))
    world = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if scene is not None and len(world):
        pts = world[["gx", "gy"]].to_numpy() * [width, height]
        world["on_carriageway"] = scene.on_carriageway(pts) if scene.road is not None else False
        world["in_crosswalk"] = scene.in_kind(pts, "crosswalks")
        world["in_island"] = scene.in_kind(pts, "islands")
        world["in_intersection"] = (point_in_polygon(pts, scene.intersection)
                                    if scene.intersection is not None else False)
    return world.sort_values(["frame", "track_id"], ignore_index=True)
