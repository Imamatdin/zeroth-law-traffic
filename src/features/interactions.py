"""Pairwise interaction features per sampled frame (Stage 4.4).

Distances are in the same depth-normalized unit as `speed_rel`: pixel distance divided
by the pair's mean box height, so thresholds do not shrink towards the far road.
tca = max(0, -(r.v)/|v|^2), dmin = |r + tca v|, following ARCHITECTURE.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def pair_features(world: pd.DataFrame, width: int, height: int, k: int = 4,
                  max_dist_rel: float = 12.0) -> pd.DataFrame:
    """Keep each object's k nearest neighbours within max_dist_rel body heights."""
    rows = []
    for frame, g in world.groupby("frame", sort=True):
        if len(g) < 2:
            continue
        p = g[["gx", "gy"]].to_numpy() * [width, height]
        v = g[["vx", "vy"]].to_numpy() * [width, height]
        h = np.maximum(g["box_h"].to_numpy() * height, 1.0)
        ids = g["track_id"].to_numpy()
        cls = g["cls_major"].to_numpy() if "cls_major" in g else g["cls"].to_numpy()
        diff = p[None, :, :] - p[:, None, :]
        dist_px = np.linalg.norm(diff, axis=2)
        scale = (h[:, None] + h[None, :]) / 2
        dist_rel = dist_px / scale
        np.fill_diagonal(dist_rel, np.inf)
        keep = set()
        for i in range(len(g)):
            for j in np.argsort(dist_rel[i])[:k]:
                if dist_rel[i, j] <= max_dist_rel:
                    keep.add((min(i, j), max(i, j)))
        t = float(g["t"].iloc[0])
        for i, j in sorted(keep):
            r = p[j] - p[i]
            dv = v[j] - v[i]
            vv = float(dv @ dv)
            tca = max(0.0, -float(r @ dv) / (vv + EPS))
            dmin_px = float(np.linalg.norm(r + tca * dv))
            dist = float(np.linalg.norm(r))
            closing = -float(r @ dv) / (dist + EPS)
            s = float(scale[i, j])
            hi = np.degrees(np.arctan2(v[i, 1], v[i, 0]))
            hj = np.degrees(np.arctan2(v[j, 1], v[j, 0]))
            rows.append(dict(frame=frame, t=t, id_a=int(ids[i]), id_b=int(ids[j]),
                             cls_a=int(cls[i]), cls_b=int(cls[j]),
                             dist=dist / s, closing_speed=closing / s, tca=tca,
                             dmin=dmin_px / s,
                             heading_diff=float(abs((hi - hj + 180) % 360 - 180))))
    cols = ["frame", "t", "id_a", "id_b", "cls_a", "cls_b", "dist", "closing_speed",
            "tca", "dmin", "heading_diff"]
    return pd.DataFrame(rows, columns=cols)
