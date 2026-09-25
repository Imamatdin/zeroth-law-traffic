"""Flow atlas: a per-camera model of normal traffic learned from unlabeled tracks (Stage 5).

It describes what is common, never what is legal (decision D-003). Cells with too few
samples are marked low-confidence and must not drive a rule on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VEHICLES = (1, 2, 3, 4, 5)
PERSON = 0


def _cell(gx, gy, nx, ny):
    cx = np.clip((np.asarray(gx) * nx).astype(int), 0, nx - 1)
    cy = np.clip((np.asarray(gy) * ny).astype(int), 0, ny - 1)
    return cx, cy


def build_atlas(world: pd.DataFrame, pairs: pd.DataFrame | None = None, grid=(32, 18),
                moving_rel_speed: float = 0.5, min_samples: int = 15, min_age_s: float = 1.0) -> dict:
    nx, ny = grid
    veh = world[world["cls_major"].isin(VEHICLES) & (world["age_s"] >= min_age_s)]
    moving = veh[veh["speed_rel"] >= moving_rel_speed]
    cx, cy = _cell(moving["gx"], moving["gy"], nx, ny)
    ang = np.radians(moving["heading"].to_numpy())
    count = np.zeros((ny, nx), int)
    sin_sum = np.zeros((ny, nx))
    cos_sum = np.zeros((ny, nx))
    np.add.at(count, (cy, cx), 1)
    np.add.at(sin_sum, (cy, cx), np.sin(ang))
    np.add.at(cos_sum, (cy, cx), np.cos(ang))
    with np.errstate(invalid="ignore", divide="ignore"):
        resultant = np.hypot(sin_sum, cos_sum) / count
    heading = np.degrees(np.arctan2(sin_sum, cos_sum))

    speed = {}
    for (x, y), g in moving.assign(cx=cx, cy=cy).groupby(["cx", "cy"]):
        speed[f"{x},{y}"] = [float(q) for q in np.percentile(g["speed_rel"], [10, 50, 90])]

    vx, vy = _cell(veh["gx"], veh["gy"], nx, ny)
    all_count = np.zeros((ny, nx), int)
    still_count = np.zeros((ny, nx), int)
    np.add.at(all_count, (vy, vx), 1)
    still = (veh["stationary_s"] > 0).to_numpy()
    np.add.at(still_count, (vy[still], vx[still]), 1)

    stops = []
    for tid, g in veh.groupby("track_id"):
        runs = g["stationary_s"].to_numpy()
        ends = np.where((runs[:-1] > 0) & (runs[1:] == 0))[0].tolist()
        if len(runs) and runs[-1] > 0:
            ends.append(len(runs) - 1)
        for i in ends:
            stops.append(dict(gx=float(g["gx"].iloc[i]), gy=float(g["gy"].iloc[i]),
                              duration_s=float(runs[i]), track_id=int(tid)))

    movements = []
    for tid, g in veh.groupby("track_id"):
        if g["t"].iloc[-1] - g["t"].iloc[0] < 2.0:
            continue
        movements.append(dict(track_id=int(tid),
                              start=[float(g["gx"].iloc[0]), float(g["gy"].iloc[0])],
                              end=[float(g["gx"].iloc[-1]), float(g["gy"].iloc[-1])],
                              duration_s=float(g["t"].iloc[-1] - g["t"].iloc[0])))

    ped = world[world["cls_major"] == PERSON]
    px, py = _cell(ped["gx"], ped["gy"], nx, ny)
    ped_count = np.zeros((ny, nx), int)
    np.add.at(ped_count, (py, px), 1)

    baseline = {}
    if pairs is not None and len(pairs):
        vv = pairs[pairs["cls_a"].isin(VEHICLES) & pairs["cls_b"].isin(VEHICLES)
                   & (pairs["closing_speed"] > 0)]
        for col in ("tca", "dmin", "closing_speed", "dist"):
            if len(vv):
                baseline[col] = {f"p{q}": float(np.percentile(vv[col], q)) for q in (1, 5, 10, 50)}
        baseline["n_pairs"] = int(len(vv))

    covered = count >= min_samples
    return {
        "version": 1,
        "grid": {"nx": nx, "ny": ny, "coords": "normalized"},
        "params": {"moving_rel_speed": moving_rel_speed, "min_samples": min_samples,
                   "min_age_s": min_age_s},
        "flow": {"count": count.tolist(), "heading_deg": np.where(count > 0, heading, 0).round(2).tolist(),
                 "resultant": np.nan_to_num(resultant).round(4).tolist(), "speed_rel_p10_50_90": speed},
        "dwell": {"vehicle_samples": all_count.tolist(), "still_samples": still_count.tolist(),
                  "stops": stops},
        "pedestrians": {"count": ped_count.tolist()},
        "movements": movements,
        "interaction_baseline": baseline,
        "coverage": {"covered_cells": int(covered.sum()), "total_cells": nx * ny},
        "note": "Describes common movement only; not evidence of legality (D-003).",
    }


class Atlas:
    def __init__(self, data: dict):
        self.data = data
        self.nx, self.ny = data["grid"]["nx"], data["grid"]["ny"]
        self.count = np.asarray(data["flow"]["count"])
        self.heading = np.asarray(data["flow"]["heading_deg"])
        self.resultant = np.asarray(data["flow"]["resultant"])
        self.min_samples = data["params"]["min_samples"]

    @classmethod
    def load(cls, path: str | Path) -> "Atlas":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def heading_deviation(self, gx, gy, heading_deg, min_resultant: float = 0.8) -> np.ndarray:
        """Absolute angle to the cell's dominant flow; NaN where coverage or consistency is low."""
        cx, cy = _cell(gx, gy, self.nx, self.ny)
        ok = (self.count[cy, cx] >= self.min_samples) & (self.resultant[cy, cx] >= min_resultant)
        dev = np.abs((np.asarray(heading_deg) - self.heading[cy, cx] + 180) % 360 - 180)
        return np.where(ok, dev, np.nan)
