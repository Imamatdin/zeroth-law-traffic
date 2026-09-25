"""From a perception cache: world/pairs tables, the flow atlas, and diagnostic overlays.

    python scripts/analyze_cache.py --cache cache/C3905_yolo11m960_s3 --video data/samples/C3905.MP4 \
        --atlas configs/atlas.json --figures data/inspection/C3905/atlas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas.flow import build_atlas
from src.features.interactions import pair_features
from src.features.tracks import compute_world
from src.perception.cache import read_meta

CLASS_COLORS = {0: (60, 220, 255), 1: (255, 160, 60), 2: (255, 80, 200), 3: (80, 255, 120),
                4: (60, 120, 255), 5: (255, 255, 80)}


def reference_frame(video: Path, index: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot read reference frame")
    h = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def heat_overlay(base, points, color_map=cv2.COLORMAP_INFERNO, sigma=6):
    h, w = base.shape[:2]
    heat = np.zeros((h, w), np.float32)
    xs = np.clip((points[:, 0] * w).astype(int), 0, w - 1)
    ys = np.clip((points[:, 1] * h).astype(int), 0, h - 1)
    np.add.at(heat, (ys, xs), 1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    heat = np.log1p(heat)
    heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
    colored = cv2.applyColorMap(heat, color_map)
    mask = (heat > 12)[..., None]
    return np.where(mask, cv2.addWeighted(base, 0.35, colored, 0.65, 0), base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    p.add_argument("--ref-frame", type=int, default=0)
    a = p.parse_args()

    meta = read_meta(a.cache)
    W, H = meta["width"], meta["height"]
    tracks = pd.read_parquet(a.cache / "tracks.parquet")
    world = compute_world(tracks, W, H, mode="centered")
    pairs = pair_features(world, W, H)
    world.to_parquet(a.cache / "world.parquet", index=False)
    pairs.to_parquet(a.cache / "pairs.parquet", index=False)
    atlas = build_atlas(world, pairs)
    atlas["source"] = {"video": meta["video_id"], "cache_request": meta["request"]["pipeline"]["config"],
                       "stride": meta["request"]["stride"]}
    a.atlas.parent.mkdir(parents=True, exist_ok=True)
    a.atlas.write_text(json.dumps(atlas, indent=1) + "\n", encoding="utf-8")

    a.figures.mkdir(parents=True, exist_ok=True)
    base = reference_frame(a.video, a.ref_frame, 1600)
    h, w = base.shape[:2]
    veh = world[world.cls_major != 0]
    ped = world[world.cls_major == 0]
    cv2.imwrite(str(a.figures / "heat_vehicles.jpg"), heat_overlay(base, veh[["gx", "gy"]].to_numpy()))
    cv2.imwrite(str(a.figures / "heat_pedestrians.jpg"),
                heat_overlay(base, ped[["gx", "gy"]].to_numpy(), cv2.COLORMAP_WINTER, 5))
    still = veh[veh.stationary_s > 2.0]
    cv2.imwrite(str(a.figures / "heat_dwell.jpg"), heat_overlay(base, still[["gx", "gy"]].to_numpy()))

    img = base.copy()
    nx, ny = atlas["grid"]["nx"], atlas["grid"]["ny"]
    count = np.asarray(atlas["flow"]["count"])
    hd = np.radians(np.asarray(atlas["flow"]["heading_deg"]))
    res = np.asarray(atlas["flow"]["resultant"])
    for cy in range(ny):
        for cx in range(nx):
            if count[cy, cx] < atlas["params"]["min_samples"]:
                continue
            c = np.array([(cx + 0.5) * w / nx, (cy + 0.5) * h / ny])
            d = np.array([np.cos(hd[cy, cx]), np.sin(hd[cy, cx])]) * 0.45 * w / nx
            color = (80, 255, 120) if res[cy, cx] >= 0.8 else (60, 160, 255)
            cv2.arrowedLine(img, tuple((c - d).astype(int)), tuple((c + d).astype(int)), color, 2,
                            tipLength=0.35)
    cv2.imwrite(str(a.figures / "flow_field.jpg"), img)

    img = base.copy()
    for tid, g in world.groupby("track_id"):
        if g.t.iloc[-1] - g.t.iloc[0] < 1.5:
            continue
        pts = (g[["gx", "gy"]].to_numpy() * [w, h]).astype(np.int32)
        cv2.polylines(img, [pts], False, CLASS_COLORS[int(g.cls_major.iloc[0])], 1, cv2.LINE_AA)
    cv2.imwrite(str(a.figures / "trajectories.jpg"), img)

    summary = {
        "tracks": int(world.track_id.nunique()),
        "tracks_by_class": world.groupby("cls_major").track_id.nunique().to_dict(),
        "world_rows": len(world), "pair_rows": len(pairs),
        "covered_cells": atlas["coverage"], "interaction_baseline": atlas["interaction_baseline"],
        "stops_over_10s": sum(s["duration_s"] >= 10 for s in atlas["dwell"]["stops"]),
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
