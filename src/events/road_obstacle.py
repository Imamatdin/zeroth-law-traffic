"""road_obstacle: debris, an animal or a fallen object on the carriageway. Start when the obstacle
appears; end when it is removed (docs/task_spec.md).

Static foreground that no tracked road user explains. Frames are illumination-normalised (divided by
their own large-scale blur, so dusk and exposure drift cancel) and compared with the video's median
background. Over each window the median image keeps what stood still and drops what moved; a
difference on the carriageway that no track box covered for most of the window is a candidate. A
candidate must be sized like an object at its depth (relative to the vehicle boxes there), textured
(a light pool or shadow is smooth), stationary, and persist min_persist_s. A candidate with standing
road users right beside it for most of its life is in a queue, where a vehicle half hidden behind a
pole or another vehicle (no track, so nothing explains it) is far likelier than debris: rejected. Its boundaries are the
first and last samples where the region differs from the background while uncovered.

Limit: an obstacle present for the whole video is part of the background and cannot be seen.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.contracts import Event
from src.events.base import VEHICLES, VideoContext
from src.events.visual import box_coverage, box_height_at, clean_background, components, link_blobs, road_mask

LABEL = "road_obstacle"


def normalised_gray(frames: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty(frames.shape[:3], np.float32)
    for k, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out[k] = g / (cv2.GaussianBlur(g, (0, 0), sigma) + 8.0)
    return out


def edge_energy(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    vs = ctx.visual
    if vs is None or ctx.scene.road is None or len(vs.t) < cfg["min_samples"]:
        return []
    n, h, w = vs.frames.shape[:3]
    norm = normalised_gray(vs.frames, cfg["illum_sigma_px"])
    road = cv2.erode(road_mask(ctx.scene, (h, w), vs.scale).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cover = box_coverage(ctx.samples, vs.t, (h, w), vs.scale, ctx.fps, cfg["box_grow"])
    bg = clean_background(norm, cover)
    bg_edges = edge_energy(bg)
    standing = ctx.samples[ctx.samples["speed_rel"] < cfg["still_rel_speed"]]
    queue = box_coverage(standing, vs.t, (h, w), vs.scale, ctx.fps, cfg["queue_grow"])
    differs = np.abs(norm - bg) > cfg["diff_thr"]
    kernel = np.ones((3, 3), np.uint8)
    starts = np.arange(vs.t[0], max(vs.t[-1] - cfg["window_s"], vs.t[0]) + 1e-9, cfg["window_step_s"])
    per_window, spans = [], []
    for s in starts:
        idx = np.flatnonzero((vs.t >= s) & (vs.t < s + cfg["window_s"]))
        spans.append(idx)
        if len(idx) < 3:
            per_window.append([])
            continue
        med = np.median(norm[idx], axis=0)
        cand = (np.abs(med - bg) > cfg["diff_thr"]) & road & (cover[idx].mean(axis=0) < cfg["max_covered_fraction"])
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        keep = []
        med_edges = edge_energy(med)
        for blob in components(cand, cfg["min_area_px"]):
            x1, y1, x2, y2 = blob["box"]
            ref = box_height_at(ctx.samples, VEHICLES, y2 / vs.scale) * vs.scale
            if not np.isfinite(ref) or not (cfg["min_size_h"] * ref <= y2 - y1 <= cfg["max_size_h"] * ref) \
                    or x2 - x1 > cfg["max_size_h"] * 2 * ref:
                continue
            m = blob["mask"]
            if med_edges[m].mean() < cfg["min_edge_gain"] * max(float(bg_edges[m].mean()), 1e-3):
                continue
            keep.append({**blob, "window": len(per_window)})
        per_window.append(keep)
    events = []
    for chain in link_blobs(per_window, cfg["link_iou"], cfg["link_gap"]):
        k0, first = chain[0]
        k1, last = chain[-1]
        c0 = np.array([(first["box"][0] + first["box"][2]) / 2, (first["box"][1] + first["box"][3]) / 2])
        c1 = np.array([(last["box"][0] + last["box"][2]) / 2, (last["box"][1] + last["box"][3]) / 2])
        if np.linalg.norm(c1 - c0) > cfg["max_drift_box"] * (first["box"][3] - first["box"][1]):
            continue
        region = first["mask"]
        seen = (differs[:, region].mean(axis=1) >= cfg["min_region_differs"]) & (cover[:, region].mean(axis=1) < 0.5)
        span = np.arange(spans[k0][0], spans[k1][-1] + 1) if len(spans[k0]) and len(spans[k1]) else np.array([], int)
        hits = span[seen[span]] if len(span) else span
        if len(hits) == 0:
            continue
        if queue[hits][:, region].any(axis=1).mean() >= cfg["max_queue_fraction"]:
            continue
        start = float(vs.t[hits[0]])
        end = min(float(vs.t[hits[-1]]) + cfg["sample_every_s"], ctx.duration)
        if end - start < cfg["min_persist_s"]:
            continue
        x1, y1, x2, y2 = first["box"]
        events.append(Event(LABEL, start, end, 0.5, (), {
            "box_source_px": [round(v / vs.scale, 1) for v in (x1, y1, x2, y2)],
            "area_px": int(first["area"]), "windows": len(chain),
            "seen_fraction": round(float(len(hits) / max(len(span), 1)), 2),
        }))
    return events
