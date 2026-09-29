"""fire_smoke: visible fire or smoke from a vehicle or on the road. Start at the first visible smoke
(or flame); end when it clears or the frame ends (docs/task_spec.md).

Two conservative visual detectors over the downscaled samples; either makes an event.
- Fire: saturated flame-coloured pixels (orange to yellow, bright, red over green over blue) on or
  beside the carriageway, outside the signal heads and outside the boxes of moving road users (tail
  lights, indicators), in a blob sized for its depth that persists fire_min_s and flickers (its area
  varies; lamps and reflections are steady). A blob on a standing vehicle must be large against that
  vehicle: a bus's orange LED destination sign scrolls (and so "flickers") but is a thin strip.
- Smoke: grey haze, brighter and less textured than the background at the same place, not covered by
  any road user, large for its depth, persisting smoke_min_s and growing or drifting.
Depth scale is the vehicle box height at the blob's foot row. For smoke, each sample is divided by its
global gain against the median background (dusk and exposure drift cancel) but not by a local blur,
which would cancel a large soft plume.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.contracts import Event
from src.events.base import VEHICLES, VideoContext
from src.events.road_obstacle import edge_energy
from src.events.visual import box_coverage, box_height_at, clean_background, components, link_blobs, polygon_mask

LABEL = "fire_smoke"


def _near_road(ctx: VideoContext, shape, scale: float, grow_px: int) -> np.ndarray:
    road = polygon_mask(ctx.scene.road, shape, scale).astype(np.uint8)
    return cv2.dilate(road, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)).astype(bool)


def _signal_mask(ctx: VideoContext, shape, scale: float) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for sig in ctx.scene.signals.values():
        x1, y1, x2, y2 = (int(round(v * scale)) for v in sig["roi_px"])
        mask[max(0, y1 - 2):y2 + 3, max(0, x1 - 2):x2 + 3] = True
    return mask


def _texture(img: np.ndarray) -> np.ndarray:
    """Local texture: edge energy averaged over a 5x5 neighbourhood (per-pixel Sobel on grain is noise)."""
    return cv2.blur(edge_energy(img), (5, 5))


def _vehicle_boxes(ctx: VideoContext, times: np.ndarray, scale: float) -> list[np.ndarray]:
    """Vehicle boxes (sample pixels) at the cached frame nearest to each sample time."""
    veh = ctx.samples[ctx.samples["cls_major"].isin(VEHICLES)]
    frames = np.sort(ctx.samples["frame"].unique())
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) * scale for f, g in veh.groupby("frame")}
    empty = np.empty((0, 4))
    return [by_frame.get(frames[int(np.argmin(np.abs(frames - t * ctx.fps)))], empty) if len(frames) else empty
            for t in times]


def _small_on_vehicle(blob: dict, boxes: np.ndarray, min_frac: float) -> bool:
    x1, y1, x2, y2 = blob["box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    on = (boxes[:, 0] <= cx) & (cx <= boxes[:, 2]) & (boxes[:, 1] <= cy) & (cy <= boxes[:, 3])
    return bool(on.any() and (y2 - y1) < min_frac * (boxes[on, 3] - boxes[on, 1]).max())


def _sized(ctx: VideoContext, blob: dict, scale: float, min_h: float) -> bool:
    ref = box_height_at(ctx.samples, VEHICLES, blob["box"][3] / scale) * scale
    return bool(np.isfinite(ref) and blob["area"] >= (min_h * ref) ** 2)


def _backdate(k0: int, box, unsized: list[list[dict]], max_gap: int) -> int:
    """Earliest sample before k0 reached through smaller blobs overlapping `box`: the first visible
    smoke or flame, before it grew large enough to confirm."""
    first, miss, k = k0, 0, k0 - 1
    while k >= 0 and miss <= max_gap:
        if any(b["box"][0] < box[2] and b["box"][2] > box[0] and b["box"][1] < box[3] and b["box"][3] > box[1]
               for b in unsized[k]):
            first, miss = k, 0
        else:
            miss += 1
        k -= 1
    return first


def _chains_to_events(ctx, vs, chains, unsized, kind, min_s, accept, max_gap) -> list[Event]:
    out = []
    step = float(np.median(np.diff(vs.t))) if len(vs.t) > 1 else 1.0
    for chain in chains:
        k0, k1 = chain[0][0], chain[-1][0]
        if vs.t[k1] + step - vs.t[k0] < min_s or not accept(chain):
            continue
        areas = np.array([b["area"] for _, b in chain], float)
        k0 = _backdate(k0, chain[0][1]["box"], unsized, max_gap)
        start, end = float(vs.t[k0]), min(float(vs.t[k1]) + step, ctx.duration)
        out.append(Event(LABEL, start, end, 0.5, (), {
            "kind": kind, "box_source_px": [round(v / vs.scale, 1) for v in chain[0][1]["box"]],
            "samples": len(chain), "area_px_median": int(np.median(areas)),
            "area_cv": round(float(areas.std() / max(areas.mean(), 1.0)), 2),
        }))
    return out


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    vs = ctx.visual
    if vs is None or ctx.scene.road is None or len(vs.t) < cfg["min_samples"]:
        return []
    n, h, w = vs.frames.shape[:3]
    area = _near_road(ctx, (h, w), vs.scale, cfg["road_grow_px"]) & ~_signal_mask(ctx, (h, w), vs.scale)
    moving_boxes = box_coverage(ctx.samples, vs.t, (h, w), vs.scale, ctx.fps, cfg["box_grow"], moving_only=True,
                                moving_rel=cfg["moving_rel"])
    any_boxes = box_coverage(ctx.samples, vs.t, (h, w), vs.scale, ctx.fps, cfg["box_grow"])
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in vs.frames]).astype(np.float32)
    bg = clean_background(gray, any_boxes)
    bg_edges = _texture(bg)
    gain = np.median(gray[:, area], axis=1) / max(float(np.median(bg[area])), 1.0)
    vehicles = _vehicle_boxes(ctx, vs.t, vs.scale)
    fire_blobs, smoke_blobs, fire_all, smoke_all = [], [], [], []
    for k in range(n):
        f = vs.frames[k]
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        b, g, r = (f[..., i].astype(int) for i in range(3))
        flame = ((hsv[..., 0] >= cfg["fire_hue"][0]) & (hsv[..., 0] <= cfg["fire_hue"][1])
                 & (hsv[..., 1] >= cfg["fire_min_sat"]) & (hsv[..., 2] >= cfg["fire_min_val"])
                 & (r > g) & (g > b) & area & ~moving_boxes[k])
        fire_all.append(components(flame, cfg["min_area_px"]))
        fire_blobs.append([bl for bl in fire_all[-1] if _sized(ctx, bl, vs.scale, cfg["fire_min_size_h"])
                           and not _small_on_vehicle(bl, vehicles[k], cfg["fire_on_vehicle_min_h"])])
        level = gray[k] / max(float(gain[k]), 1e-3)
        haze = ((level - bg >= cfg["smoke_diff"]) & (hsv[..., 1] <= cfg["smoke_max_sat"])
                & (_texture(level) <= cfg["smoke_edge_ratio"] * bg_edges + 1e-3) & area & ~any_boxes[k])
        haze = cv2.morphologyEx(haze.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        smoke_all.append(components(haze, cfg["min_area_px"]))
        smoke_blobs.append([bl for bl in smoke_all[-1] if _sized(ctx, bl, vs.scale, cfg["smoke_min_size_h"])])

    def flickers(chain):
        a = np.array([bl["area"] for _, bl in chain], float)
        return a.std() / max(a.mean(), 1.0) >= cfg["fire_min_flicker"]

    def evolves(chain):
        a = np.array([bl["area"] for _, bl in chain], float)
        c = np.array([[(bl["box"][0] + bl["box"][2]) / 2, (bl["box"][1] + bl["box"][3]) / 2] for _, bl in chain])
        size = np.sqrt(np.median(a))
        return a.max() / max(a.min(), 1.0) >= cfg["smoke_min_growth"] or \
            np.linalg.norm(c - c[0], axis=1).max() >= cfg["smoke_min_drift"] * size

    events = _chains_to_events(ctx, vs, link_blobs(fire_blobs, cfg["link_iou"], cfg["link_gap"]), fire_all,
                               "fire", cfg["fire_min_s"], flickers, cfg["link_gap"])
    events += _chains_to_events(ctx, vs, link_blobs(smoke_blobs, cfg["link_iou"], cfg["link_gap"]), smoke_all,
                                "smoke", cfg["smoke_min_s"], evolves, cfg["link_gap"])
    return events
