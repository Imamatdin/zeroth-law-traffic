"""congestion: traffic at a standstill or crawling across all lanes of a direction. Start when the
queue stops moving; end when it clears (docs/task_spec.md).

A direction is an approach in camera.yaml. Its corridor is the set of flow-atlas cells whose normal
traffic runs that way (enough samples, a consistent heading within corridor_deg of the approach);
mixed cells in the junction belong to no corridor. At each sampled frame the direction is at a
standstill when enough vehicles stand in its corridor, nearly all of them slow, none moving freely,
and the slow ones spread across most of the corridor's width (all lanes, not one queue).

A queue waiting at a red light is normal. On an approach whose signal is known, the standstill must
also persist through min_green_s of green; the event still starts when the queue stopped. On an
approach without a known signal, the standstill must outlast min_unsignalled_s, longer than a cycle.
"""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import VideoContext, along_segment, condition_segments, signed_line_distance
from src.scene.geometry import point_in_polygon

LABEL = "congestion"


def corridor_cells(ctx: VideoContext, approach_id: str, cfg: dict) -> np.ndarray:
    """(ny, nx) bool mask of atlas cells carrying this approach's normal flow on the carriageway."""
    a = ctx.atlas
    d = ctx.approach_direction(approach_id)
    target = np.degrees(np.arctan2(d[1], d[0]))
    dev = np.abs((a.heading - target + 180) % 360 - 180)
    mask = (a.count >= a.min_samples) & (a.resultant >= cfg["min_resultant"]) & (dev <= cfg["corridor_deg"])
    cy, cx = np.mgrid[0:a.ny, 0:a.nx]
    centres = np.stack([(cx.ravel() + 0.5) / a.nx * ctx.width, (cy.ravel() + 0.5) / a.ny * ctx.height], axis=1)
    keep = np.ones(len(centres), bool)
    if ctx.scene.road is not None:
        keep &= ctx.scene.on_carriageway(centres)
    # The junction box belongs to no direction, and the cells queued behind a stop line belong only to
    # that line's approach: otherwise an approach and the exit it feeds (similar headings) share a
    # corridor, and free flow beyond the junction would hide a standstill before it.
    if ctx.scene.intersection is not None:
        keep &= ~point_in_polygon(centres, ctx.scene.intersection)
    for line_id, (p, q) in ctx.scene.stop_lines.items():
        owner = ctx.scene.stop_line_meta.get(line_id, {}).get("approach")
        if owner not in ctx.scene.approaches:
            continue
        behind = (signed_line_distance(centres, p, q, ctx.approach_direction(owner)) < 0)             & along_segment(centres, p, q, cfg["stop_line_margin"])
        keep &= behind if owner == approach_id else ~behind
    return mask & keep.reshape(a.ny, a.nx)


def _controlling_signal(ctx: VideoContext, approach_id: str):
    for line_id, meta in ctx.scene.stop_line_meta.items():
        if meta.get("approach") == approach_id and meta.get("signal") in ctx.signals:
            return ctx.signals[meta["signal"]]
    return None


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    if ctx.atlas is None or not ctx.scene.approaches:
        return []
    a = ctx.atlas
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])]
    edge = cfg["frame_edge_px"]
    veh = veh[((veh["x1"] > edge) & (veh["x2"] < ctx.width - edge) & (veh["y2"] < ctx.height - edge)).to_numpy()]
    frames = np.sort(ctx.samples["frame"].unique())
    t = frames / ctx.fps
    cx = np.clip((veh["gx"].to_numpy() * a.nx).astype(int), 0, a.nx - 1)
    cy = np.clip((veh["gy"].to_numpy() * a.ny).astype(int), 0, a.ny - 1)
    foot = ctx.foot_px(veh)
    events = []
    for approach_id in ctx.scene.approaches:
        cells = corridor_cells(ctx, approach_id, cfg)
        if cells.sum() < cfg["min_corridor_cells"]:
            continue
        d = ctx.approach_direction(approach_id)
        perp = np.array([-d[1], d[0]])
        gy, gx = np.nonzero(cells)
        centres = np.stack([(gx + 0.5) / a.nx * ctx.width, (gy + 0.5) / a.ny * ctx.height], axis=1) @ perp
        width = float(centres.max() - centres.min()) + ctx.width / a.nx
        inside = cells[cy, cx]
        rows = veh[inside]
        lateral = foot[inside] @ perp
        speed = rows["speed_rel"].to_numpy()
        by_frame = {f: idx for f, idx in rows.groupby("frame").indices.items()}
        still = np.zeros(len(frames), bool)
        count = np.zeros(len(frames), int)
        for k, f in enumerate(frames):
            idx = by_frame.get(f)
            if idx is None or len(idx) < cfg["min_vehicles"]:
                continue
            s = speed[idx]
            slow = s < cfg["crawl_rel_speed"]
            count[k] = len(idx)
            if slow.mean() < cfg["min_slow_fraction"] or (s >= cfg["free_rel_speed"]).any():
                continue
            spread = np.ptp(lateral[idx][slow]) if slow.sum() > 1 else 0.0
            still[k] = spread >= cfg["min_lateral_cover"] * width
        signal = _controlling_signal(ctx, approach_id)
        for start, end, i0, i1 in condition_segments(t, still, cfg["min_on_s"], cfg["max_gap_s"], ctx.step_s):
            run = np.zeros(len(t), bool)
            run[i0:i1 + 1] = True
            if signal is not None:
                green_s = float((still & run & (signal.at(t) == "green")).sum() * ctx.step_s)
                if green_s < cfg["min_green_s"]:
                    continue
            else:
                green_s = None
                if end - start < cfg["min_unsignalled_s"]:
                    continue
            events.append(Event(LABEL, start, min(end, ctx.duration), 0.6, (), {
                "approach": approach_id, "signal_known": signal is not None,
                "standstill_green_s": None if green_s is None else round(green_s, 1),
                "median_vehicles": int(np.median(count[i0:i1 + 1])),
                "corridor_cells": int(cells.sum()),
            }))
    return events
