"""illegal_turn: a turn from the wrong lane or in a prohibited direction. Start when the vehicle
starts turning; end when it completes the turn (docs/task_spec.md).

A turn is illegal only against a restriction in camera.yaml whose source is a sign, a road marking
or organizer guidance (Scene.load enforces that): a `left`/`right` restriction whose zone holds the
turn's start, a `movement` restriction from its zone into to_zone, or lane arrows that do not allow
the turn in the lane the vehicle was in when it began to turn. Observed traffic never makes a turn
illegal (D-003), so on a map without restrictions this engine returns nothing. U-turns belong to
illegal_u_turn.
"""

from __future__ import annotations

import numpy as np

from src.contracts import Event
from src.events.base import VideoContext
from src.events.turns import find_turns
from src.scene.geometry import point_in_polygon

LABEL = "illegal_turn"


def lane_before(foot: np.ndarray, t: np.ndarray, i0: int, lanes: list[dict], lookback_s: float) -> dict | None:
    """The lane-arrow zone the vehicle was last in, at most lookback_s before it began to turn."""
    for k in range(i0, -1, -1):
        if t[i0] - t[k] > lookback_s:
            break
        for lane in lanes:
            if point_in_polygon(foot[k:k + 1], lane["zone"])[0]:
                return lane
    return None


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    restrictions = [r for r in ctx.scene.turn_restrictions if r["kind"] in ("left", "right", "movement")]
    lanes = ctx.scene.lane_arrows
    if not restrictions and not lanes:
        return []
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])].sort_values(["track_id", "t"])
    for tid, g in veh.groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        foot = ctx.foot_px(g)
        for turn in find_turns(ctx, g, cfg):
            if abs(turn.change_deg) >= cfg["u_turn_deg"]:
                continue
            broken = []
            p0 = foot[turn.i0:turn.i0 + 1]
            for r in restrictions:
                if not point_in_polygon(p0, r["zone"])[0]:
                    continue
                if r["kind"] == turn.direction:
                    broken.append(r["id"])
                elif r["kind"] == "movement" and point_in_polygon(foot[turn.i1:], r["to_zone"]).any():
                    broken.append(r["id"])
            lane = lane_before(foot, t, turn.i0, lanes, cfg["lane_lookback_s"])
            if lane is not None and turn.direction not in lane["allowed"]:
                broken.append(lane["id"])
            if not broken:
                continue
            events.append(Event(LABEL, turn.start, turn.end, 0.7, (int(tid),), {
                "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]), "direction": turn.direction,
                "heading_change_deg": round(turn.change_deg, 1), "restrictions": broken,
                "foot_start": [round(float(v), 1) for v in foot[turn.i0]],
            }))
    return events
