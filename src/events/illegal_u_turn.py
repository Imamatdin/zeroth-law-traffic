"""illegal_u_turn: a U-turn where the road markings or signs prohibit it. Start when the vehicle
starts turning; end when it completes the turn (docs/task_spec.md).

A U-turn is a turn manoeuvre (src/events/turns.py) of u_turn_deg or more. It is illegal only where a
camera.yaml `u_turn` restriction, sourced from a sign, a marking or organizer guidance, covers the
place the vehicle turned: its foot point at any sample of the manoeuvre. Observed traffic never
prohibits a U-turn (D-003); on a map without such restrictions this engine returns nothing.
"""

from __future__ import annotations

from src.contracts import Event
from src.events.base import VideoContext
from src.events.turns import find_turns
from src.scene.geometry import point_in_polygon

LABEL = "illegal_u_turn"


def detect(ctx: VideoContext, cfg: dict) -> list[Event]:
    zones = [r for r in ctx.scene.turn_restrictions if r["kind"] == "u_turn"]
    if not zones:
        return []
    events = []
    veh = ctx.samples[ctx.samples["cls_major"].isin(cfg["vehicle_classes"])].sort_values(["track_id", "t"])
    for tid, g in veh.groupby("track_id", sort=True):
        foot = ctx.foot_px(g)
        for turn in find_turns(ctx, g, cfg):
            if abs(turn.change_deg) < cfg["u_turn_deg"]:
                continue
            path = foot[turn.i0:turn.i1 + 1]
            broken = [r["id"] for r in zones if point_in_polygon(path, r["zone"]).any()]
            if not broken:
                continue
            events.append(Event(LABEL, turn.start, turn.end, 0.7, (int(tid),), {
                "track_id": int(tid), "cls": int(g["cls_major"].iloc[0]),
                "heading_change_deg": round(turn.change_deg, 1), "restrictions": broken,
                "foot_start": [round(float(v), 1) for v in path[0]],
                "foot_end": [round(float(v), 1) for v in path[-1]],
            }))
    return events
