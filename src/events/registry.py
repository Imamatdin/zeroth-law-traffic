"""Engine registry and the Part A event pass over one video context."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts import Event
from src.events import (failure_to_yield, jaywalking, red_light, solid_line_crossing, stop_line, stopped_vehicle,
                        wrong_way)
from src.events.base import VideoContext
from src.postprocess.segments import SegmentRules, postprocess

ENGINES = {"red_light": red_light.detect, "stop_line": stop_line.detect, "jaywalking": jaywalking.detect,
           "wrong_way": wrong_way.detect, "stopped_vehicle": stopped_vehicle.detect,
           "failure_to_yield": failure_to_yield.detect, "solid_line_crossing": solid_line_crossing.detect}


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_engines(ctx: VideoContext, config: dict, labels: list[str] | None = None,
                only_enabled: bool = True) -> tuple[list[Event], list[Event]]:
    """Returns (raw engine events, postprocessed segments)."""
    defaults = config.get("defaults", {})
    raw, rules = [], {}
    for label, detect in ENGINES.items():
        cfg = {**defaults, **config.get(label, {})}
        if labels is not None and label not in labels:
            continue
        if only_enabled and not cfg.get("enabled", False):
            continue
        raw += detect(ctx, cfg)
        rules[label] = SegmentRules(cfg.get("merge_gap_s", 0.0), cfg.get("min_dur_s", 0.0))
    return raw, postprocess(raw, ctx.duration, ctx.fps, rules)
