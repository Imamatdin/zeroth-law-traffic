"""Replay event engines from a perception cache (no detector run) and write rich + official output.

    python scripts/replay_events.py --cache cache/C3905_yolo11m960_s3 --labels red_light --all
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events.base import VideoContext
from src.events.registry import ENGINES, load_config, run_engines
from src.postprocess.segments import to_official


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera.yaml")
    p.add_argument("--events", type=Path, default=ROOT / "configs" / "events.yaml")
    p.add_argument("--atlas", type=Path, default=ROOT / "configs" / "atlas.json")
    p.add_argument("--labels", nargs="*", choices=sorted(ENGINES))
    p.add_argument("--all", action="store_true", help="run disabled classes too (development only)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "events")
    a = p.parse_args()
    ctx = VideoContext.from_cache(a.cache, a.camera, a.atlas)
    raw, segments = run_engines(ctx, load_config(a.events), a.labels, only_enabled=not a.all)
    a.out.mkdir(parents=True, exist_ok=True)
    record = {
        "video": ctx.video_id, "duration": ctx.duration, "fps": ctx.fps,
        "status": "in-sample, provisional: no locked ground truth (G1 open)",
        "raw_events": [asdict(e) for e in raw],
        "segments": [asdict(e) for e in segments],
        "official": to_official(segments),
    }
    out = a.out / f"{Path(ctx.video_id).stem}.json"
    out.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    for e in raw:
        ev = {k: v for k, v in e.evidence.items() if k not in ("merged",)}
        print(f"{e.label:<14} {e.start:7.2f}-{e.end:7.2f}  tracks={list(e.track_ids)}  {ev}")
    print(f"{len(raw)} raw events, {len(segments)} segments -> {out}")


if __name__ == "__main__":
    main()
