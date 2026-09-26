"""Replay the causal risk model over a perception cache, frame by frame in time order.

Uses the raw (unstitched) tracks and each frame's own signal reading, as Part B would see them.
Writes the curve and prints alarm starts computed with the official evaluator's alarm rule.

    python scripts/replay_risk.py --cache cache/C3905_yolo11m960_s3 --out outputs/risk/C3905.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anticipation.risk import RiskModel
from src.contracts import TrackState


def official_alarm_starts():
    spec = importlib.util.spec_from_file_location("official_evaluator", ROOT / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alarm_starts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera.yaml")
    p.add_argument("--atlas", type=Path, default=ROOT / "configs" / "atlas.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.cache / "meta.json").read_text(encoding="utf-8"))
    model = RiskModel.from_files(a.camera, meta["width"], meta["height"], a.atlas)
    model.reset({k: meta[k] for k in ("video_id", "fps", "width", "height", "n_frames")})
    tracks = pd.read_parquet(a.cache / "tracks.parquet")
    frames = pd.read_parquet(a.cache / "frames.parquet")
    series = a.cache / "signal_series.parquet"
    signals = pd.read_parquet(series) if series.exists() else None
    by_frame = dict(tuple(tracks.groupby("frame")))
    curve, evidence = [], []
    for f, t in zip(frames["frame"], frames["t"]):
        g = by_frame.get(f)
        states = [] if g is None else [
            TrackState(int(r.track_id), int(f), float(t), int(r.cls), float(r.score),
                       (float(r.x1), float(r.y1), float(r.x2), float(r.y2))) for r in g.itertuples()]
        sig = None
        if signals is not None:
            rows = signals[signals["frame"] == f]
            sig = dict(zip(rows["signal"], rows["state"]))
        risk = model.update(states, float(t), sig)
        curve.append([round(float(t), 4), round(risk, 4)])
        evidence.append(model.last_evidence)
    starts = official_alarm_starts()(curve)
    raw = np.array([e["raw"] for e in evidence])
    risk = np.array([c[1] for c in curve])
    summary = {
        "video": meta["video_id"], "status": "in-sample; C3905 has no annotated accident",
        "samples": len(curve), "alarm_starts": starts,
        "raw_percentiles": {q: round(float(np.percentile(raw, q)), 4) for q in (50, 90, 99, 99.9, 100)},
        "risk_percentiles": {q: round(float(np.percentile(risk, q)), 4) for q in (50, 90, 99, 99.9, 100)},
        "top": sorted(evidence, key=lambda e: -e["risk"])[:10],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "curve": curve, "evidence": evidence}, indent=1) + "\n",
                     encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
