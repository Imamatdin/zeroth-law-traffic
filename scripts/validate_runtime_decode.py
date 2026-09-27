"""Compare FFmpeg preprocessing with an existing cache; NEVER build or modify that cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare(old, new, height):
    a, b = old[["x1", "y1", "x2", "y2"]].to_numpy(), new.xyxy
    far = (a[:, 3] - a[:, 1]) <= 0.03 * height
    matches = []
    if len(a) and len(b):
        wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                        np.maximum(a[:, None, :2], b[None, :, :2]))
        inter = wh.prod(axis=2)
        union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1) - inter
        iou = inter / np.maximum(union, 1e-9)
        valid = (old.cls.to_numpy()[:, None] == new.cls[None, :]) & (iou >= 0.5)
        ii, jj = linear_sum_assignment(-np.where(valid, 1 + iou, 0))
        matches = [(int(i), int(j), float(iou[i, j])) for i, j in zip(ii, jj) if valid[i, j]]
    return dict(cache_count=len(a), new_count=len(b), matched=len(matches),
                matched_ious=[x[2] for x in matches], far_cache=int(far.sum()),
                far_matched=sum(bool(far[i]) for i, _, _ in matches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)  # Preserve the user's exact string.
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=32)
    a = p.parse_args()
    meta = json.loads((a.cache / "meta.json").read_text())
    source = meta["request"]["source"]
    stat = Path(a.video).stat()
    if a.video != source["path"] or stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
        raise ValueError("Source path/size/mtime does not match the existing cache; refusing comparison")
    from src.models_registry import new_detector, WEIGHTS, CONFIG, DEVICE
    from src.runtime.decode import frames, restore
    from src.runtime.pipeline import signal_rows
    from src.scene.geometry import Scene
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != meta["request"]["pipeline"]["weights"][0]["sha256"]:
        raise ValueError("Checkpoint differs from cache")
    if CONFIG != meta["request"]["pipeline"]["config"]:
        raise ValueError("Perception config differs from cache")
    chosen = set((np.linspace(0, (meta["n_frames"] - 1) // 3, a.frames).round().astype(int) * 3).tolist())
    old = pd.read_parquet(a.cache / "detections.parquet")
    detector = new_detector()
    samples, signals, rows, scene = [], [], [], None
    started = time.perf_counter()
    for idx, frame in frames(a.video, meta):
        if scene is None:
            scene = Scene.load(ROOT / "configs/camera.yaml", frame.shape[1], frame.shape[0])
        signals.extend(signal_rows(frame, idx, meta["fps"], scene))
        if idx in chosen:
            dets = restore(detector.predict([frame])[0], frame.shape, meta)
            row = compare(old[old.frame == idx], dets, meta["height"])
            samples.append({"frame": idx, **row})
            print(f"frame={idx} cache={row['cache_count']} new={row['new_count']} matched={row['matched']}", flush=True)
            rows.extend(dict(frame=idx, cls=int(c), score=float(s), x1=float(b[0]), y1=float(b[1]),
                             x2=float(b[2]), y2=float(b[3])) for b, c, s in zip(dets.xyxy, dets.cls, dets.score))
    elapsed = time.perf_counter() - started
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out / "new_detections.parquet", index=False)
    new_signals = pd.DataFrame(signals)
    new_signals.to_parquet(a.out / "new_signals.parquet", index=False)
    old_signals = pd.read_parquet(a.cache / "signal_series.parquet")
    joined = old_signals.merge(new_signals, on=["frame", "signal"], suffixes=("_cache", "_new"), validate="one_to_one")
    signal_report = []
    for sid, g in joined.groupby("signal"):
        entry = dict(signal=sid, samples=len(g), changed=int((g.state_cache != g.state_new).sum()),
                     changes=g.groupby(["state_cache", "state_new"]).size().rename("count").reset_index().to_dict("records"))
        for name in ("red", "yellow", "green"):
            key = "lit_" + name
            if key + "_cache" in g:
                diff = (g[key + "_cache"] - g[key + "_new"]).abs()
                entry[key] = dict(mean_absolute=float(diff.mean()), max_absolute=float(diff.max()))
        signal_report.append(entry)
    totals = {key: sum(r[key] for r in samples) for key in ("cache_count", "new_count", "matched", "far_cache", "far_matched")}
    ious = [v for r in samples for v in r["matched_ious"]]
    totals.update(mean_matched_iou=float(np.mean(ious)) if ious else None,
                  cache_match_recall=totals["matched"] / max(1, totals["cache_count"]),
                  far_cache_match_recall=totals["far_matched"] / max(1, totals["far_cache"]))
    report = dict(source=a.video, source_identity_checked=True, cache_rebuilt=False, device=DEVICE,
                  detector_frames=sorted(chosen), wall_s=elapsed, totals=totals, signals=signal_report,
                  signal_cache_rows=len(old_signals), signal_compared_rows=len(joined),
                  far_definition="cache box height <= 3% native height; proxy recall, not labelled truth",
                  samples=samples)
    (a.out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
