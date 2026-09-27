"""Export pipeline output for the website, read-only.

Reads one video's perception cache, the event and risk replays, the signal series, the flow atlas and
camera.yaml, and decodes the video once for lighting and a reference frame. Writes compact JSON and a
JPG per video under web/public/data/<video_id>/ plus web/public/data/index.json. Nothing here feeds
back into the pipeline.

    python scripts/export_web.py --video data/samples/C3905.MP4 --cache cache/C3905_yolo11m960_s3 \
        --events outputs/web_replay/events/C3905.json --risk outputs/web_replay/risk/C3905.json

Relative inputs resolve against --source-root (default: this checkout), so a worktree without caches
can point at the main checkout.
"""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import OBJECT_CLASSES
from src.scene.signal import smooth_states

VEHICLES = (1, 2, 3, 4, 5)
Q = 10000  # normalized image coordinates are stored as integers in 1/10000 of the frame


def rel(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def dump(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text)


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runs(values, times, end_t):
    out = []
    for i, v in enumerate(values):
        if out and out[-1][2] == v:
            continue
        if out:
            out[-1][1] = round(float(times[i]), 3)
        out.append([round(float(times[i]), 3), None, v])
    if out:
        out[-1][1] = round(float(end_t), 3)
    return out


def export_tracks(cache: Path, meta: dict, keep_every: int, min_samples: int):
    tracks = pd.read_parquet(cache / "tracks_stitched.parquet")
    world = pd.read_parquet(cache / "world.parquet")[["frame", "track_id", "speed_rel", "heading", "stationary_s"]]
    frames = pd.read_parquet(cache / "frames.parquet")
    df = tracks.merge(world, on=["frame", "track_id"], how="left")
    k_of_frame = {int(f): i for i, f in enumerate(frames["frame"])}
    df["k"] = df["frame"].map(k_of_frame)
    W, H = meta["width"], meta["height"]
    out, raw_to_track = [], {}
    for tid, g in df.sort_values("frame").groupby("track_id"):
        if len(g) < min_samples:
            continue
        for r in g["raw_track_id"].unique():
            raw_to_track[int(r)] = int(tid)
        w = (g["x2"] - g["x1"]).rolling(5, center=True, min_periods=1).median()
        h = (g["y2"] - g["y1"]).rolling(5, center=True, min_periods=1).median()
        fx = g["fx"].rolling(3, center=True, min_periods=1).mean()
        fy = g["fy"].rolling(3, center=True, min_periods=1).mean()
        sel = (g["k"] % keep_every == 0).to_numpy().copy()
        sel[0] = sel[-1] = True
        cls = int(g["cls"].mode().iloc[0])
        out.append({
            "id": int(tid),
            "cls": cls,
            "raw": sorted(int(r) for r in g["raw_track_id"].unique()),
            "k": g["k"].to_numpy()[sel].astype(int).tolist(),
            "x": np.round(fx.to_numpy()[sel] / W * Q).astype(int).tolist(),
            "y": np.round(fy.to_numpy()[sel] / H * Q).astype(int).tolist(),
            "w": np.round(w.to_numpy()[sel] / W * Q).astype(int).tolist(),
            "h": np.round(h.to_numpy()[sel] / H * Q).astype(int).tolist(),
            "v": np.round(g["speed_rel"].fillna(0).to_numpy()[sel] * 100).astype(int).tolist(),
            "hd": np.round(g["heading"].fillna(0).to_numpy()[sel]).astype(int).tolist(),
            "st": np.round(g["stationary_s"].fillna(0).to_numpy()[sel] * 10).astype(int).tolist(),
        })
    return out, raw_to_track, df


def export_risk(risk_rec: dict, cache: Path, meta: dict, raw_to_track: dict):
    raw_tracks = pd.read_parquet(cache / "tracks.parquet")
    W, H = meta["width"], meta["height"]
    by_id = {int(i): g.sort_values("t") for i, g in raw_tracks.groupby("track_id")}

    def pose(tid, t):
        g = by_id.get(tid)
        if g is None:
            return None
        past = g[(g["t"] <= t + 1e-6) & (g["t"] >= t - 0.5)]
        if past.empty:
            return None
        p = past.iloc[-1]
        if len(past) >= 2:
            dt = past["t"].iloc[-1] - past["t"].iloc[0]
            v = ((past["fx"].iloc[-1] - past["fx"].iloc[0]) / dt, (past["fy"].iloc[-1] - past["fy"].iloc[0]) / dt)
        else:
            v = (0.0, 0.0)
        return float(p["fx"]), float(p["fy"]), v

    pairs = []
    for i, e in enumerate(risk_rec["evidence"]):
        if "pair" not in e:
            continue
        a, b = (int(x) for x in e["pair"])
        pa, pb = pose(a, e["t"]), pose(b, e["t"])
        if pa is None or pb is None:
            continue
        tca = min(float(e["tca_s"]), 5.0)
        cx = ((pa[0] + pa[2][0] * tca) + (pb[0] + pb[2][0] * tca)) / 2
        cy = ((pa[1] + pa[2][1] * tca) + (pb[1] + pb[2][1] * tca)) / 2
        pairs.append({
            "i": i, "a": raw_to_track.get(a), "b": raw_to_track.get(b), "raw_ids": [a, b],
            "cls": e["cls"], "tca": e["tca_s"], "dmin_h": e["dmin_h"], "closing_h": e["closing_h_per_s"],
            "dist_h": e["dist_h"], "held": e["course_held_s"], "flags": e["flags"],
            "px": int(round(np.clip(cx / W, -0.2, 1.2) * Q)), "py": int(round(np.clip(cy / H, -0.2, 1.2) * Q)),
        })
    ev = risk_rec["evidence"]
    return {
        "status": risk_rec["summary"]["status"],
        "threshold": 0.5,
        "horizon_s": 5.0,
        "alarm_starts": risk_rec["summary"]["alarm_starts"],
        "percentiles": {"raw": risk_rec["summary"]["raw_percentiles"],
                        "risk": risk_rec["summary"]["risk_percentiles"]},
        "t": [round(float(c[0]), 3) for c in risk_rec["curve"]],
        "risk": [float(c[1]) for c in risk_rec["curve"]],
        "raw": [float(e["raw"]) for e in ev],
        "smoothed": [float(e["smoothed"]) for e in ev],
        "pairs": pairs,
        "note": "risk = calibrated P(accident within 5 s); raw/smoothed = pair score before calibration. "
                "Pair ids are Part B's own (unstitched) tracks mapped to the replayed tracks where possible. "
                "px/py = midpoint of both road users extrapolated by tca along their recent image velocity, "
                "for display only.",
    }


def official_classes(solution: Path) -> list[str]:
    """CLASSES from solution.py, read without importing it (importing loads the models)."""
    tree = ast.parse(solution.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(getattr(t, "id", None) == "CLASSES" for t in targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit(f"CLASSES not found in {solution}")


def export_events(ev_rec: dict, events_cfg: dict, classes: list[str]):
    enabled = {k: bool(v.get("enabled")) for k, v in events_cfg.items() if isinstance(v, dict) and "enabled" in v}

    def clean(e):
        return {"label": e["label"], "start": round(e["start"], 3), "end": round(e["end"], 3),
                "confidence": e["confidence"], "track_ids": e["track_ids"], "evidence": e["evidence"]}

    segs = [clean(e) for e in ev_rec["segments"]]
    submitted = [[s["start"], s["end"], s["label"]] for s in segs if enabled.get(s["label"])]
    return {
        "status": ev_rec["status"],
        "mode": "development replay with every engine forced on (--all); not the submission output",
        "classes": classes,
        "enabled_in_submission": enabled,
        "submitted": submitted,
        "raw_events": [clean(e) for e in ev_rec["raw_events"]],
        "segments": segs,
    }


def export_signal(cache: Path, meta: dict):
    """Raw per-sample readings (what Part B sees) and the smoothed timeline the Part A engines use.
    `bridged` lists raw gaps the smoothing filled with the surrounding state."""
    s = pd.read_parquet(cache / "signal_series.parquet").sort_values("frame")
    out = {}
    for sid, g in s.groupby("signal"):
        t = g["t"].to_numpy()
        raw = g["state"].astype(str).tolist()
        smooth = smooth_states(raw)
        bridged = [[s0, s1] for s0, s1, v in runs([r if r == m else "bridged" for r, m in zip(raw, smooth)],
                                                   t, meta["duration"]) if v == "bridged"]
        out[sid] = {"raw": runs(raw, t, meta["duration"]), "smoothed": runs(smooth, t, meta["duration"]),
                    "bridged": bridged}
    crossings_path = cache / "near_stop_crossings.csv"
    crossings = []
    if crossings_path.exists():
        c = pd.read_csv(crossings_path)
        crossings = [{"t": round(float(r.t), 3), "track": int(r.track_id), "cls": int(r.cls)} for r in c.itertuples()]
    return {"signals": out, "near_stop_crossings": crossings,
            "note": "raw: per-frame lamp classifier (unknown = no lamp clearly lit). smoothed: "
                    "src.scene.signal.smooth_states, as used by the Part A engines."}


def export_field(cache: Path, keep_every: int, max_tca: float, max_dmin: float, max_dist: float):
    """Every plausible approaching pair from pairs.parquet, safe ones included, with the midpoint of both
    road users extrapolated to closest approach. Distances are in box heights (depth-normalised)."""
    pairs = pd.read_parquet(cache / "pairs.parquet")
    world = pd.read_parquet(cache / "world.parquet")[["frame", "track_id", "gx", "gy", "vx", "vy"]]
    frames = pd.read_parquet(cache / "frames.parquet")
    k_of_frame = {int(f): i for i, f in enumerate(frames["frame"])}
    m = ((pairs["closing_speed"] > 0) & (pairs["tca"] <= max_tca) & (pairs["dmin"] <= max_dmin)
         & (pairs["dist"] <= max_dist) & ~((pairs["cls_a"] == 0) & (pairs["cls_b"] == 0)))
    p = pairs[m].copy()
    p["k"] = p["frame"].map(k_of_frame)
    p = p[p["k"] % keep_every == 0]
    wa = world.rename(columns=lambda c: c if c == "frame" else c + "_a").rename(columns={"track_id_a": "id_a"})
    wb = world.rename(columns=lambda c: c if c == "frame" else c + "_b").rename(columns={"track_id_b": "id_b"})
    p = p.merge(wa, on=["frame", "id_a"]).merge(wb, on=["frame", "id_b"]).sort_values(["k", "id_a", "id_b"])
    cx = ((p["gx_a"] + p["vx_a"] * p["tca"]) + (p["gx_b"] + p["vx_b"] * p["tca"])) / 2
    cy = ((p["gy_a"] + p["vy_a"] * p["tca"]) + (p["gy_b"] + p["vy_b"] * p["tca"])) / 2
    return {
        "filter": {"closing_speed": "> 0", "tca_s": f"<= {max_tca}", "dmin_h": f"<= {max_dmin}",
                   "dist_h": f"<= {max_dist}", "pedestrian_pairs": "excluded", "keep_every": keep_every},
        "k": p["k"].astype(int).tolist(),
        "a": p["id_a"].astype(int).tolist(), "b": p["id_b"].astype(int).tolist(),
        "px": np.round(np.clip(cx, -0.2, 1.2) * Q).astype(int).tolist(),
        "py": np.round(np.clip(cy, -0.2, 1.2) * Q).astype(int).tolist(),
        "tca": np.round(p["tca"] * 10).astype(int).tolist(),
        "dmin": np.round(p["dmin"] * 100).astype(int).tolist(),
        "units": {"tca": "seconds x 10", "dmin": "box heights x 100", "px,py": "normalized x Q"},
        "note": "Part A world pairs (stitched ids). Midpoint extrapolation is for display only.",
    }


def export_eda(df: pd.DataFrame, meta: dict, lighting: dict, signal: dict, video: Path, atlas: dict):
    W, H = meta["width"], meta["height"]
    dur = meta["duration"]
    bins = np.arange(0, np.ceil(dur) + 1)
    df = df.copy()
    df["sec"] = np.floor(df["t"]).astype(int)
    n_samples_per_sec = df.groupby("sec")["frame"].nunique()
    counts = {}
    for c in range(len(OBJECT_CLASSES)):
        g = df[df["cls"] == c].groupby("sec").size()
        mean = (g / n_samples_per_sec).reindex(range(len(bins) - 1)).fillna(0)
        counts[OBJECT_CLASSES[c]] = np.round(mean.to_numpy(), 2).tolist()
    unique = {OBJECT_CLASSES[c]: int(df[df["cls"] == c]["track_id"].nunique()) for c in range(len(OBJECT_CLASSES))}
    nx, ny = 64, 36

    def heat(sub):
        gx = np.clip((sub["fx"] / W * nx).astype(int), 0, nx - 1)
        gy = np.clip((sub["fy"] / H * ny).astype(int), 0, ny - 1)
        m = np.zeros((ny, nx), int)
        np.add.at(m, (gy.to_numpy(), gx.to_numpy()), 1)
        return m.tolist()

    moving = df[(df["cls"].isin(VEHICLES)) & (df["speed_rel"] >= atlas["params"]["moving_rel_speed"])]
    edges = np.round(np.arange(0, 8.01, 0.25), 2)
    speeds = {}
    for c in VEHICLES:
        v = moving[moving["cls"] == c]["speed_rel"].to_numpy()
        if len(v) < 20:
            continue
        hist, _ = np.histogram(np.clip(v, 0, edges[-1] - 1e-9), bins=edges)
        speeds[OBJECT_CLASSES[c]] = {"hist": hist.tolist(), "median": round(float(np.median(v)), 2),
                                     "p90": round(float(np.percentile(v, 90)), 2), "n": int(len(v))}
    phases = {}
    for sid, rec in signal["signals"].items():
        rr = rec["smoothed"]
        for s0, s1, st in rr:
            phases.setdefault(sid, {}).setdefault(st, []).append(round(s1 - s0, 2))
    return {
        "video": {"file": video.name, "width": W, "height": H, "fps": round(meta["fps"], 3),
                  "n_frames": meta["n_frames"], "duration_s": round(dur, 3),
                  "size_mb": round(video.stat().st_size / 1e6, 1),
                  "bitrate_mbps": round(video.stat().st_size * 8 / dur / 1e6, 1),
                  "cache_stride": meta["request"]["stride"],
                  "detector": meta["request"]["pipeline"]["config"]["detector"]["weights"],
                  "detector_imgsz": meta["request"]["pipeline"]["config"]["detector"]["imgsz"],
                  "pipeline": meta["request"]["pipeline"]["config"],
                  "weights": meta["request"]["pipeline"]["weights"],
                  "packages": meta["request"]["pipeline"]["packages"]},
        "classes": list(OBJECT_CLASSES),
        "counts_per_second": counts,
        "unique_tracks": unique,
        "heatmaps": {"nx": nx, "ny": ny, "vehicles": heat(df[df["cls"].isin(VEHICLES)]),
                     "pedestrians": heat(df[df["cls"] == 0])},
        "speed_rel": {"unit": "box heights per second (depth-normalised image speed)", "edges": edges.tolist(),
                      "by_class": speeds},
        "signal_phases_s": phases,
        "lighting": lighting,
    }


def decode(video: Path, out_jpg: Path, meta: dict, road_poly, ref_frame: int, jpg_width: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = meta["fps"]
    step = max(1, int(round(fps)))
    small_w = 480
    small_h = int(round(meta["height"] * small_w / meta["width"]))
    mask = np.zeros((small_h, small_w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(road_poly) * [small_w, small_h]).astype(np.int32)], 1)
    ts, luma, road, sat = [], [], [], []
    i = 0
    wrote = False
    while True:
        want = i % step == 0 or i == ref_frame
        ok = cap.grab()
        if not ok:
            break
        if want:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if i == ref_frame and not wrote:
                h = int(round(meta["height"] * jpg_width / meta["width"]))
                cv2.imwrite(str(out_jpg), cv2.resize(frame, (jpg_width, h), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 84])
                wrote = True
            if i % step == 0:
                small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                y = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
                ts.append(round(i / fps, 3))
                luma.append(round(float(y.mean()), 1))
                road.append(round(float(y[mask > 0].mean()), 1))
                sat.append(round(float(hsv[..., 2].mean()), 1))
        i += 1
    cap.release()
    return {"t": ts, "mean_luma": luma, "road_luma": road, "mean_value": sat,
            "note": "8-bit decoded luma (0-255) of one frame per second, downscaled to 480 px wide"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True, help="replay_events.py output (rich events)")
    p.add_argument("--risk", type=Path, required=True, help="replay_risk.py output (curve + evidence)")
    p.add_argument("--camera", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--atlas", type=Path, default=Path("configs/atlas.json"))
    p.add_argument("--events-config", type=Path, default=Path("configs/events.yaml"))
    p.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "data")
    p.add_argument("--keep-every", type=int, default=2, help="keep every N-th cached sample of each track")
    p.add_argument("--min-samples", type=int, default=5, help="drop tracks seen in fewer cached samples")
    p.add_argument("--ref-frame", type=int, default=0, help="frame index for the reference JPG")
    p.add_argument("--jpg-width", type=int, default=1920)
    p.add_argument("--reuse-video", action="store_true",
                   help="keep frame.jpg and the lighting series from a previous export instead of decoding")
    a = p.parse_args()
    src = a.source_root.resolve()
    video, cache = rel(src, a.video), rel(src, a.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    camera = yaml.safe_load(rel(src, a.camera).read_text(encoding="utf-8"))
    atlas = json.loads(rel(src, a.atlas).read_text(encoding="utf-8"))
    events_cfg = yaml.safe_load(rel(src, a.events_config).read_text(encoding="utf-8"))
    ev_rec = json.loads(rel(src, a.events).read_text(encoding="utf-8"))
    risk_rec = json.loads(rel(src, a.risk).read_text(encoding="utf-8"))
    vid = Path(meta["video_id"]).stem
    out = a.out / vid
    out.mkdir(parents=True, exist_ok=True)

    frames = pd.read_parquet(cache / "frames.parquet")
    tracks, raw_to_track, df = export_tracks(cache, meta, a.keep_every, a.min_samples)
    sizes = {}
    sizes["replay.json"] = dump(out / "replay.json", {
        "video": meta["video_id"], "fps": meta["fps"], "duration": meta["duration"],
        "width": meta["width"], "height": meta["height"], "q": Q,
        "t": np.round(frames["t"].to_numpy(), 3).tolist(),
        "classes": list(OBJECT_CLASSES),
        "units": {"x,y": "foot point (bottom-centre of the box), normalized x Q", "w,h": "box size, normalized x Q",
                  "v": "speed_rel x 100 (box heights per second)", "hd": "image-space heading, degrees, y down",
                  "st": "stationary seconds x 10", "k": "index into t"},
        "tracks": tracks,
    })
    signal = export_signal(cache, meta)
    sizes["signal.json"] = dump(out / "signal.json", signal)
    sizes["field.json"] = dump(out / "field.json", export_field(cache, a.keep_every, 5.0, 3.0, 8.0))
    classes = official_classes(src / "solution.py")
    sizes["events.json"] = dump(out / "events.json", export_events(ev_rec, events_cfg, classes))
    sizes["risk.json"] = dump(out / "risk.json", export_risk(risk_rec, cache, meta, raw_to_track))
    sizes["scene.json"] = dump(out / "scene.json", {"camera": camera, "source": "configs/camera.yaml"})
    sizes["atlas.json"] = dump(out / "atlas.json", {
        k: atlas[k] for k in ("grid", "params", "flow", "dwell", "pedestrians", "movements",
                              "interaction_baseline", "coverage", "note")})
    previous = out / "eda.json"
    if a.reuse_video and previous.exists() and (out / "frame.jpg").exists():
        lighting = json.loads(previous.read_text(encoding="utf-8"))["lighting"]
    else:
        lighting = decode(video, out / "frame.jpg", meta, camera["road"], a.ref_frame, a.jpg_width)
    sizes["eda.json"] = dump(out / "eda.json", export_eda(df, meta, lighting, signal, video, atlas))

    index_path = a.out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"videos": []}
    index["videos"] = [v for v in index["videos"] if v["id"] != vid] + [{
        "id": vid, "file": meta["video_id"], "duration": round(meta["duration"], 3), "fps": round(meta["fps"], 3),
        "width": meta["width"], "height": meta["height"], "ref_frame": a.ref_frame,
        "pipeline_commit": git_head(src), "status": ev_rec["status"],
    }]
    index["videos"].sort(key=lambda v: v["id"])
    dump(index_path, index)
    for k, v in sizes.items():
        print(f"{vid}/{k:12s} {v / 1e3:8.1f} kB")
    print(f"{vid}: {len(tracks)} tracks, {len(signal['near_stop_crossings'])} stop-line crossings -> {out}")


if __name__ == "__main__":
    main()
