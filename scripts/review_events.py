"""Replay every event engine on every available cache and lay each detection out for a human yes/no.

    python scripts/review_events.py                                  # every cache under cache/, zips unpacked
    python scripts/review_events.py --cache cache/C3896_yolo11m960_s3.zip --video-dir D:/videos
    python scripts/review_events.py --summary-only                   # recount after editing verdicts

A detection is one segment the official output would contain (raw per-track events merged by the
postprocess), with every class switched on. Each gets a frame strip, review/<video>/<label>/<n>.jpg,
and a row in review/<video>/detections.json whose "verdict" (null) the reviewer sets to "yes" (a real
event of that class) or "no" (a false fire), with an optional "note". Verdicts survive re-runs while
the detection keeps its label, tracks and start. review/summary.md and summary.json count detections,
yes, no and pending per class and video; a class is free of false fires on a video only when every
detection there is reviewed "yes", or there are none.

Caches without world.parquet (a plain perception cache) get the Part A world model in memory, as the
runtime builds it. The visual classes and the signal timeline need the video: it is found through the
cache's recorded source path or by name under --video-dir, decoded once, and its samples are kept in
outputs/visual/. Without the video, the visual classes are reported as not run.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_scene import draw
from src.atlas.flow import Atlas
from src.contracts import EVENT_LABELS
from src.events.base import SignalTimeline, VideoContext
from src.events.registry import ENGINES, load_config, run_engines
from src.events.visual import VISUAL_LABELS, VisualSamples, sample_video
from src.features.stitching import prepare_tracks
from src.features.tracks import compute_world
from src.scene.geometry import Scene
from src.scene.signal import classify_head

VISUAL_EVERY_S, VISUAL_WIDTH = 1.0, 640
TRACK_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2"]


def unpack(path: Path, cache_root: Path) -> Path | None:
    """A cache directory; a .zip is extracted under cache_root once. None if no meta.json inside."""
    if path.suffix.lower() == ".zip":
        target = cache_root / path.stem
        if not target.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
        path = target
    metas = sorted(path.glob("meta.json")) or sorted(path.glob("*/meta.json"))
    return metas[0].parent if metas else None


def find_caches(paths: list[Path] | None, cache_root: Path) -> list[Path]:
    candidates = paths or sorted(p for p in cache_root.iterdir() if p.is_dir() or p.suffix.lower() == ".zip")
    out = []
    for p in candidates:
        c = unpack(p, cache_root)
        if c is None or c in out:
            continue
        if not json.loads((c / "meta.json").read_text(encoding="utf-8")).get("complete", False):
            print(f"skipping {c}: cache not complete (still being built?)", flush=True)
            continue
        out.append(c)
    return out


def find_video(meta: dict, video_dirs: list[Path]) -> Path | None:
    recorded = Path(meta.get("request", {}).get("source", {}).get("path", ""))
    if recorded.name and recorded.exists():
        return recorded
    for d in video_dirs:
        for candidate in (d / meta["video_id"], *d.glob(f"{Path(meta['video_id']).stem}.*")):
            if candidate.is_file():
                return candidate
    return None


def load_context(cache: Path, camera: Path, atlas: Path) -> VideoContext:
    if (cache / "world.parquet").exists():
        return VideoContext.from_cache(cache, camera, atlas)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    tracks = pd.read_parquet(cache / "tracks.parquet")
    scene = Scene.load(camera, meta["width"], meta["height"])
    tracks, _ = prepare_tracks(tracks, width=meta["width"], height=meta["height"])
    world = compute_world(tracks, meta["width"], meta["height"], scene=scene)
    samples = world.merge(tracks[TRACK_COLUMNS], on=["frame", "track_id"], validate="one_to_one")
    return VideoContext(meta["video_id"], meta["fps"], meta["width"], meta["height"], meta["duration"],
                        meta["request"]["stride"], samples, scene, {}, Atlas.load(atlas))


def visual_pass(ctx: VideoContext, video: Path, out_dir: Path, need_signals: bool) -> VisualSamples:
    """Visual samples (cached), and the signal timeline read in the same decode when the cache lacks it."""
    stem = f"{ctx.video_id}_{VISUAL_WIDTH}_{VISUAL_EVERY_S}s"
    samples_path, signals_path = out_dir / f"{stem}.npz", out_dir / f"{ctx.video_id}_signals.parquet"
    if samples_path.exists() and (not need_signals or signals_path.exists()):
        return VisualSamples.load(samples_path)
    lamps = {sid: s["lamps_px"] for sid, s in ctx.scene.signals.items() if s.get("lamps_px")}
    rows = []

    def signals(idx, frame):
        if need_signals and idx % 3 == 0:
            rows.extend(dict(frame=idx, t=idx / ctx.fps, signal=sid, state=classify_head(frame, cells))
                        for sid, cells in lamps.items())

    print(f"  decoding {video.name} once for visual samples{' and signals' if need_signals else ''} ...", flush=True)
    vs = sample_video(video, VISUAL_EVERY_S, VISUAL_WIDTH, on_frame=signals if need_signals else None)
    vs.save(samples_path)
    if need_signals:
        pd.DataFrame(rows).to_parquet(signals_path, index=False)
    return vs


def attach_signals(ctx: VideoContext, out_dir: Path) -> None:
    path = out_dir / f"{ctx.video_id}_signals.parquet"
    if not ctx.signals and path.exists():
        df = pd.read_parquet(path)
        ctx.signals = {sid: SignalTimeline.from_series(df, sid) for sid in df["signal"].unique()}


def detection_id(stem: str, seg) -> str:
    return f"{stem}:{seg.label}:{seg.start:.2f}:{'-'.join(str(t) for t in seg.track_ids)}"


def region_of(ctx: VideoContext, seg, pad: float) -> tuple[int, int, int, int]:
    """Crop (source pixels) around the involved tracks or the reported box, else the whole frame."""
    boxes = []
    if seg.track_ids:
        rows = ctx.samples[ctx.samples["track_id"].isin(seg.track_ids)
                           & (ctx.samples["t"] >= seg.start - pad) & (ctx.samples["t"] <= seg.end + pad)]
        boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(float).tolist()
    for ev in _flat_evidence(seg.evidence):
        if "box_source_px" in ev:
            boxes.append(ev["box_source_px"])
    if not boxes:
        return 0, 0, ctx.width, ctx.height
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = max(x2 - x1, 0.2 * ctx.width), max(y2 - y1, 0.2 * ctx.height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = 1.4 * w, 1.4 * h
    return (int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(ctx.width, cx + w / 2)),
            int(min(ctx.height, cy + h / 2)))


def _flat_evidence(ev: dict) -> list[dict]:
    if "merged" in ev:
        return [e for m in ev["merged"] for e in _flat_evidence(m)]
    return [ev]


def render_strip(cap, ctx: VideoContext, seg, path: Path, frames: int, pad: float, tile_h: int) -> None:
    times = np.linspace(max(0.0, seg.start - pad), min(ctx.duration - 1.0 / ctx.fps, seg.end + pad), frames)
    x1, y1, x2, y2 = region_of(ctx, seg, pad)
    cached = np.sort(ctx.samples["frame"].unique())
    involved = set(seg.track_ids)
    tiles = []
    for t in times:
        idx = int(round(t * ctx.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = draw(frame, ctx.scene)
        for line in ctx.scene.solid_lines.values():
            cv2.polylines(frame, [line.astype(np.int32)], False, (0, 255, 255), 3, cv2.LINE_AA)
        if len(cached):
            f = cached[int(np.argmin(np.abs(cached - idx)))]
            for r in ctx.samples[ctx.samples["frame"] == f].itertuples():
                mine = r.track_id in involved
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)),
                              (0, 0, 255) if mine else (160, 160, 160), 5 if mine else 2)
                if mine:
                    cv2.putText(frame, str(r.track_id), (int(r.x1), int(r.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.6, (0, 0, 255), 4)
        for ev in _flat_evidence(seg.evidence):
            if "box_source_px" in ev:
                bx = [int(v) for v in ev["box_source_px"]]
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 255), 5)
        tile = frame[y1:y2, x1:x2]
        tile = cv2.resize(tile, (max(1, round(tile.shape[1] * tile_h / tile.shape[0])), tile_h),
                          interpolation=cv2.INTER_AREA)
        inside = seg.start <= t <= seg.end
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255) if inside else (90, 90, 90), 3)
        cv2.rectangle(tile, (0, 0), (170, 24), (0, 0, 0), -1)
        cv2.putText(tile, f"t={t:.1f}s{' *' if inside else ''}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    body = np.hstack(tiles)
    head = np.zeros((30, body.shape[1], 3), np.uint8)
    cv2.putText(head, f"{seg.label}  {seg.start:.2f}-{seg.end:.2f} s  tracks {list(seg.track_ids)}  "
                      f"(red frame = inside the segment)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def review_video(cache: Path, args, config: dict) -> dict:
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    stem = Path(meta["video_id"]).stem
    print(f"{stem}: {cache}", flush=True)
    ctx = load_context(cache, args.camera, args.atlas)
    video = find_video(meta, args.video_dir)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    if video is not None:
        ctx.visual = visual_pass(ctx, video, args.visual_dir, need_signals=not ctx.signals)
        attach_signals(ctx, args.visual_dir)
    labels = [l for l in (args.labels or ENGINES) if video is not None or l not in VISUAL_LABELS]
    raw, segments = run_engines(ctx, config, labels, only_enabled=False)
    out_dir = args.out / stem
    old_path = out_dir / "detections.json"
    old = {r["id"]: r for r in json.loads(old_path.read_text(encoding="utf-8"))} if old_path.exists() else {}
    cap = cv2.VideoCapture(str(video)) if video is not None and not args.no_strips else None
    rows, counts = [], {}
    for seg in sorted(segments, key=lambda s: (s.label, s.start)):
        n = counts[seg.label] = counts.get(seg.label, 0) + 1
        strip = out_dir / seg.label / f"{n:03d}_{seg.start:07.2f}-{seg.end:07.2f}.jpg"
        if cap is not None:
            render_strip(cap, ctx, seg, strip, args.strip_frames, args.pad, args.tile_h)
        did = detection_id(stem, seg)
        prev = old.get(did, {})
        members = [asdict(e) for e in raw if e.label == seg.label and e.start < seg.end and e.end > seg.start]
        rows.append({"id": did, "video": meta["video_id"], "label": seg.label, "start": round(seg.start, 3),
                     "end": round(seg.end, 3), "track_ids": list(seg.track_ids),
                     "strip": str(strip.relative_to(args.out)) if cap is not None else None,
                     "verdict": prev.get("verdict"), "note": prev.get("note", ""), "raw_events": members})
    if cap is not None:
        cap.release()
    dropped = [r for r in old.values() if r["id"] not in {x["id"] for x in rows} and r.get("verdict") is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    old_path.write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")
    if dropped:
        (out_dir / "retired_verdicts.json").write_text(json.dumps(dropped, indent=1, default=str) + "\n",
                                                       encoding="utf-8")
    print(f"  {len(raw)} raw events -> {len(rows)} detections; video {'found' if video else 'MISSING'}"
          f"{'; ' + str(len(dropped)) + ' reviewed detections no longer produced (retired_verdicts.json)' if dropped else ''}",
          flush=True)
    return {"video": meta["video_id"], "cache": str(cache), "video_found": video is not None,
            "classes_run": sorted(labels)}


def summarise(out: Path, runs: dict[str, dict]) -> None:
    """Per class and video: detections, yes, no, pending, and the false-fire verdict."""
    table: dict[str, dict[str, dict]] = {}
    videos = []
    for det_path in sorted(out.glob("*/detections.json")):
        video = det_path.parent.name
        videos.append(video)
        run = runs.get(video, {})
        rows = json.loads(det_path.read_text(encoding="utf-8"))
        for label in sorted(EVENT_LABELS):
            mine = [r for r in rows if r["label"] == label]
            ran = label in run.get("classes_run", sorted(EVENT_LABELS))
            yes = sum(r["verdict"] == "yes" for r in mine)
            no = sum(r["verdict"] == "no" for r in mine)
            pending = len(mine) - yes - no
            status = ("not run (no video)" if not ran else "no false fires" if no == 0 and pending == 0
                      else f"{no} false fire{'s' * (no != 1)}" + (f", {pending} pending" if pending else "") if no
                      else f"{pending} pending review")
            table.setdefault(label, {})[video] = {"detections": len(mine), "yes": yes, "no": no,
                                                  "pending": pending, "status": status}
    (out / "summary.json").write_text(json.dumps({"videos": videos, "runs": runs, "classes": table}, indent=1) + "\n",
                                      encoding="utf-8")
    lines = ["# Event review: false fires per class", "",
             "Every class forced on. A class is free of false fires on a video only when every detection there",
             "is reviewed \"yes\" (or there are none). This holds for the footage listed, not in general.", "",
             "| class | " + " | ".join(videos) + " |", "|---|" + "---|" * len(videos)]
    for label in sorted(table):
        cells = []
        for v in videos:
            c = table[label].get(v)
            cells.append("-" if c is None else f"{c['detections']} det: {c['yes']} yes / {c['no']} no / "
                                                f"{c['pending']} pending ({c['status']})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, nargs="*", help="cache directories or zips (default: all under --cache-root)")
    p.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    p.add_argument("--video-dir", type=Path, nargs="*", default=[ROOT / "data" / "samples"])
    p.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera.yaml")
    p.add_argument("--events", type=Path, default=ROOT / "configs" / "events.yaml")
    p.add_argument("--atlas", type=Path, default=ROOT / "configs" / "atlas.json")
    p.add_argument("--labels", nargs="*", choices=sorted(ENGINES))
    p.add_argument("--out", type=Path, default=ROOT / "review")
    p.add_argument("--visual-dir", type=Path, default=ROOT / "outputs" / "visual")
    p.add_argument("--strip-frames", type=int, default=5)
    p.add_argument("--pad", type=float, default=1.0, help="seconds shown before and after each detection")
    p.add_argument("--tile-h", type=int, default=260)
    p.add_argument("--no-strips", action="store_true")
    p.add_argument("--summary-only", action="store_true", help="recount the verdicts without replaying")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs_path = a.out / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else {}
    if not a.summary_only:
        config = load_config(a.events)
        for cache in find_caches(a.cache, a.cache_root):
            run = review_video(cache, a, config)
            runs[Path(run["video"]).stem] = run
        runs_path.write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    summarise(a.out, runs)


if __name__ == "__main__":
    main()
