"""Regenerate the website data for every sample video that has a perception cache, in one command.

For each complete cache under <source-root>/cache/: builds missing derived tables (stitched tracks, world,
pairs via analyze_cache.py; the signal series via signal_series.py), replays the event engines (--all) and
the causal risk model with this checkout's code, then runs export_web.py. Finally checks that every sample
video in <source-root>/data/samples/ made it into web/public/data, and fails if one did not.

    python scripts/refresh_web_data.py                                   # repo with caches
    python scripts/refresh_web_data.py --source-root ../zeroth-law-traffic   # worktree without caches

Existing derived tables in a cache are never rewritten; the atlas analyze_cache.py builds goes to
outputs/web_replay/atlas/, never configs/. The video is decoded again only for new videos or with
--redecode (lighting series and reference frame).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}
# Not samples: clip20.mp4 is the first 20 s of C3905.MP4 (identical frames), cut for the smoke test.
DEFAULT_SKIP = ["clip20.mp4"]


def run(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def complete_caches(src: Path) -> dict[str, Path]:
    by_video: dict[str, list[Path]] = {}
    for meta_path in sorted((src / "cache").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete"):
            by_video.setdefault(meta["video_id"], []).append(meta_path.parent)
    dupes = {v: c for v, c in by_video.items() if len(c) > 1}
    if dupes:
        lines = "\n".join(f"  {v}: {', '.join(p.name for p in c)}" for v, c in dupes.items())
        raise SystemExit(f"More than one complete cache per video; keep exactly one:\n{lines}")
    return {v: c[0] for v, c in by_video.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, default=ROOT, help="checkout holding cache/, data/ and outputs/")
    p.add_argument("--python", default=sys.executable, help="interpreter for the pipeline scripts")
    p.add_argument("--skip", nargs="*", default=DEFAULT_SKIP, help="files in data/samples that are not samples")
    p.add_argument("--redecode", action="store_true", help="decode videos again for lighting and the frame")
    p.add_argument("--allow-missing", action="store_true", help="do not fail when a sample has no cache")
    a = p.parse_args()
    src = a.source_root.resolve()
    py = a.python
    scripts = ROOT / "scripts"
    replay = src / "outputs" / "web_replay"
    web_data = ROOT / "web" / "public" / "data"

    caches = complete_caches(src)
    samples = sorted(f.name for f in (src / "data" / "samples").iterdir()
                     if f.suffix.lower() in VIDEO_EXT and f.name not in a.skip)
    print(f"samples: {', '.join(samples) or 'none'}")
    print(f"caches:  {', '.join(f'{v} -> {c.name}' for v, c in caches.items()) or 'none'}")

    done = []
    for video_id, cache in caches.items():
        if video_id in a.skip:
            continue
        stem = Path(video_id).stem
        video = src / "data" / "samples" / video_id
        if not video.exists():
            raise SystemExit(f"{cache.name}: video {video} not found")
        meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        if not all((cache / f).exists() for f in ("tracks_stitched.parquet", "world.parquet", "pairs.parquet")):
            run(py, scripts / "analyze_cache.py", "--cache", cache, "--video", video,
                "--atlas", replay / "atlas" / f"{stem}.json", "--figures", replay / "figures" / stem)
        if not (cache / "signal_series.parquet").exists():
            run(py, scripts / "signal_series.py", "--video", video, "--out", cache / "signal_series.parquet",
                "--stride", meta["request"]["stride"])
        run(py, scripts / "replay_events.py", "--cache", cache, "--all", "--out", replay / "events")
        run(py, scripts / "replay_risk.py", "--cache", cache, "--out", replay / "risk" / f"{stem}.json")
        export = [py, scripts / "export_web.py", "--source-root", src, "--video", video, "--cache", cache,
                  "--events", replay / "events" / f"{stem}.json", "--risk", replay / "risk" / f"{stem}.json",
                  "--out", web_data]
        if not a.redecode and (web_data / stem / "eda.json").exists() and (web_data / stem / "frame.jpg").exists():
            export.append("--reuse-video")
        run(*export)
        done.append(video_id)

    index = json.loads((web_data / "index.json").read_text(encoding="utf-8")) if (web_data / "index.json").exists() else {"videos": []}
    published = {v["file"] for v in index["videos"]}
    missing = [s for s in samples if s not in published]
    print(f"\nexported this run: {', '.join(done) or 'none'}")
    print(f"on the site:       {', '.join(sorted(published)) or 'none'}")
    if missing:
        msg = f"samples without website data (no complete cache?): {', '.join(missing)}"
        if a.allow_missing:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg)
    stray = sorted(published - set(samples))
    if stray:
        print(f"WARNING: on the site but not in data/samples: {', '.join(stray)}")


if __name__ == "__main__":
    main()
