"""Extract exact-index review sheets from real video; reuse identical requests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def extract(video: Path, out: Path, start: int, end: int, stride: int,
            width: int = 640, columns: int = 3, rows: int = 3) -> dict:
    if start < 0 or end < start or min(stride, width, columns, rows) <= 0:
        raise ValueError("Invalid frame range or sheet dimensions")
    stat = video.stat()
    request = {"source": str(video.resolve()), "source_size": stat.st_size,
               "source_mtime_ns": stat.st_mtime_ns, "start_frame": start,
               "end_frame": end, "stride": stride, "width": width,
               "columns": columns, "rows": rows, "version": 1}
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request") != request:
            raise ValueError("Existing review uses another source or request; choose a new --out directory")
        if not all((out / p["file"]).is_file() for p in manifest["pages"]):
            raise ValueError("Review cache is incomplete; choose a new --out directory")
        print(f"Reused {len(manifest['pages'])} sheets; no video decoded")
        return manifest
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output has no complete manifest; choose a new --out directory")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or end >= count:
            raise ValueError(f"Frame range exceeds video ({count} frames, {fps} fps)")
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("Decoder cannot seek to requested frame")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"request": request, "fps": fps, "n_frames": count,
                    "duration": count / fps, "pages": []}
        sheet = None
        entries = []
        cell_height = 0

        def save_page() -> None:
            name = f"sheet_{len(manifest['pages']) + 1:03d}.jpg"
            if not cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(f"Could not save {name}")
            manifest["pages"].append({"file": name, "frames": entries.copy()})

        for frame_idx in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decode stopped at frame {frame_idx}; no complete manifest written")
            if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame_idx + 1:
                raise ValueError("Decoder position disagrees with requested frame index")
            if (frame_idx - start) % stride:
                continue
            height = round(frame.shape[0] * width / frame.shape[1])
            cell_height = height + 28
            if sheet is None:
                sheet = np.zeros((cell_height * rows, width * columns, 3), dtype=np.uint8)
            tile = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            i = len(entries)
            x, y = (i % columns) * width, (i // columns) * cell_height
            sheet[y:y + height, x:x + width] = tile
            cv2.putText(sheet, f"frame {frame_idx} | {frame_idx / fps:.4f} s",
                        (x + 8, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            entries.append({"frame": frame_idx, "t": frame_idx / fps})
            if len(entries) == columns * rows:
                save_page()
                entries.clear()
                sheet = None
        if entries:
            save_page()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Created {len(manifest['pages'])} sheets in {out}")
        return manifest
    finally:
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start-frame", required=True, type=int)
    p.add_argument("--end-frame", required=True, type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--columns", type=int, default=3)
    p.add_argument("--rows", type=int, default=3)
    a = p.parse_args()
    try:
        extract(a.video, a.out, a.start_frame, a.end_frame, a.stride, a.width, a.columns, a.rows)
    except (ValueError, OSError) as e:
        p.exit(2, f"review_frames: {e}\n")


if __name__ == "__main__":
    main()
