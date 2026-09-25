"""Review sheets for a time window: exact frames with cached track boxes, IDs and the scene map.

    python scripts/render_review.py --video data/samples/C3905.MP4 --cache cache/C3905_yolo11m960_s3 \
        --start 10 --end 31 --every 3 --out data/inspection/C3905/review/jay_10_31 [--ids 12 16 18]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scene.geometry import Scene
from scripts.render_scene import draw

COLORS = {0: (60, 220, 255), 1: (255, 160, 60), 2: (255, 80, 200), 3: (80, 255, 120),
          4: (60, 120, 255), 5: (255, 255, 80)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    p.add_argument("--ids", type=int, nargs="*", help="highlight only these track IDs")
    p.add_argument("--crop", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   help="normalized crop of the frame")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tracks = pd.read_parquet(a.cache / "tracks.parquet")
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    scene = Scene.load(ROOT / "configs" / "camera.yaml", W, H)
    cached = np.sort(tracks.frame.unique())
    tiles = []
    t = a.start
    while t <= a.end + 1e-9:
        f = int(cached[np.argmin(np.abs(cached - round(t * fps)))])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw(frame, scene)
        for r in tracks[tracks.frame == f].itertuples():
            if a.ids and r.track_id not in a.ids:
                continue
            c = COLORS[int(r.cls)]
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3)
            cv2.circle(frame, (int(r.fx), int(r.fy)), 7, c, -1)
            cv2.putText(frame, str(r.track_id), (int(r.x1), int(r.y1) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, c, 3)
        if a.crop:
            x1, y1, x2, y2 = a.crop
            frame = frame[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]
        h = round(frame.shape[0] * a.width / frame.shape[1])
        tile = cv2.resize(frame, (a.width, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (260, 26), (0, 0, 0), -1)
        cv2.putText(tile, f"f{f}  t={f / fps:.2f}s", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
        t += a.every
    a.out.mkdir(parents=True, exist_ok=True)
    per = a.cols * 2
    for k in range(0, len(tiles), per):
        page = tiles[k:k + per]
        while len(page) % a.cols:
            page.append(np.zeros_like(page[0]))
        rows = [np.hstack(page[i:i + a.cols]) for i in range(0, len(page), a.cols)]
        cv2.imwrite(str(a.out / f"sheet_{k // per + 1:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(tiles)} frames -> {a.out}")


if __name__ == "__main__":
    main()
