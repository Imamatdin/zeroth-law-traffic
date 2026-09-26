"""Contact sheet of stitching links: A's last box beside B's first box, for visual verification."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def crop(cap, tracks, track_id, which, half=260, size=220):
    g = tracks[tracks.track_id == track_id].sort_values("t")
    r = g.iloc[-1] if which == "last" else g.iloc[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    ok, f = cap.read()
    cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 0, 255), 4)
    cx, cy = int((r.x1 + r.x2) / 2), int((r.y1 + r.y2) / 2)
    h = max(half, int(1.5 * (r.y2 - r.y1)))
    H, W = f.shape[:2]
    c = f[max(0, cy - h):min(H, cy + h), max(0, cx - h):min(W, cx + h)]
    c = cv2.resize(c, (size, size))
    cv2.putText(c, f"{track_id} {which} t={r.t:.1f}", (4, 16), 0, 0.45, (255, 255, 255), 1)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True, help="tracks parquet the links refer to")
    p.add_argument("--links", type=Path, required=True)
    p.add_argument("--rows", nargs="*", type=int, help="link row indices to show")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tracks = pd.read_parquet(a.tracks)
    links = pd.read_csv(a.links)
    rows = a.rows if a.rows else range(len(links))
    cap = cv2.VideoCapture(str(a.video))
    tiles = []
    for i in rows:
        l = links.iloc[i]
        pair = np.hstack([crop(cap, tracks, int(l["from"]), "last"), crop(cap, tracks, int(l["to"]), "first")])
        pair = cv2.copyMakeBorder(pair, 18, 4, 2, 2, cv2.BORDER_CONSTANT)
        cv2.putText(pair, f"#{i} gap={l.gap_s:.1f}s d={l.dist_h:.2f}/{l.allowed_h:.2f}h", (4, 13), 0, 0.42,
                    (0, 255, 255), 1)
        tiles.append(pair)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
    print(a.out)


if __name__ == "__main__":
    main()
