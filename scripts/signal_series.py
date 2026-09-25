"""Per-frame signal state for every head with lamp cells, plus per-lamp lit fractions."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scene.geometry import Scene
from src.scene.signal import classify_head, lamp_lit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--crops", type=Path, help="optional folder for ROI crops every --crop-every frames")
    p.add_argument("--crop-every", type=int, default=150)
    p.add_argument("--stride", type=int, default=3)
    a = p.parse_args()
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    scene = Scene.load(ROOT / "configs" / "camera.yaml", W, H)
    rows, i = [], 0
    if a.crops:
        a.crops.mkdir(parents=True, exist_ok=True)
    while cap.grab():
        if i % a.stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            for sid, sig in scene.signals.items():
                if not sig["lamps_px"]:
                    continue
                x1, y1, x2, y2 = sig["roi_px"]
                crop = frame[y1:y2, x1:x2]
                lit = {f"lit_{k}": lamp_lit(frame[b:d, a:c]) for k, (a, b, c, d) in sig["lamps_px"].items()}
                rows.append({"frame": i, "t": i / fps, "signal": sid,
                             "state": classify_head(frame, sig["lamps_px"]), **lit})
                if a.crops and i % a.crop_every == 0:
                    cv2.imwrite(str(a.crops / f"{sid}_{i:05d}.png"), cv2.resize(crop, None, fx=3, fy=3,
                                interpolation=cv2.INTER_NEAREST))
        i += 1
    pd.DataFrame(rows).to_parquet(a.out, index=False)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
