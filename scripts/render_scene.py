"""Draw configs/camera.yaml over an image (a video frame or an atlas heatmap)."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scene.geometry import Scene

COLORS = {"crosswalks": (70, 220, 100), "islands": (200, 150, 60), "sidewalks": (180, 180, 180),
          "median": (160, 100, 200)}


def draw(image: np.ndarray, scene: Scene) -> np.ndarray:
    out = image.copy()
    if scene.road is not None:
        cv2.polylines(out, [scene.road.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    if scene.intersection is not None:
        cv2.polylines(out, [scene.intersection.astype(np.int32)], True, (0, 140, 255), 2, cv2.LINE_AA)
    for region in scene.regions:
        pts = region.polygon.astype(np.int32)
        cv2.polylines(out, [pts], True, COLORS[region.kind], 2, cv2.LINE_AA)
        cv2.putText(out, region.id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[region.kind], 1)
    for name, line in scene.stop_lines.items():
        cv2.line(out, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (40, 200, 255), 3)
    for name, sig in scene.signals.items():
        x1, y1, x2, y2 = sig["roi_px"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera.yaml")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Cannot read image")
    scene = Scene.load(a.camera, image.shape[1], image.shape[0])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), draw(image, scene))
    print(a.out)


if __name__ == "__main__":
    main()
