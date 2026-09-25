"""Overlay a normalized JSON geometry draft on an extracted source frame."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    draft = json.loads(a.map.read_text(encoding="utf-8"))
    frame = cv2.imread(str(a.frame))
    if frame is None:
        p.error("Cannot read source frame")
    h, w = frame.shape[:2]
    palette = {"crosswalk": (70, 220, 100), "island": (200, 150, 60),
               "stop_line": (40, 200, 255), "signal": (80, 100, 255)}
    for shape in draft["shapes"]:
        points = np.asarray(shape["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            p.error(f"Invalid normalized points: {shape['id']}")
        xy = np.rint(points * [w - 1, h - 1]).astype(np.int32)
        color = palette[shape["kind"]]
        closed = shape["kind"] != "stop_line"
        cv2.polylines(frame, [xy], closed, color, 2, cv2.LINE_AA)
        x, y = xy[0]
        cv2.putText(frame, shape["id"], (max(0, x), max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for flow in draft.get("observed_flows", []):
        points = np.asarray([flow["from"], flow["to"]], dtype=float)
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            p.error("Invalid flow arrow")
        xy = np.rint(points * [w - 1, h - 1]).astype(int)
        cv2.arrowedLine(frame, tuple(xy[0]), tuple(xy[1]), (255, 240, 160), 3, tipLength=.12)
    frame = cv2.copyMakeBorder(frame, 0, 50, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(frame, "DRAFT: approximate visible geometry; lane legality and signal associations unverified",
                (12, h + 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Green = crossings | blue = refuges | yellow = stop line | red = signal candidates",
                (12, h + 43), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1, cv2.LINE_AA)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(a.out), frame):
        p.error("Could not write overlay")
    print(a.out)


if __name__ == "__main__":
    main()
