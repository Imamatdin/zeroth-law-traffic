"""Reject harness errors/timeouts even when the official format validator passes."""
import argparse
import json
from pathlib import Path

import cv2


def check(predictions, videos):
    from run_submission import VIDEO_EXTS
    source = Path(videos)
    paths = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix in VIDEO_EXTS)
    if not paths:
        raise ValueError("No sample videos")
    data = json.loads(Path(predictions).read_text())
    expected = {p.name for p in paths}
    if set(data["videos"]) != expected or set(data["log"]) != expected:
        raise ValueError("Output video inventory differs from input")
    for path in paths:
        record = data["log"][path.name]
        if record.get("errors"):
            raise ValueError(f"{path.name}: {record['errors']}")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot inspect {path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if count <= 0 or len(data["videos"][path.name]["risk"]) != count:
            raise ValueError(f"{path.name}: incomplete Part B output")
        print(f"{path.name}: {count} risk samples; wall/video = "
              f"{record['total_sec'] / record['duration']:.3f}x (rounded harness times)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--videos", required=True)
    args = parser.parse_args()
    check(args.pred, args.videos)
