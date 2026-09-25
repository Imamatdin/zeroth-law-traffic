"""Factory used by scripts/build_cache.py and, later, by models_registry."""

from __future__ import annotations

from src.perception.detector import UltralyticsDetector
from src.perception.tracker import ByteTrackTracker


def ultralytics_bytetrack(meta: dict, config: dict):
    det = config["detector"]
    trk = config.get("tracker", {})
    detector = UltralyticsDetector(det["weights"], imgsz=det.get("imgsz", 960),
                                   conf=det.get("conf", 0.1), iou=det.get("iou", 0.6),
                                   device=det.get("device", "cpu"),
                                   prescale_width=det.get("prescale_width", 1920),
                                   half=det.get("half", False))
    tracker = ByteTrackTracker(meta["fps"], meta["request"]["stride"], **trk)
    return detector, tracker
