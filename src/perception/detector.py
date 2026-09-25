"""Ultralytics detectors behind the Detector protocol, returning source-pixel boxes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.contracts import Detections

# COCO ids -> internal ids (person, bicycle, motorcycle, car, bus, truck).
COCO_TO_INTERNAL = {0: 0, 1: 1, 3: 2, 2: 3, 5: 4, 7: 5}


class UltralyticsDetector:
    """Downscales once before inference so 4K decode output is not letterboxed twice."""

    def __init__(self, weights: str | Path, imgsz: int = 960, conf: float = 0.1,
                 iou: float = 0.6, device: str = "cpu", prescale_width: int = 1920,
                 half: bool = False):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} missing; run weights/download.sh")
        from ultralytics import RTDETR, YOLO

        model_cls = RTDETR if weights.name.startswith("rtdetr") else YOLO
        self.model = model_cls(str(weights))
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device, self.half = device, half
        self.prescale_width = prescale_width
        self.coco_ids = sorted(COCO_TO_INTERNAL)
        self._lut = np.full(max(COCO_TO_INTERNAL) + 1, -1, dtype=np.int64)
        for coco, internal in COCO_TO_INTERNAL.items():
            self._lut[coco] = internal

    def _prescale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        w = frame.shape[1]
        if not self.prescale_width or w <= self.prescale_width:
            return frame, 1.0
        scale = self.prescale_width / w
        h = round(frame.shape[0] * scale)
        return cv2.resize(frame, (self.prescale_width, h), interpolation=cv2.INTER_AREA), scale

    def predict(self, frames: list[np.ndarray]) -> list[Detections]:
        if not frames:
            return []
        scaled = [self._prescale(f) for f in frames]
        extra = {"half": True} if self.half else {}
        results = self.model.predict([s[0] for s in scaled], imgsz=self.imgsz, conf=self.conf,
                                     iou=self.iou, classes=self.coco_ids, device=self.device,
                                     verbose=False, **extra)
        out = []
        for (image, scale), (frame, r) in zip(scaled, zip(frames, results)):
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float64) / scale
            h, w = frame.shape[:2]
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
            cls = self._lut[boxes.cls.cpu().numpy().astype(np.int64)]
            keep = (cls >= 0) & (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            out.append(Detections(xyxy[keep], cls[keep], boxes.conf.cpu().numpy()[keep]))
        return out
