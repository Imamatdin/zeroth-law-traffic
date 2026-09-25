"""ByteTrack (Ultralytics implementation) behind the Tracker protocol."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.contracts import Detections, TrackState


class _DetView:
    """The minimal Results-like view BYTETracker consumes: conf, cls, xywh, xyxy, bool indexing."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([self.xyxy[:, :2] + wh / 2, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_DetView":
        return _DetView(self.xyxy[mask], self.conf[mask], self.cls[mask])


class ByteTrackTracker:
    """Track lifetime is set in seconds so behaviour does not change with inference stride."""

    def __init__(self, fps: float, stride: int, track_high_thresh: float = 0.25,
                 track_low_thresh: float = 0.1, new_track_thresh: float = 0.25,
                 match_thresh: float = 0.8, buffer_s: float = 2.0, fuse_score: bool = True):
        self.fps = fps
        steps_per_s = fps / stride
        self.args = SimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh, new_track_thresh=new_track_thresh,
            match_thresh=match_thresh, fuse_score=fuse_score,
            track_buffer=max(1, round(buffer_s * steps_per_s)))
        self.reset()

    def reset(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(self.args)

    def update(self, dets: Detections, frame_idx: int) -> list[TrackState]:
        view = _DetView(dets.xyxy.astype(np.float32), dets.score.astype(np.float32),
                        dets.cls.astype(np.float32))
        rows = self._tracker.update(view)
        t = frame_idx / self.fps
        tracks = []
        for x1, y1, x2, y2, track_id, score, cls, det_idx in rows:
            # The detection box is the observation; the Kalman box would lag on turning vehicles.
            bx1, by1, bx2, by2 = (float(v) for v in dets.xyxy[int(det_idx)])
            tracks.append(TrackState(int(track_id), frame_idx, t, int(cls),
                                     float(min(max(score, 0.0), 1.0)), (bx1, by1, bx2, by2)))
        return tracks
