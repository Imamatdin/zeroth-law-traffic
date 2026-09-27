"""Observation-clock tracking, with private IDs for each pipeline consumer."""
import threading

from src.perception.tracker import ByteTrackTracker

_ID_LOCK = threading.Lock()


class PrivateTracker:
    def __init__(self, fps, config=None):
        from ultralytics.trackers.basetrack import BaseTrack
        self.fps, self.stride = fps, 3
        config = dict(config or {})
        self.buffer_s = config.get("buffer_s", 2.0)
        self.frame_rate = fps / self.stride
        with _ID_LOCK:
            saved = BaseTrack._count
            try:
                self.tracker = ByteTrackTracker(fps, self.stride, **config)
            finally:
                BaseTrack._count = saved
        self.last_frame, self.counter = None, 0

    def _set_stride(self, stride):
        if stride == self.stride:
            return
        engine = self.tracker._tracker
        ratio = stride / self.stride
        # Preserve elapsed seconds and velocity when observation cadence changes.
        # ByteTrack timestamps may be fractional here; update() stamps new
        # observations with its integer clock again. Never fabricate detections.
        for track in engine.tracked_stracks + engine.lost_stracks:
            track.frame_id = engine.frame_id - (engine.frame_id - track.frame_id) / ratio
            track.start_frame = engine.frame_id - (engine.frame_id - track.start_frame) / ratio
            if track.mean is not None:
                track.mean[4:] *= ratio
                track.covariance[4:, :] *= ratio
                track.covariance[:, 4:] *= ratio
        self.stride, self.frame_rate = stride, self.fps / stride
        self.tracker.args.track_buffer = max(1, round(self.buffer_s * self.frame_rate))
        engine.max_frames_lost = self.tracker.args.track_buffer

    def update(self, dets, idx):
        from ultralytics.trackers.basetrack import BaseTrack
        if self.last_frame is not None and idx <= self.last_frame:
            raise ValueError("Tracker requires strictly increasing inference frames")
        with _ID_LOCK:
            saved = BaseTrack._count
            BaseTrack._count = self.counter
            try:
                if self.last_frame is not None:
                    self._set_stride(idx - self.last_frame)
                out = self.tracker.update(dets, idx)
                self.counter, self.last_frame = BaseTrack._count, idx
                return out
            finally:
                BaseTrack._count = saved
