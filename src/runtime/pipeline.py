"""Submission wiring. No caches or Part A output enter the causal estimator."""
import importlib
import importlib.util
import json
import threading
import time
import pandas as pd
from src.anticipation.risk import RiskModel
from src.atlas.flow import Atlas
from src.contracts import Detections
from src.events.base import SignalTimeline, VideoContext
from src.events.registry import load_config, run_engines
from src.features.tracks import compute_world
from src.models_registry import CONFIG, ROOT, new_detector
from src.perception.tracker import ByteTrackTracker
from src.postprocess.segments import to_official
from src.runtime.budget import BudgetGuard, log
from src.runtime.decode import frames, metadata, restore
from src.scene.geometry import Scene
from src.scene.signal import classify_head, classify_signal, lamp_lit

TRACK_COLUMNS = ["frame", "t", "track_id", "cls", "score", "x1", "y1", "x2", "y2", "fx", "fy"]
# Static priors, read once at import (outside every video's budget). Neither depends on any video.
CAMERA = ROOT / "configs/camera.yaml"
ATLAS_DATA = json.loads((ROOT / "configs/atlas.json").read_text(encoding="utf-8"))
_ID_LOCK = threading.Lock()


class PrivateTracker:
    """Keep ByteTrack's base-3 clock and isolate its library-global ID allocator."""
    def __init__(self, fps):
        from ultralytics.trackers.basetrack import BaseTrack
        with _ID_LOCK:
            saved = BaseTrack._count
            try:
                self.tracker = ByteTrackTracker(fps, 3, **CONFIG.get("tracker", {}))
            finally:
                BaseTrack._count = saved
        self.last_frame, self.counter = -3, 0

    def update(self, dets, idx):
        from ultralytics.trackers.basetrack import BaseTrack
        with _ID_LOCK:
            saved = BaseTrack._count
            BaseTrack._count = self.counter
            try:
                for missing in range(self.last_frame + 3, idx, 3):
                    self.tracker.update(Detections.empty(), missing)
                out = self.tracker.update(dets, idx)
                self.counter, self.last_frame = BaseTrack._count, idx
                return out
            finally:
                BaseTrack._count = saved


def signal_rows(frame, idx, fps, scene):
    rows = []
    for sid, sig in scene.signals.items():
        lamps = sig["lamps_px"]
        x1, y1, x2, y2 = sig["roi_px"]
        state = classify_head(frame, lamps) if lamps else classify_signal(frame[y1:y2, x1:x2])
        row = dict(frame=idx, t=idx / fps, signal=sid, state=state)
        for name, (a, b, c, d) in lamps.items():
            row["lit_" + name] = lamp_lit(frame[b:d, a:c])
        rows.append(row)
    return rows


def stitch_if_present(tracks, width, height):
    name = "src.features.stitching"
    if importlib.util.find_spec(name) is None:
        log("stitching", available=False)
        return tracks
    stitched, info = importlib.import_module(name).prepare_tracks(tracks, width=width, height=height)
    log("stitching", available=True, info=info)
    return stitched


def detect_events(video_path):
    meta = metadata(video_path)
    guard = BudgetGuard("A", meta)
    detector, tracker = new_detector(), PrivateTracker(meta["fps"])
    scene = Scene.load(ROOT / "configs/camera.yaml", meta["width"], meta["height"])
    small_scene = None
    rows, signals = [], []
    stream = frames(video_path, meta)
    try:
        for idx, frame in stream:
            if small_scene is None:
                small_scene = Scene.load(ROOT / "configs/camera.yaml", frame.shape[1], frame.shape[0])
            signals.extend(signal_rows(frame, idx, meta["fps"], small_scene))
            if not guard.due(idx):
                continue
            started = time.perf_counter()
            dets = restore(detector.predict([frame])[0], frame.shape, meta)
            for tr in tracker.update(dets, idx):
                rows.append((idx, tr.t, tr.track_id, tr.cls, tr.score, *tr.xyxy, *tr.foot_point))
            guard.observed(idx, time.perf_counter() - started)
    finally:
        stream.close()
    tracks = stitch_if_present(pd.DataFrame(rows, columns=TRACK_COLUMNS), meta["width"], meta["height"])
    log("perception_summary", part="A", track_rows=len(tracks),
        tracks=int(tracks.track_id.nunique()), signal_rows=len(signals))
    if tracks.empty:
        world = pd.DataFrame(columns=[*TRACK_COLUMNS, "cls_major", "gx", "gy", "vx", "vy",
                                      "ax", "ay", "speed", "speed_rel", "heading", "box_h",
                                      "stationary_s", "age_s", "on_carriageway", "in_crosswalk",
                                      "in_island", "in_intersection"])
    else:
        world = compute_world(tracks, meta["width"], meta["height"], scene=scene)
        world = world.merge(tracks[["frame", "track_id", "x1", "y1", "x2", "y2"]],
                            on=["frame", "track_id"], validate="one_to_one")
    series = pd.DataFrame(signals)
    timelines = ({sid: SignalTimeline.from_series(series, sid) for sid in series.signal.unique()}
                 if len(series) else {})
    ctx = VideoContext(meta["video_id"], meta["fps"], meta["width"], meta["height"],
                       meta["n_frames"] / meta["fps"], 3, world, scene, timelines, Atlas(ATLAS_DATA))
    _, segments = run_engines(ctx, load_config(ROOT / "configs/events.yaml"), only_enabled=True)
    guard.finish()
    return to_official(segments)


class RiskEstimator:
    def reset(self, meta):
        self.meta = dict(meta)
        self.guard = BudgetGuard("B", self.meta)
        self.detector = new_detector("B")
        self.tracker = PrivateTracker(self.meta["fps"])
        # Causal risk: own tracks and this frame's own signal reading only (see
        # private/handoff-2026-09-26-risk.md). Never Part A tracks, stitching or signal timelines.
        self.risk = RiskModel(Scene.load(CAMERA, self.meta["width"], self.meta["height"]), ATLAS_DATA)
        self.risk.reset(self.meta)
        self.lamps = {sid: sig["lamps_px"] for sid, sig in self.risk.scene.signals.items() if sig.get("lamps_px")}
        self.last_score, self.last_t = 0.0, -1.0
        self.tracks = []

    def step(self, frame, t_sec):
        if t_sec < self.last_t:
            raise ValueError("RiskEstimator requires chronological frames")
        self.last_t = t_sec
        idx = round(t_sec * self.meta["fps"])
        if idx % 3 or not self.guard.due(idx):
            if idx == self.meta["n_frames"] - 1:
                self.guard.finish()
            return self.last_score
        started = time.perf_counter()
        self.tracks = self.tracker.update(self.detector.predict([frame])[0], idx)
        signals = {sid: classify_head(frame, lamps) for sid, lamps in self.lamps.items()}
        self.last_score = self.risk.update(self.tracks, t_sec, signals)
        self.guard.observed(idx, time.perf_counter() - started)
        if idx == self.meta["n_frames"] - 1:
            self.guard.finish()
        return self.last_score
