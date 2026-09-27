"""Same perception, stitching, world, events and causal risk components as solution.py.

No import of submission model registry: this CPU profile owns one model loaded
once and reused sequentially. Part B decodes again and owns independent state.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.anticipation.risk import RiskModel
from src.atlas.flow import Atlas
from src.events.base import SignalTimeline, VideoContext
from src.events.registry import load_config, run_engines
from src.features.stitching import prepare_tracks
from src.features.tracks import compute_world
from src.perception.detector import UltralyticsDetector
from src.runtime.decode import restore
from src.runtime.tracker import PrivateTracker
from src.scene.geometry import Scene
from src.scene.signal import classify_head, classify_signal
from demo.scene_check import check_scene
from demo.video import inspect_video, sampled_frames
from demo.schema import replay_document, event_document, signal_document, risk_document

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['frame', 't', 'track_id', 'cls', 'score', 'x1', 'y1', 'x2', 'y2', 'fx', 'fy']


@dataclass(frozen=True)
class Profile:
    name: str = 'cpu-yolo11m-640-s9'
    detector: str = 'yolo11m.pt'
    imgsz: int = 640
    stride: int = 9
    decode_width: int = 960
    threads: int = 2
    device: str = 'cpu'
    half: bool = False


class Engine:
    def __init__(self, profile=None, weights=None):
        import torch
        self.profile = profile or Profile()
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_OFFLINE'] = 'true'
        torch.set_num_threads(self.profile.threads)
        torch.manual_seed(0)
        np.random.seed(0)
        config = json.loads((ROOT / 'configs/perception.json').read_text())
        self.tracker_config = config['tracker']
        self.detector = UltralyticsDetector(
            str(weights or ROOT / 'weights' / self.profile.detector), imgsz=self.profile.imgsz,
            conf=config['detector']['conf'], iou=config['detector']['iou'],
            device='cpu', half=False, prescale_width=self.profile.decode_width)
        self.detector.predict([np.zeros((540, 960, 3), np.uint8)])
        self.config = load_config(ROOT / 'configs/events.yaml')
        self.atlas = json.loads((ROOT / 'configs/atlas.json').read_text())

    @staticmethod
    def signals(frame, scene, t, idx):
        out = []
        for sid, sig in scene.signals.items():
            x1, y1, x2, y2 = sig['roi_px']
            state = classify_head(frame, sig['lamps_px']) if sig['lamps_px'] else classify_signal(frame[y1:y2, x1:x2])
            out.append(dict(frame=idx, t=t, signal=sid, state=state))
        return out

    def process(self, path, progress=lambda stage, fraction: None):
        started = time.perf_counter()
        progress('decoding', 0)
        meta, first = inspect_video(path)
        gate = check_scene(first, meta['width'], meta['height'])
        del first
        matched = gate['matches']
        scene = Scene.load(ROOT / 'configs/camera.yaml', meta['width'], meta['height']) if matched else Scene(meta['width'], meta['height'])
        if not matched:
            scene.road = np.array([[0, 0], [meta['width'], 0], [meta['width'], meta['height']], [0, meta['height']]], float)
        profile = self.profile
        rows, signals, times, detections = [], [], [], []
        tracker = PrivateTracker(meta['fps'], self.tracker_config)
        small_scene = None
        stream = sampled_frames(path, meta, profile.stride, profile.threads, profile.decode_width)
        try:
            for idx, frame in stream:
                t = idx / meta['fps']
                det = restore(self.detector.predict([frame])[0], frame.shape, meta)
                if matched:
                    if small_scene is None:
                        small_scene = Scene.load(ROOT / 'configs/camera.yaml', frame.shape[1], frame.shape[0])
                    signals.extend(self.signals(frame, small_scene, t, idx))
                times.append((idx, t))
                for box, cls, score in zip(det.xyxy, det.cls, det.score):
                    detections.append([idx, round(t, 3), int(cls), round(float(score), 4), *[round(float(v), 2) for v in box]])
                for tr in tracker.update(det, idx):
                    rows.append((idx, tr.t, tr.track_id, tr.cls, tr.score, *tr.xyxy, *tr.foot_point))
                progress('detecting', (idx + 1) / meta['n_frames'])
        finally:
            stream.close()
        tracks = pd.DataFrame(rows, columns=COLUMNS)
        if len(tracks):
            tracks, _ = prepare_tracks(tracks, width=meta['width'], height=meta['height'])
            world = compute_world(tracks, meta['width'], meta['height'], scene=scene)
            world = world.merge(tracks[['frame', 'track_id', 'x1', 'y1', 'x2', 'y2']], on=['frame', 'track_id'], validate='one_to_one')
        else:
            tracks['raw_track_id'] = pd.Series(dtype=int)
            world = pd.DataFrame()
        progress('events', 0)
        raw, segments = [], []
        if matched and len(world):
            series = pd.DataFrame(signals)
            timelines = {sid: SignalTimeline.from_series(series, sid) for sid in series.signal.unique()} if len(series) else {}
            ctx = VideoContext(meta['video_id'], meta['fps'], meta['width'], meta['height'],
                               meta['duration'], profile.stride, world, scene, timelines, Atlas(self.atlas))
            raw, segments = run_engines(ctx, self.config, only_enabled=True)
        progress('events', 1)
        replay, mapping = replay_document(tracks, world, times, meta)
        events = event_document(raw, segments, self.config, matched)
        # Separate video pass, tracker and causal state. No Part A rows enter risk.
        tracker = PrivateTracker(meta['fps'], self.tracker_config)
        risk = RiskModel(scene, self.atlas if matched else None)
        risk.reset(meta)
        curve, evidence, poses = [], [], []
        stream = sampled_frames(path, meta, profile.stride, profile.threads, profile.decode_width)
        progress('risk', 0)
        try:
            for idx, frame in stream:
                t = idx / meta['fps']
                det = restore(self.detector.predict([frame])[0], frame.shape, meta)
                observed = tracker.update(det, idx)
                sig = {r['signal']: r['state'] for r in self.signals(frame, small_scene, t, idx)} if matched else {}
                score = risk.update(observed, t, sig)
                curve.append([t, score])
                ev = dict(risk.last_evidence)
                evidence.append(ev)
                point = None
                if 'pair' in ev:
                    points = []
                    for tid in ev['pair']:
                        st = risk.tracks[tid]
                        p = np.array(st.samples[-1][1:3])
                        v = np.array(st.vels[-1][1:3])
                        points.append(p + v * min(ev['tca_s'], 5))
                    point = np.mean(points, axis=0)
                poses.append(point)
                progress('risk', (idx + 1) / meta['n_frames'])
        finally:
            stream.close()
        elapsed = time.perf_counter() - started
        return dict(metadata=dict(profile=asdict(profile), scene_check=gate,
                    scene_events_skipped=not matched, processing_seconds=round(elapsed, 3),
                    wall_time_ratio=round(elapsed / meta['duration'], 3),
                    warnings=[] if matched else ['Camera mismatch: scene-dependent events, signals and atlas disabled; generic risk is uncalibrated.']),
                    replay=replay, events=events, signal=signal_document(signals, meta['duration']),
                    risk=risk_document(curve, evidence, poses, mapping, meta, matched),
                    detections=dict(columns=['frame', 't', 'cls', 'score', 'x1', 'y1', 'x2', 'y2'],
                                    coords='native pixels', rows=detections))
