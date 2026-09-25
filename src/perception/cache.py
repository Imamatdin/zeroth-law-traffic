"""Reusable perception caches. A cache is published only after complete decoding."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Callable

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from src.contracts import Detector, Tracker

FRAME_SCHEMA = pa.schema([("frame", pa.int64()), ("t", pa.float64())])
DET_SCHEMA = pa.schema(list(FRAME_SCHEMA) + [
    ("x1", pa.float32()), ("y1", pa.float32()), ("x2", pa.float32()), ("y2", pa.float32()),
    ("cls", pa.int64()), ("score", pa.float32()),
])
TRACK_SCHEMA = pa.schema(list(DET_SCHEMA) + [
    ("track_id", pa.int64()), ("fx", pa.float32()), ("fy", pa.float32()),
])
SCHEMAS = {"frames": FRAME_SCHEMA, "detections": DET_SCHEMA, "tracks": TRACK_SCHEMA}


def source_identity(video: Path) -> dict:
    s = video.stat()
    return {"path": str(video.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def read_meta(directory: Path, expected: dict | None = None) -> dict:
    """Reject incomplete, stale, or truncated caches instead of silently reusing them."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("complete") is not True or meta.get("schema_version") != 1:
        raise ValueError("Cache is incomplete or has an unsupported version")
    if expected is not None and meta["request"] != expected:
        raise ValueError("Stale cache: source, pipeline, code or stride changed; choose a new cache directory")
    for name, schema in SCHEMAS.items():
        file = pq.ParquetFile(directory / f"{name}.parquet")
        if file.schema_arrow != schema or file.metadata.num_rows != meta["row_counts"][name]:
            raise ValueError(f"Cache table schema/count mismatch: {name}")
    return meta


class CacheWriter:
    """Write bounded row groups; metadata is the final completion marker."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.writers = {name: pq.ParquetWriter(directory / f"{name}.parquet", schema)
                        for name, schema in SCHEMAS.items()}
        self.buffers = {name: [] for name in SCHEMAS}
        self.counts = {name: 0 for name in SCHEMAS}
        self.closed = False

    def append(self, frame_idx, t, detections, tracks):
        self.buffers["frames"].append({"frame": frame_idx, "t": t})
        for box, cls, score in zip(detections.xyxy, detections.cls, detections.score):
            self.buffers["detections"].append(dict(frame=frame_idx, t=t, x1=float(box[0]),
                y1=float(box[1]), x2=float(box[2]), y2=float(box[3]), cls=int(cls), score=float(score)))
        ids = set()
        for track in tracks:
            if track.frame_idx != frame_idx or abs(track.t - t) > 1e-6:
                raise ValueError("Tracker frame/timestamp disagrees with frame_idx / fps")
            if track.track_id in ids:
                raise ValueError("Duplicate track ID in a frame")
            ids.add(track.track_id)
            x1, y1, x2, y2 = track.xyxy
            fx, fy = track.foot_point
            self.buffers["tracks"].append(dict(frame=frame_idx, t=t, x1=x1, y1=y1, x2=x2,
                y2=y2, cls=track.cls, score=track.score, track_id=track.track_id, fx=fx, fy=fy))
        if len(self.buffers["frames"]) >= 128:
            self.flush()

    def flush(self):
        for name, rows in self.buffers.items():
            if rows:
                self.writers[name].write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]))
                self.counts[name] += len(rows)
                rows.clear()

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                for writer in self.writers.values():
                    writer.close()
                self.closed = True

    def finish(self, metadata):
        self.close()
        meta = {**metadata, "schema_version": 1, "complete": True, "row_counts": self.counts}
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
        return meta


def build_cache(video: Path, directory: Path, pipeline: dict,
                factory: Callable[[dict], tuple[Detector, Tracker]], stride: int = 1) -> dict:
    """Factory gets video metadata and is called only when no reusable cache exists."""
    if stride < 1:
        raise ValueError("Stride must be positive")
    request = {"source": source_identity(video), "pipeline": pipeline, "stride": stride}
    if directory.exists():
        return read_meta(directory, request)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A second process must not build the same destination concurrently.
    lock_path = directory.parent / f".{directory.name}.lock"
    with lock_path.open("x"):
        pass
    pending = None
    writer = None
    cap = None
    try:
        if directory.exists():
            return read_meta(directory, request)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or n_frames <= 0:
            raise ValueError("Invalid video frame count or fps")
        meta = {"video_id": video.name, "fps": fps, "n_frames": n_frames,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duration": n_frames / fps,
                "coords": "pixels", "request": request}
        detector, tracker = factory(meta)
        tracker.reset()
        pending = Path(tempfile.mkdtemp(prefix=f".{directory.name}-partial-", dir=directory.parent))
        writer = CacheWriter(pending)
        decoded = 0
        while cap.grab():
            i = decoded
            decoded += 1
            if i % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"Could not retrieve frame {i}")
            batch = detector.predict([frame])
            if len(batch) != 1:
                raise ValueError("Detector must return one Detections object per input frame")
            tracks = tracker.update(batch[0], i)
            writer.append(i, i / fps, batch[0], tracks)
        if decoded != n_frames:
            raise ValueError(f"Decoded {decoded}/{n_frames} frames; cache remains incomplete")
        if source_identity(video) != request["source"]:
            raise ValueError("Source changed during extraction")
        result = writer.finish(meta)
        # Both paths must remain siblings under the explicitly chosen cache parent.
        if pending.resolve().parent != directory.resolve().parent or directory.exists():
            raise ValueError("Unsafe or conflicting cache destination")
        pending.rename(directory)
        return result
    finally:
        try:
            if writer:
                writer.close()
        finally:
            try:
                if cap:
                    cap.release()
            finally:
                lock_path.unlink(missing_ok=True)
        # Failed partial directories are retained for diagnosis; never marked reusable.
