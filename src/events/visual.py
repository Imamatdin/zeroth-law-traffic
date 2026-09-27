"""Downscaled frame samples for the visual engines (road_obstacle, fire_smoke), and raster helpers.

Part A may read the whole video, so the engines see every sample at once. Samples are taken every
`every_s` seconds at `width` pixels wide from whatever frames the decoder yields; track boxes (source
pixels) map onto them with `scale`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VISUAL_LABELS = ("road_obstacle", "fire_smoke")


@dataclass
class VisualSamples:
    t: np.ndarray          # (N,) seconds
    frames: np.ndarray     # (N, h, w, 3) uint8 BGR
    scale: float           # sample pixels per source pixel

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, frames=self.frames, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> "VisualSamples":
        d = np.load(path)
        return cls(d["t"], d["frames"], float(d["scale"]))


class VisualCollector:
    def __init__(self, fps: float, source_width: int, source_height: int, every_s: float = 1.0, width: int = 640):
        self.fps, self.every_s = fps, every_s
        self.size = (width, max(2, round(source_height * width / source_width)))
        self.scale = width / source_width
        self.next_t = 0.0
        self.t: list[float] = []
        self.frames: list[np.ndarray] = []

    def due(self, idx: int) -> bool:
        return idx / self.fps >= self.next_t - 1e-9

    def add(self, idx: int, frame: np.ndarray) -> None:
        t = idx / self.fps
        self.t.append(t)
        self.frames.append(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA))
        self.next_t = t + self.every_s

    def result(self) -> VisualSamples | None:
        if not self.frames:
            return None
        return VisualSamples(np.array(self.t), np.stack(self.frames), self.scale)


def sample_video(path: Path, every_s: float = 1.0, width: int = 640, on_frame=None) -> VisualSamples | None:
    """Decode the whole video once, keeping a sample every `every_s`; on_frame(idx, frame) sees every
    frame (the review tool reads signal lamps in the same pass)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    collector = VisualCollector(fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                every_s, width)
    idx = 0
    try:
        while cap.grab():
            want = collector.due(idx)
            if want or on_frame is not None:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if on_frame is not None:
                    on_frame(idx, frame)
                if want:
                    collector.add(idx, frame)
            idx += 1
    finally:
        cap.release()
    return collector.result()


def needs_visual(config: dict) -> bool:
    return any(config.get(label, {}).get("enabled", False) for label in VISUAL_LABELS)


def polygon_mask(polygon_px: np.ndarray, shape: tuple[int, int], scale: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon_px * scale).astype(np.int32)], 1)
    return mask.astype(bool)


def road_mask(scene, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Carriageway at sample resolution: the road polygon minus sidewalks, islands and median."""
    if scene.road is None:
        return np.zeros(shape, bool)
    mask = polygon_mask(scene.road, shape, scale)
    for kind in ("sidewalks", "islands", "median"):
        for region in scene.regions_of(kind):
            mask &= ~polygon_mask(region.polygon, shape, scale)
    return mask


def box_coverage(samples: pd.DataFrame, times: np.ndarray, shape: tuple[int, int], scale: float, fps: float,
                 grow: float, moving_only: bool = False, moving_rel: float = 0.5) -> np.ndarray:
    """(N, h, w) bool: pixels inside any track box (grown by `grow` of its size) at each sample time,
    taken from the nearest cached frame."""
    frames = np.sort(samples["frame"].unique())
    out = np.zeros((len(times),) + shape, bool)
    if len(frames) == 0:
        return out
    rows = samples if not moving_only else samples[samples["speed_rel"] >= moving_rel]
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in rows.groupby("frame")}
    h, w = shape
    for k, t in enumerate(times):
        f = frames[int(np.argmin(np.abs(frames - t * fps)))]
        for x1, y1, x2, y2 in by_frame.get(f, ()):
            gx, gy = grow * (x2 - x1), grow * (y2 - y1)
            a, b = int(max(0, (x1 - gx) * scale)), int(max(0, (y1 - gy) * scale))
            c, d = int(min(w, np.ceil((x2 + gx) * scale))), int(min(h, np.ceil((y2 + gy) * scale)))
            out[k, b:d, a:c] = True
    return out


def clean_background(stack: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Per-pixel median over the samples where no road user covers the pixel (plain median where one
    always does). A plain median shows the queue wherever traffic stands most of the time."""
    masked = np.where(covered, np.nan, stack.astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN pixels fall back below
        bg = np.nanmedian(masked, axis=0)
    return np.where(np.isfinite(bg), bg, np.median(stack, axis=0)).astype(np.float32)


def box_height_at(samples: pd.DataFrame, classes, y_px: float) -> float:
    """Typical box height of `classes` whose foot point lies near image row y_px (source pixels);
    a linear fit of height on foot row, since apparent size grows towards the camera."""
    rows = samples[samples["cls_major"].isin(classes)]
    if len(rows) < 10:
        return float("nan")
    fy = rows["y2"].to_numpy(float)
    h = (rows["y2"] - rows["y1"]).to_numpy(float)
    if np.ptp(fy) < 1.0:
        return float(np.median(h))
    slope, icpt = np.polyfit(fy, h, 1)
    return float(max(slope * y_px + icpt, np.percentile(h, 5)))


def link_blobs(per_sample: list[list[dict]], min_iou: float, max_gap: int) -> list[list[tuple[int, dict]]]:
    """Chain blobs (dicts with 'box' = x1, y1, x2, y2) across samples by box IoU; returns chains of
    (sample index, blob), allowing up to max_gap samples without a match."""
    chains: list[list[tuple[int, dict]]] = []
    open_chains: list[list[tuple[int, dict]]] = []
    for k, blobs in enumerate(per_sample):
        still_open = []
        used = set()
        for chain in open_chains:
            last_k, last = chain[-1]
            if k - last_k > max_gap + 1:
                chains.append(chain)
                continue
            best, best_iou = None, min_iou
            for j, b in enumerate(blobs):
                if j in used:
                    continue
                iou = _iou(last["box"], b["box"])
                if iou >= best_iou:
                    best, best_iou = j, iou
            if best is not None:
                used.add(best)
                chain.append((k, blobs[best]))
            still_open.append(chain)
        for j, b in enumerate(blobs):
            if j not in used:
                still_open.append([(k, b)])
        open_chains = still_open
    return chains + open_chains


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def components(mask: np.ndarray, min_area: int) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append({"box": (float(x), float(y), float(x + w), float(y + h)), "area": int(area),
                        "mask": labels == i})
    return out


__all__ = ["VisualSamples", "VisualCollector", "sample_video", "needs_visual", "road_mask", "polygon_mask", "box_coverage",
           "box_height_at", "clean_background", "link_blobs", "components", "VISUAL_LABELS"]
