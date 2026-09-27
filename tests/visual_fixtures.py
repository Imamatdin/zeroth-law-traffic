"""Small frame sequences over the 1000x1000 fixture scene for the visual engines (200 px wide samples)."""

from __future__ import annotations

import cv2
import numpy as np

from src.events.visual import VisualSamples

SIZE = 200
SCALE = SIZE / 1000.0


def background(seed: int = 0) -> np.ndarray:
    """Asphalt-like texture: grey with fine grain."""
    rng = np.random.default_rng(seed)
    g = np.clip(100 + rng.normal(0, 12, (SIZE, SIZE)), 0, 255).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    return np.repeat(g[..., None], 3, axis=2)


def samples(n: int = 60, paint=None, gain_of_t=None) -> VisualSamples:
    """n one-second samples; paint(frame, t) edits a float BGR frame in place; gain_of_t scales brightness."""
    base = background()
    frames = []
    for k in range(n):
        f = base.copy()
        if paint is not None:
            paint(f, float(k))
        if gain_of_t is not None:
            f *= gain_of_t(float(k))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return VisualSamples(np.arange(n, dtype=float), np.stack(frames), SCALE)


def checker(frame: np.ndarray, x: int, y: int, size: int = 10, colour=(40, 40, 40)) -> None:
    """A textured object: a checkerboard patch with its top-left corner at (x, y) in sample pixels."""
    yy, xx = np.mgrid[0:size, 0:size]
    board = ((yy // 2 + xx // 2) % 2).astype(bool)
    patch = frame[y:y + size, x:x + size]
    patch[board] = colour
    patch[~board] = (220, 220, 220)


def glow(frame: np.ndarray, x: int, y: int, sigma: float = 6.0, amp: float = 60.0) -> None:
    """A smooth light pool centred at (x, y)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    frame += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))[..., None]
