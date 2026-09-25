"""Traffic-signal state from a fixed ROI crop (HSV baseline; Javohir's V4 module may replace it).

Red+amber (shown before green here) counts as red: moving off is still prohibited.
Frames with no clearly lit lamp return "unknown", never a guess.
"""

from __future__ import annotations

import cv2
import numpy as np

BANDS = {"red": ((0, 10), (170, 180)), "yellow": ((11, 35),), "green": ((40, 100),)}


def lamp_fractions(crop_bgr: np.ndarray, min_v: int = 100, min_s: int = 80) -> dict[str, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v >= min_v) & (s >= min_s)
    out = {}
    for name, ranges in BANDS.items():
        band = np.zeros_like(lit)
        for lo, hi in ranges:
            band |= (h >= lo) & (h <= hi)
        out[name] = float((lit & band).mean())
    return out


def classify_signal(crop_bgr: np.ndarray, min_frac: float = 0.004, min_v: int = 100,
                    min_s: int = 80) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    f = lamp_fractions(crop_bgr, min_v, min_s)
    lit = {k for k, v in f.items() if v >= min_frac}
    if lit == {"green"}:
        return "green"
    if "red" in lit and "green" not in lit:
        return "red"
    if lit == {"yellow"}:
        return "yellow"
    return "unknown"


def smooth_states(states: list[str], min_run: int = 3, bridge_unknown: int = 6) -> list[str]:
    """Bridge short unknown gaps between identical states (flashing green blinks the lamp off),
    then replace runs shorter than min_run samples with the preceding state."""
    out = list(states)
    i = 0
    while i < len(out):
        if out[i] == "unknown" and i > 0:
            j = i
            while j < len(out) and out[j] == "unknown":
                j += 1
            if j < len(out) and j - i <= bridge_unknown and out[j] == out[i - 1]:
                out[i:j] = [out[i - 1]] * (j - i)
            i = j
        else:
            i += 1
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run and i > 0:
            out[i:j] = [out[i - 1]] * (j - i)
        i = j
    return out


def lamp_lit(cell_bgr: np.ndarray, min_v: int = 130, min_s: int = 70) -> float:
    """Fraction of a lamp cell that is bright and saturated (any hue)."""
    if cell_bgr is None or cell_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] >= min_v) & (hsv[..., 1] >= min_s)).mean())


def classify_head(frame_bgr: np.ndarray, lamps_px: dict[str, tuple], min_lit: float = 0.12) -> str:
    """State from per-lamp cells: position, not hue, decides which lamp is on."""
    lit = set()
    for name, (x1, y1, x2, y2) in lamps_px.items():
        if lamp_lit(frame_bgr[y1:y2, x1:x2]) >= min_lit:
            lit.add(name)
    if lit == {"green"}:
        return "green"
    if "red" in lit and "green" not in lit:
        return "red"
    if lit == {"yellow"}:
        return "yellow"
    return "unknown"
