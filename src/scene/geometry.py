"""Scene layout from configs/camera.yaml (normalized coordinates) and geometric queries in pixels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REGION_KEYS = ("crosswalks", "islands", "sidewalks", "median")


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd rule; points (N,2), polygon (M,2). Boundary points may go either way."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    polygon = np.asarray(polygon, dtype=float)
    x, y = points[:, 0:1], points[:, 1:2]
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.count_nonzero(straddles & (x < x_cross), axis=1) % 2 == 1


def side_of_line(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign of the cross product: +1 / -1 on either side of the directed line a->b, 0 on it."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    d = np.asarray(b, float) - np.asarray(a, float)
    rel = points - np.asarray(a, float)
    return np.sign(d[0] * rel[:, 1] - d[1] * rel[:, 0])


def crosses_segment(p0, p1, a, b) -> bool:
    """True if the motion p0->p1 crosses the finite segment a-b."""
    s_motion = side_of_line(np.array([a, b]), p0, p1)
    s_line = side_of_line(np.array([p0, p1]), a, b)
    return bool(s_motion[0] * s_motion[1] < 0 and s_line[0] * s_line[1] < 0)


@dataclass
class Region:
    id: str
    kind: str
    polygon: np.ndarray  # pixels


@dataclass
class Scene:
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    road: np.ndarray | None = None
    intersection: np.ndarray | None = None
    stop_lines: dict[str, np.ndarray] = field(default_factory=dict)
    signals: dict[str, dict] = field(default_factory=dict)
    verified: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, width: int, height: int) -> "Scene":
        """Scale the normalized map to the actual video size, so resolution changes do not break it."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw.get("coords") != "normalized":
            raise ValueError("camera.yaml must use normalized coordinates")
        scale = np.array([width, height], dtype=float)

        def px(points) -> np.ndarray:
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or np.any((arr < 0) | (arr > 1)):
                raise ValueError(f"Invalid normalized points: {points}")
            return arr * scale

        scene = cls(width, height, verified=raw.get("verified", {}))
        for key in REGION_KEYS:
            for item in raw.get(key, []) or []:
                scene.regions.append(Region(item["id"], key, px(item["polygon"])))
        if raw.get("road"):
            scene.road = px(raw["road"])
        if raw.get("intersection"):
            scene.intersection = px(raw["intersection"])
        for line in raw.get("stop_lines", []) or []:
            scene.stop_lines[line["id"]] = px(line["points"])
        for sig in raw.get("signals", []) or []:
            x1, y1, x2, y2 = sig["roi"]
            roi = px([[x1, y1], [x2, y2]]).round().astype(int).ravel()
            lamps = {name: tuple(int(v) for v in px([[b[0], b[1]], [b[2], b[3]]]).round().astype(int).ravel())
                     for name, b in (sig.get("lamps") or {}).items()}
            scene.signals[sig["id"]] = {**sig, "roi_px": tuple(int(v) for v in roi), "lamps_px": lamps}
        return scene

    def regions_of(self, kind: str) -> list[Region]:
        return [r for r in self.regions if r.kind == kind]

    def in_kind(self, points: np.ndarray, kind: str) -> np.ndarray:
        points = np.atleast_2d(points)
        hit = np.zeros(len(points), dtype=bool)
        for region in self.regions_of(kind):
            hit |= point_in_polygon(points, region.polygon)
        return hit

    def on_carriageway(self, points: np.ndarray) -> np.ndarray:
        """Inside the road polygon and not on a sidewalk, island or median."""
        points = np.atleast_2d(points)
        if self.road is None:
            raise ValueError("camera.yaml has no road polygon")
        inside = point_in_polygon(points, self.road)
        for kind in ("sidewalks", "islands", "median"):
            inside &= ~self.in_kind(points, kind)
        return inside
