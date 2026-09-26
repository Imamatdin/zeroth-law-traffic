"""Offline tracklet stitching for Part A: join fragments of one object split by occlusion or an
ID switch. Uses future samples, so Part B must never call it.

A link A -> B is allowed when B starts after A ends (within max_gap_s), the classes are compatible,
B's first foot point lies near A's constant-velocity prediction (distance in A/B box heights, with
tolerance growing with the gap), and box sizes agree. Links are chosen greedily by cost with at most
one successor and one predecessor per tracklet, so chains never overlap in time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VEHICLE_GROUP = frozenset((3, 4, 5))   # car, bus, truck are often confused with each other
RIDER_GROUP = frozenset((0, 1, 2))     # a rider may be detected as person or two-wheeler


@dataclass(frozen=True)
class StitchConfig:
    max_gap_s: float = 3.0
    base_dist_h: float = 0.6      # allowed distance at zero gap, in box heights
    dist_per_s_h: float = 0.4     # extra allowance per second of gap (0.8 admitted wrong vehicle links)
    cross_class_scale: float = 0.75  # person <-> two-wheeler links must be this much tighter
    max_size_ratio: float = 1.6
    velocity_window_s: float = 1.0
    max_pred_s: float = 1.5       # extrapolate at most this long; longer gaps assume the object waited
    gap_cost_per_s: float = 0.2
    moving_h_per_s: float = 0.3   # both fragments faster than this must agree in heading and speed
    min_heading_cos: float = 0.7
    max_speed_ratio: float = 2.0
    border_px: float = 8.0        # a box this close to the frame edge is leaving/entering, not occluded


@dataclass
class _Tracklet:
    id: int
    cls: int
    t0: float
    t1: float
    p0: np.ndarray
    p1: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    h0: float
    h1: float
    starts_at_border: bool = False
    ends_at_border: bool = False


def _velocity(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    if len(t) < 2 or t[-1] - t[0] <= 0:
        return np.zeros(2)
    tc = t - t.mean()
    return (tc[:, None] * (p - p.mean(axis=0))).sum(axis=0) / (tc * tc).sum()


def summarize(tracks: pd.DataFrame, cfg: StitchConfig, width: int | None = None,
              height: int | None = None) -> list[_Tracklet]:
    def at_border(r) -> bool:
        if width is None or height is None:
            return False
        b = cfg.border_px
        return bool(r.x1 <= b or r.y1 <= b or r.x2 >= width - b or r.y2 >= height - b)

    out = []
    for tid, g in tracks.sort_values(["track_id", "t"]).groupby("track_id", sort=True):
        t = g["t"].to_numpy(float)
        p = g[["fx", "fy"]].to_numpy(float)
        h = (g["y2"] - g["y1"]).to_numpy(float)
        tail = t >= t[-1] - cfg.velocity_window_s
        head = t <= t[0] + cfg.velocity_window_s
        k = min(3, len(g))
        out.append(_Tracklet(int(tid), int(g["cls"].mode().iloc[0]), float(t[0]), float(t[-1]),
                             p[:k].mean(axis=0), p[-k:].mean(axis=0), _velocity(t[head], p[head]),
                             _velocity(t[tail], p[tail]),
                             float(np.median(h[:k])), float(np.median(h[-k:])),
                             at_border(g.iloc[0]), at_border(g.iloc[-1])))
    return out


def compatible(a: int, b: int) -> tuple[bool, bool]:
    """(allowed, cross_class)."""
    if a == b:
        return True, False
    if a in VEHICLE_GROUP and b in VEHICLE_GROUP:
        return True, False
    if a in RIDER_GROUP and b in RIDER_GROUP:
        return True, True
    return False, False


def candidate_links(tracklets: list[_Tracklet], cfg: StitchConfig) -> list[dict]:
    starts = sorted(tracklets, key=lambda k: k.t0)
    start_t = np.array([k.t0 for k in starts])
    links = []
    for a in tracklets:
        if a.ends_at_border:
            continue
        lo = np.searchsorted(start_t, a.t1, side="right")
        hi = np.searchsorted(start_t, a.t1 + cfg.max_gap_s, side="right")
        for b in starts[lo:hi]:
            if b.starts_at_border:
                continue
            ok, cross = compatible(a.cls, b.cls)
            if not ok:
                continue
            ratio = max(a.h1, b.h0) / max(min(a.h1, b.h0), 1.0)
            if ratio > cfg.max_size_ratio:
                continue
            gap = b.t0 - a.t1
            pred = a.p1 + a.v1 * min(gap, cfg.max_pred_s)
            scale = (a.h1 + b.h0) / 2
            dist = float(np.linalg.norm(b.p0 - pred) / max(scale, 1.0))
            allowed = (cfg.base_dist_h + cfg.dist_per_s_h * gap) * (cfg.cross_class_scale if cross else 1.0)
            if dist > allowed:
                continue
            sa, sb = np.linalg.norm(a.v1) / max(a.h1, 1.0), np.linalg.norm(b.v0) / max(b.h0, 1.0)
            if min(sa, sb) > cfg.moving_h_per_s:
                cos = float(a.v1 @ b.v0 / (np.linalg.norm(a.v1) * np.linalg.norm(b.v0)))
                if cos < cfg.min_heading_cos or max(sa, sb) / min(sa, sb) > cfg.max_speed_ratio:
                    continue
            links.append({"from": a.id, "to": b.id, "gap_s": round(gap, 3), "dist_h": round(dist, 3),
                          "allowed_h": round(allowed, 3), "cross_class": cross,
                          "cost": dist / allowed + cfg.gap_cost_per_s * gap})
    return links


def choose_links(links: list[dict]) -> list[dict]:
    used_from, used_to, chosen = set(), set(), []
    for link in sorted(links, key=lambda l: (l["cost"], l["from"], l["to"])):
        if link["from"] in used_from or link["to"] in used_to:
            continue
        used_from.add(link["from"])
        used_to.add(link["to"])
        chosen.append(link)
    return chosen


def _foot_on_box(people: pd.DataFrame, bikes: pd.DataFrame, grow: float) -> np.ndarray:
    """(n_people, n_bikes) mask: person foot point inside the bike box enlarged by `grow` of its size."""
    bw = (bikes["x2"] - bikes["x1"]).to_numpy()
    bh = (bikes["y2"] - bikes["y1"]).to_numpy()
    fx, fy = people["fx"].to_numpy()[:, None], people["fy"].to_numpy()[:, None]
    return ((fx >= bikes["x1"].to_numpy() - grow * bw) & (fx <= bikes["x2"].to_numpy() + grow * bw)
            & (fy >= bikes["y1"].to_numpy() - grow * bh) & (fy <= bikes["y2"].to_numpy() + grow * bh))


def fuse_riders(tracks: pd.DataFrame, min_shared: int = 5, min_fraction: float = 0.6,
                min_life_fraction: float = 0.5, grow: float = 0.25) -> tuple[pd.DataFrame, list[dict]]:
    """Merge person tracks that ride a two-wheeler into that two-wheeler's track.

    The detector often reports a rider and the bike as two objects; they are one road user. A person
    track is a rider of bike B if its foot lies on B's (enlarged) box in >= min_fraction of the frames
    they share (and at least min_shared frames), and those frames are >= min_life_fraction of the
    person's own detections: someone who waits beside a bike, then walks off, stays a pedestrian.
    Classes are per-track majorities, so a person briefly misread as a bicycle is not a bike.
    Its rows in shared frames are dropped (the bike box represents the pair); its other rows keep
    the object alive when only the rider is detected.
    """
    major = tracks.groupby("track_id")["cls"].agg(lambda c: int(c.mode().iloc[0]))
    life = tracks.groupby("track_id").size()
    people = tracks[tracks["track_id"].map(major) == 0]
    bikes = tracks[tracks["track_id"].map(major).isin((1, 2))]
    if people.empty or bikes.empty:
        return tracks, []
    hits: dict[tuple[int, int], int] = {}
    shared: dict[tuple[int, int], int] = {}
    for frame, pf in people.groupby("frame", sort=False):
        bf = bikes[bikes["frame"] == frame]
        if bf.empty:
            continue
        on = _foot_on_box(pf, bf, grow)
        for i, pid in enumerate(pf["track_id"].to_numpy()):
            for j, bid in enumerate(bf["track_id"].to_numpy()):
                key = (int(pid), int(bid))
                shared[key] = shared.get(key, 0) + 1
                hits[key] = hits.get(key, 0) + int(on[i, j])
    best: dict[int, tuple[float, int, int]] = {}
    for (pid, bid), n in shared.items():
        h = hits[(pid, bid)]
        if h >= min_shared and h / n >= min_fraction and h >= min_life_fraction * life[pid]:
            cand = (h / n, h, -bid)
            if pid not in best or cand > best[pid][0:3]:
                best[pid] = (*cand, bid)
    if not best:
        return tracks, []
    out = tracks.copy()
    bike_cls = major
    bike_frames = {bid: set(g["frame"]) for bid, g in bikes.groupby("track_id")}
    fused = []
    drop = np.zeros(len(out), bool)
    for pid, (frac, h, _, bid) in sorted(best.items()):
        rows = (out["track_id"] == pid).to_numpy()
        in_bike = out["frame"].isin(bike_frames[bid]).to_numpy()
        drop |= rows & in_bike
        keep = rows & ~in_bike
        out.loc[keep, "track_id"] = bid
        out.loc[keep, "cls"] = bike_cls[bid]
        fused.append({"rider": pid, "bike": bid, "shared_frames": shared[(pid, bid)], "on_bike_frames": h,
                      "fraction": round(frac, 3)})
    out = out[~drop]
    # Two riders on one bike can leave two rows for the same frame: keep the more confident one.
    out = (out.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
              .drop_duplicates(["track_id", "frame"]).sort_values(["frame", "track_id"], ignore_index=True))
    return out, fused


def stitch(tracks: pd.DataFrame, cfg: StitchConfig | None = None, width: int | None = None,
           height: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Returns tracks with `track_id` replaced by the chain's first raw id (`raw_track_id` kept),
    and the chosen links as evidence. Pass the frame size so border contacts are judged exactly."""
    cfg = cfg or StitchConfig()
    chosen = choose_links(candidate_links(summarize(tracks, cfg, width, height), cfg))
    successor = {l["from"]: l["to"] for l in chosen}
    has_pred = {l["to"] for l in chosen}
    root = {}
    for tid in sorted(tracks["track_id"].unique()):
        if tid in has_pred:
            continue
        node = int(tid)
        while True:
            root[node] = int(tid)
            if node not in successor:
                break
            node = successor[node]
    out = tracks.copy()
    out["raw_track_id"] = out["track_id"]
    out["track_id"] = out["raw_track_id"].map(root).astype(np.int64)
    return out, chosen


def prepare_tracks(tracks: pd.DataFrame, width: int, height: int,
                   cfg: StitchConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Part A hook between perception and the world model: fuse riders, then stitch fragments."""
    fused, riders = fuse_riders(tracks)
    stitched, links = stitch(fused, cfg, width, height)
    return stitched, {"riders": riders, "links": links,
                      "tracks_before": int(tracks["track_id"].nunique()),
                      "tracks_after": int(stitched["track_id"].nunique())}
