"""Causal accident-risk model for Part B.

update() sees only the tracks visible at time t (from Part B's own tracker) plus optional signal
states read from the current frame, and keeps its own history. It never reads the video, future
samples, Part A output or stitched tracks.

Pipeline per update: per-track causal kinematics -> pairwise conflict features (time to closest
approach, predicted minimum clearance, closing speed, braking) with exclusions (followers, riders,
occupants, pedestrians off the carriageway, parked pairs) and modifiers (signal-expected stops,
red-light threats, wrong-way, pedestrians on the carriageway) -> frame score = worst pair ->
causal smoother (fast rise, slow decay) -> monotonic calibration to P(accident within 5 s).

Distances are in pair box heights and speeds in box heights per second, so thresholds hold at
every depth of this oblique view (no metric calibration exists).
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.contracts import TrackState
from src.events.base import front_point
from src.scene.geometry import Scene, point_in_polygon

PERSON, BICYCLE = 0, 1
VEHICLES = frozenset((2, 3, 4, 5))
TWO_WHEELERS = frozenset((1, 2))


@dataclass
class RiskConfig:
    history_s: float = 2.0
    vel_window_s: float = 0.6
    min_fit_samples: int = 3          # velocity/acceleration fits never use fewer samples than this
    min_age_s: float = 0.5
    border_px: float = 8.0
    stale_s: float = 1.0
    signal_hold_s: float = 1.0       # measured from last known reading, never refreshed by unknown
    moving_h_per_s: float = 0.3
    max_pair_dist_h: float = 8.0
    horizon_s: float = 7.5            # pairs whose closest approach is further away are ignored
    clearance_h: float = 0.5          # predicted clearance scale (box heights)
    time_tau_s: float = 1.5           # urgency decays with time to closest approach beyond ...
    time_free_s: float = 4.5          # ... this; a course closing within it is fully urgent
    course_clear_h: float = 0.35      # a pair is on a collision course while clearance stays below this
    course_tca_tol_s: float = 0.6     # ... and each tca is the previous one minus elapsed time, within this
    persist_tau_s: float = 1.0        # score grows with how long the course has held (normal conflicts resolve)
    closing_ref_h_per_s: float = 1.5
    follow_cos: float = 0.87          # |cos| above this = parallel/opposing paths (within 30 degrees)
    rear_end_tca_s: float = 1.0       # parallel or queueing pairs count only as an imminent rear-end:
    rear_end_closing_h_per_s: float = 2.0  # closing this fast, this soon, with nobody braking
    comfortable_decel_h_per_s2: float = 2.0
    braking_h_per_s2: float = 0.8     # deceleration counted as "braking"
    red_close_s: float = 0.8          # within this time of the line and not braking -> red-light threat
    red_boost: float = 1.5
    wrong_way_deg: float = 135.0
    wrong_way_boost: float = 1.5
    pedestrian_boost: float = 1.3
    ped_margin_h: float = 0.5
    hard_brake_h_per_s2: float = 2.5
    hard_brake_boost: float = 1.3
    rise_alpha: float = 0.6           # fraction of an upward jump taken per update
    decay_tau_s: float = 2.0          # downward half-life scale
    # Raw score at P = 0.5. Chosen so synthetic collision courses (0.5-2 box heights/s) cross 0.5 3.3-4.8 s
    # before contact while C3905's normal traffic peaks at raw 0.24 (P 0.04). In-sample, no accident in
    # the samples: refit with Calibrator.fit_isotonic on dev folds once labelled accidents exist.
    calib_mid: float = 0.40
    calib_slope: float = 0.05


@dataclass
class _Track:
    samples: deque = field(default_factory=deque)     # (t, fx, fy, h, x1, y1, x2, y2)
    vels: deque = field(default_factory=deque)        # (t, vx, vy)
    last_t: float = 0.0
    first_t: float = 0.0
    wrong_way_since: float | None = None


def _ls_slope(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    tc = t - t.mean()
    d = float(tc @ tc)
    if d <= 0:
        return np.zeros(y.shape[1:])
    return (tc[:, None] * (y - y.mean(axis=0))).sum(axis=0) / d


class Calibrator:
    """Monotonic map raw -> probability: logistic by default, isotonic (PAV) once labels exist."""

    def __init__(self, mid: float, slope: float):
        self.mid, self.slope = mid, slope
        self.xs: np.ndarray | None = None
        self.ys: np.ndarray | None = None

    def __call__(self, raw: float) -> float:
        if self.xs is not None:
            return float(np.interp(raw, self.xs, self.ys))
        # Logistic rescaled so that no conflict at all (raw 0) is exactly 0 risk.
        base = 1.0 / (1.0 + math.exp(self.mid / self.slope))
        p = 1.0 / (1.0 + math.exp(-(raw - self.mid) / self.slope))
        return float(max(0.0, (p - base) / (1.0 - base)))

    def fit_isotonic(self, raw: np.ndarray, labels: np.ndarray) -> None:
        order = np.argsort(raw, kind="stable")
        x, y = np.asarray(raw, float)[order], np.asarray(labels, float)[order]
        blocks = [[yi, 1.0, xi, xi] for xi, yi in zip(x, y)]   # mean, weight, x_lo, x_hi
        merged: list[list[float]] = []
        for b in blocks:
            merged.append(b)
            while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
                b2, b1 = merged.pop(), merged.pop()
                w = b1[1] + b2[1]
                merged.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
        self.xs = np.array([(b[2] + b[3]) / 2 for b in merged])
        self.ys = np.clip(np.array([b[0] for b in merged]), 0.0, 1.0)

    def to_dict(self) -> dict:
        d = {"mid": self.mid, "slope": self.slope}
        if self.xs is not None:
            d["isotonic"] = {"x": self.xs.tolist(), "y": self.ys.tolist()}
        return d


class RiskModel:
    def __init__(self, scene: Scene, atlas: dict | None = None, cfg: RiskConfig | None = None):
        self.scene = scene
        self.cfg = cfg or RiskConfig()
        self.calibrator = Calibrator(self.cfg.calib_mid, self.cfg.calib_slope)
        self._atlas = atlas
        self.reset({})

    @classmethod
    def from_files(cls, camera: str | Path, width: int, height: int, atlas: str | Path | None = None,
                   cfg: RiskConfig | None = None) -> "RiskModel":
        scene = Scene.load(camera, width, height)
        data = json.loads(Path(atlas).read_text(encoding="utf-8")) if atlas else None
        return cls(scene, data, cfg)

    # -- lifecycle ---------------------------------------------------------------------------
    def reset(self, meta: dict) -> None:
        self.meta = dict(meta)
        self.tracks: dict[int, _Track] = {}
        # Tiny per-ID counts survive kinematic eviction if the tracker reacquires
        # the same ID. Reset clears them between videos; no future votes enter.
        self.class_votes: dict[int, Counter] = {}
        self.known_signals: dict[str, tuple[str, float]] = {}
        self.signal_states: dict[str, str] = {}
        self.smoothed = 0.0
        self.courses: dict[tuple[int, int], tuple[float, float, float]] = {}   # pair -> (since, last_t, last_tca)
        self.last_t: float | None = None
        self.last_score = 0.0
        self.last_evidence: dict = {}

    # -- per-track kinematics ----------------------------------------------------------------
    def _observe(self, tracks: list[TrackState], t: float) -> None:
        c = self.cfg
        for tr in tracks:
            st = self.tracks.get(tr.track_id)
            if st is None:
                st = self.tracks[tr.track_id] = _Track(first_t=t)
            x1, y1, x2, y2 = tr.xyxy
            fx, fy = tr.foot_point
            st.samples.append((t, fx, fy, y2 - y1, x1, y1, x2, y2))
            self.class_votes.setdefault(tr.track_id, Counter())[tr.cls] += 1
            st.last_t = t
            while len(st.samples) > c.min_fit_samples and st.samples[0][0] < t - c.history_s:
                st.samples.popleft()
            # The window always holds at least min_fit_samples: when the runtime guard widens the stride
            # (0.5 s between updates at stride 15), a fixed 0.6 s window would read every track as still.
            recent = [s for s in st.samples if s[0] >= t - c.vel_window_s]
            if len(recent) < c.min_fit_samples:
                recent = list(st.samples)[-c.min_fit_samples:]
            win = np.array(recent)
            v = _ls_slope(win[:, 0], win[:, 1:3]) if len(win) >= c.min_fit_samples else np.zeros(2)
            st.vels.append((t, float(v[0]), float(v[1])))
            while len(st.vels) > c.min_fit_samples and st.vels[0][0] < t - c.vel_window_s:
                st.vels.popleft()
        for tid in [k for k, s in self.tracks.items() if t - s.last_t > c.stale_s]:
            del self.tracks[tid]

    def _state_arrays(self, t: float) -> dict | None:
        c = self.cfg
        rows = []
        for tid, st in self.tracks.items():
            if st.last_t != t or t - st.first_t < c.min_age_s:
                continue
            _, fx, fy, h, x1, y1, x2, y2 = st.samples[-1]
            _, vx, vy = st.vels[-1]
            vel = np.array(st.vels)
            acc = _ls_slope(vel[:, 0], vel[:, 1:3]) if len(vel) >= 3 else np.zeros(2)
            cls = self._class_vote(tid)
            rows.append((tid, cls, fx, fy, max(h, 1.0), x1, y1, x2, y2, vx, vy, acc[0], acc[1]))
        if not rows:
            return None
        a = np.array(rows, float)
        s = {"id": a[:, 0].astype(int), "cls": a[:, 1].astype(int), "p": a[:, 2:4], "h": a[:, 4],
             "box": a[:, 5:9], "v": a[:, 9:11], "acc": a[:, 11:13]}
        speed = np.linalg.norm(s["v"], axis=1)
        s["speed_h"] = speed / s["h"]
        unit = np.where(speed[:, None] > 0, s["v"] / np.maximum(speed[:, None], 1e-9), 0.0)
        s["decel_h"] = -(s["acc"] * unit).sum(axis=1) / s["h"]
        # Even a newly seen bike can explain its rider before the bike has enough
        # samples for a velocity fit. Use only boxes observed in this update.
        bikes = [st.samples[-1][4:8] for tid, st in self.tracks.items()
                 if st.last_t == t and self._class_vote(tid) in TWO_WHEELERS]
        rider = np.zeros(len(rows), bool)
        for x1, y1, x2, y2 in bikes:
            w, h = x2 - x1, y2 - y1
            fx, fy = s["p"].T
            rider |= ((s["cls"] == PERSON) & (fx >= x1 - .25 * w) & (fx <= x2 + .25 * w)
                      & (fy >= y1 - .25 * h) & (fy <= y2 + .25 * h))
        s["rider"] = rider
        return s

    def _class_vote(self, tid: int) -> int:
        """Lifetime causal majority; lowest class ID breaks ties deterministically."""
        counts = self.class_votes[tid]
        return min(counts, key=lambda cls: (-counts[cls], cls))

    def _hold_signals(self, readings: dict | None, t: float) -> dict[str, str]:
        readings = readings or {}
        states = {}
        for sid in self.known_signals.keys() | readings.keys():
            state = readings.get(sid, "unknown")
            if state in ("red", "yellow", "green"):
                self.known_signals[sid] = (state, t)
            else:
                previous, known_t = self.known_signals.get(sid, ("unknown", -math.inf))
                state = previous if t - known_t <= self.cfg.signal_hold_s else "unknown"
            states[sid] = state
        return states

    # -- modifiers ---------------------------------------------------------------------------
    def _signal_modifiers(self, s: dict, signal_states: dict | None) -> tuple[np.ndarray, np.ndarray]:
        """(expected_stop, red_threat) per track for the approaches controlled by a signal."""
        c = self.cfg
        n = len(s["id"])
        stop, threat = np.zeros(n, bool), np.zeros(n, bool)
        if not signal_states:
            return stop, threat
        for line_id, (a, b) in self.scene.stop_lines.items():
            meta = self.scene.stop_line_meta.get(line_id, {})
            state = signal_states.get(meta.get("signal"))
            approach = self.scene.approaches.get(meta.get("approach"))
            if state not in ("red", "yellow") or approach is None:
                continue
            d = np.asarray(approach["direction"], float) * [self.scene.width, self.scene.height]
            d /= np.linalg.norm(d)
            normal = np.array([-(b - a)[1], (b - a)[0]])
            normal /= np.linalg.norm(normal)
            if normal @ d < 0:
                normal = -normal
            x1, y1, x2, y2 = s["box"].T
            front = front_point(x1, y1, x2, y2, d)
            dist = (front - a) @ normal                                      # < 0: before the line
            seg = b - a
            along = ((front - a) @ seg) / (seg @ seg)
            v_along = s["v"] @ d
            candidate = (np.isin(s["cls"], list(VEHICLES)) & (dist < 0) & (along > -0.05) & (along < 1.05)
                         & (v_along > 0) & ((s["v"] @ d) > 0.5 * np.linalg.norm(s["v"], axis=1)))
            gap = np.maximum(-dist, 1.0)
            need = (v_along ** 2) / (2 * gap) / s["h"]                       # required decel, h/s^2
            braking = s["decel_h"] >= c.braking_h_per_s2
            can_stop = (need <= c.comfortable_decel_h_per_s2) | braking
            stop |= candidate & can_stop
            if state == "red":
                close = gap / np.maximum(v_along, 1e-6) <= c.red_close_s
                threat |= candidate & close & ~braking
        return stop, threat

    def _wrong_way(self, s: dict, t: float) -> np.ndarray:
        flags = np.zeros(len(s["id"]), bool)
        atlas = (self._atlas or {}).get("flow")
        if atlas is None:
            return flags
        count, heading = np.asarray(atlas["count"]), np.asarray(atlas["heading_deg"])
        res = np.asarray(atlas["resultant"])
        ny, nx = count.shape
        min_n = self._atlas["params"]["min_samples"]
        for i, tid in enumerate(s["id"]):
            st = self.tracks[tid]
            if s["cls"][i] not in VEHICLES or s["speed_h"][i] < 2 * self.cfg.moving_h_per_s:
                st.wrong_way_since = None
                continue
            cx = int(np.clip(s["p"][i, 0] / self.scene.width * nx, 0, nx - 1))
            cy = int(np.clip(s["p"][i, 1] / self.scene.height * ny, 0, ny - 1))
            if count[cy, cx] < min_n or res[cy, cx] < 0.8:
                st.wrong_way_since = None
                continue
            h = math.degrees(math.atan2(s["v"][i, 1], s["v"][i, 0]))
            dev = abs((h - heading[cy, cx] + 180) % 360 - 180)
            if dev > self.cfg.wrong_way_deg:
                st.wrong_way_since = t if st.wrong_way_since is None else st.wrong_way_since
                flags[i] = t - st.wrong_way_since >= 0.5
            else:
                st.wrong_way_since = None
        return flags

    # -- pair scoring ------------------------------------------------------------------------
    def _pair_scores(self, s: dict, signal_states: dict | None, t: float) -> tuple[float, dict]:
        c = self.cfg
        n = len(s["id"])
        if n < 2:
            return 0.0, {}
        stop, threat = self._signal_modifiers(s, signal_states)
        wrong = self._wrong_way(s, t)
        v = np.where(stop[:, None], 0.0, s["v"])                   # an expected stop will not cross the line
        # Pedestrians count on the carriageway, crossings included (a car driving at someone on a
        # crossing is a real precursor); off-crossing pedestrians (jaywalkers) are boosted.
        if self.scene.road is not None:
            # Same rule as the jaywalking engine: within ~1 m (half a body height) of a curb, refuge,
            # sidewalk or median counts as off the carriageway.
            person = s["cls"] == PERSON
            margin = c.ped_margin_h * s["h"]
            ped_on_carriageway = person & ~self.scene.near_safe_area(s["p"], margin, ("islands", "sidewalks", "median"))
            ped_on_road = ped_on_carriageway & ~self.scene.near_safe_area(s["p"], margin)
        else:
            ped_on_carriageway = ped_on_road = np.zeros(n, bool)
        i, j = np.triu_indices(n, 1)
        cls_i, cls_j = s["cls"][i], s["cls"][j]
        scale = (s["h"][i] + s["h"][j]) / 2
        r = s["p"][j] - s["p"][i]
        dv = v[j] - v[i]
        dist = np.linalg.norm(r, axis=1)
        vv = (dv * dv).sum(axis=1)
        tca = np.where(vv > 1e-9, np.maximum(0.0, -(r * dv).sum(axis=1) / np.maximum(vv, 1e-9)), np.inf)
        dmin = np.linalg.norm(r + np.where(np.isfinite(tca), tca, 0.0)[:, None] * dv, axis=1)
        closing = -(r * dv).sum(axis=1) / np.maximum(dist, 1e-9) / scale
        moving_i, moving_j = s["speed_h"][i] > c.moving_h_per_s, s["speed_h"][j] > c.moving_h_per_s
        cos = (s["v"][i] * s["v"][j]).sum(axis=1) / np.maximum(
            np.linalg.norm(s["v"][i], axis=1) * np.linalg.norm(s["v"][j], axis=1), 1e-9)

        # Fast pairs are far apart seconds before contact: the reach grows with closing speed.
        reach = c.max_pair_dist_h + np.maximum(closing, 0.0) * c.horizon_s
        keep = (dist / scale <= reach) & (closing > 0) & (tca <= c.horizon_s) & (moving_i | moving_j)
        # Boxes cut by the frame border have no reliable foot point or velocity (entering/leaving).
        b = c.border_px
        inside = ((s["box"][:, 0] > b) & (s["box"][:, 1] > b) & (s["box"][:, 2] < self.scene.width - b)
                  & (s["box"][:, 3] < self.scene.height - b))
        keep &= inside[i] & inside[j]
        # The bike represents the attached person against every other road user,
        # not just in the rider-bike pair. Reevaluate attachment each update.
        keep &= ~s["rider"][i] & ~s["rider"][j]
        people_i, people_j = cls_i == PERSON, cls_j == PERSON
        keep &= ~(people_i & people_j)
        keep &= ~(people_i & ~ped_on_carriageway[i]) & ~(people_j & ~ped_on_carriageway[j])
        keep &= ~(np.isin(cls_i, [BICYCLE]) & np.isin(cls_j, [BICYCLE]))
        # Riders and occupants duplicate their vehicle: foot on the other's box.
        keep &= ~self._attached(s, i, j) & ~self._attached(s, j, i)
        # Conflict geometry. In this oblique view parallel lanes project almost on top of each other, so
        # constant-velocity clearance alone flags every overtaking, oncoming or queueing pair.
        veh_i, veh_j = np.isin(cls_i, list(VEHICLES)), np.isin(cls_j, list(VEHICLES))
        braking_i = s["decel_h"][i] >= c.braking_h_per_s2
        braking_j = s["decel_h"][j] >= c.braking_h_per_s2
        rear_end = (tca < c.rear_end_tca_s) & (closing > c.rear_end_closing_h_per_s) & ~(braking_i | braking_j)
        both = moving_i & moving_j
        crossing = both & (np.abs(cos) <= c.follow_cos)
        oncoming = both & (cos < -c.follow_cos) & (wrong[i] | wrong[j])
        following = both & (cos > c.follow_cos) & rear_end
        # One road user stationary: a moving vehicle closing on a pedestrian standing in the road counts;
        # closing on a stationary vehicle is queueing unless an imminent rear-end.
        one_still = moving_i ^ moving_j
        mover_is_vehicle = np.where(moving_i, veh_i, veh_j)
        # A pedestrian standing on a crossing is waiting or yielding (normal); standing in the road
        # outside a crossing is the hazard. Walking across a crossing is covered by `crossing`.
        still_is_person = np.where(moving_i, people_j & ped_on_road[j], people_i & ped_on_road[i])
        still_case = one_still & mover_is_vehicle & (still_is_person | rear_end)
        keep &= crossing | oncoming | following | still_case
        if not keep.any():
            return 0.0, {}

        clear = np.exp(-(dmin / scale / c.clearance_h) ** 2)
        persistence = self._course_persistence(s["id"][i], s["id"][j], keep & (dmin / scale < c.course_clear_h),
                                               tca, t)
        urgency = np.exp(-np.maximum(tca - c.time_free_s, 0.0) / c.time_tau_s)
        severity = 0.5 + 0.5 * np.clip(closing / c.closing_ref_h_per_s, 0.0, 1.0)
        boost = np.ones(len(i))
        for flag, factor in ((threat, c.red_boost), (wrong, c.wrong_way_boost),
                             (ped_on_road, c.pedestrian_boost),
                             (s["decel_h"] >= c.hard_brake_h_per_s2, c.hard_brake_boost)):
            boost *= np.where(flag[i] | flag[j], factor, 1.0)
        held = 1.0 - np.exp(-persistence / c.persist_tau_s)
        score = np.where(keep, clear * urgency * severity * boost * held, 0.0)
        k = int(np.argmax(score))
        if score[k] <= 0.0:
            return 0.0, {}
        ev = {"pair": [int(s["id"][i[k]]), int(s["id"][j[k]])], "cls": [int(cls_i[k]), int(cls_j[k])],
              "tca_s": round(float(tca[k]), 2), "course_held_s": round(float(persistence[k]), 2), "dmin_h": round(float(dmin[k] / scale[k]), 2),
              "closing_h_per_s": round(float(closing[k]), 2), "dist_h": round(float(dist[k] / scale[k]), 2),
              "flags": [name for name, f in (("red_threat", threat), ("wrong_way", wrong),
                                             ("pedestrian_on_road", ped_on_road)) if f[i[k]] or f[j[k]]],
              "expected_stops": int(stop.sum())}
        return float(score[k]), ev

    def _course_persistence(self, ids_a, ids_b, on_course, tca, t: float) -> np.ndarray:
        """Seconds each pair has stayed on a consistent collision course (0 when it just started)."""
        c = self.cfg
        out = np.zeros(len(ids_a))
        alive = {}
        for k in np.flatnonzero(on_course):
            key = (int(ids_a[k]), int(ids_b[k]))
            prev = self.courses.get(key)
            since = t
            if prev is not None:
                p_since, p_t, p_tca = prev
                if t - p_t <= c.stale_s and abs(tca[k] - (p_tca - (t - p_t))) <= c.course_tca_tol_s:
                    since = p_since
            alive[key] = (since, t, float(tca[k]))
            out[k] = t - since
        # A pair not scored this update (a detection missed) keeps its course until stale; a pair
        # scored and off course is dropped.
        scored = {(int(a), int(b)) for a, b in zip(ids_a, ids_b)}
        for key, val in self.courses.items():
            if key not in alive and key not in scored and t - val[1] <= c.stale_s:
                alive[key] = val
        self.courses = alive
        return out

    @staticmethod
    def _attached(s: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Track a is a rider/occupant of track b: a is a person whose foot lies on b's box."""
        x1, y1, x2, y2 = s["box"][b].T
        w, h = x2 - x1, y2 - y1
        fx, fy = s["p"][a].T
        on = (fx >= x1 - 0.25 * w) & (fx <= x2 + 0.25 * w) & (fy >= y1 - 0.25 * h) & (fy <= y2 + 0.25 * h)
        return (s["cls"][a] == PERSON) & (s["cls"][b] != PERSON) & on

    # -- public step -------------------------------------------------------------------------
    def update(self, tracks: list[TrackState], t_sec: float, signal_states: dict | None = None) -> float:
        """Risk at t_sec from tracks observed at t_sec (and this model's own past). In [0, 1]."""
        if self.last_t is not None and t_sec < self.last_t:
            raise ValueError("update() must be called with non-decreasing time")
        self._observe(tracks, t_sec)
        self.signal_states = self._hold_signals(signal_states, t_sec)
        state = self._state_arrays(t_sec)
        raw, ev = (0.0, {}) if state is None else self._pair_scores(state, self.signal_states, t_sec)
        dt = 0.0 if self.last_t is None else t_sec - self.last_t
        if raw >= self.smoothed:
            self.smoothed += self.cfg.rise_alpha * (raw - self.smoothed)
        else:
            self.smoothed = max(raw, self.smoothed * math.exp(-dt / self.cfg.decay_tau_s))
        self.last_t = t_sec
        self.last_score = min(1.0, max(0.0, self.calibrator(self.smoothed)))
        self.last_evidence = {"t": round(t_sec, 3), "raw": round(raw, 4), "smoothed": round(self.smoothed, 4),
                              "risk": round(self.last_score, 4), **ev}
        return self.last_score
