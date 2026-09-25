# Architecture and contracts

The organizer spec (`docs/task_spec.md`) and the starter kit code (`run_submission.py`, `evaluate.py`) are the source of truth. Where this document conflicts with them, they win. Stage IDs (for example "Stage 5", "7.1") refer to the lead's build sequence; teammate task IDs (J1–J7, V1–V7) refer to `docs/TEAM_BRIEF.md`.

## Data flow

```
                        configs/camera.yaml (law)     configs/atlas.json (normality)
                                   \                    /
video -> Detector -> Tracker -> world model (tracks, kinematics, pairs)
      -> event engines (rule AND atlas; learned classifiers where they win)
      -> rich Event -> segment postprocess -> adapter -> [[start_sec, end_sec, label], ...]

Part B (RiskEstimator.step, fed frame by frame by the harness):
      own Detector/Tracker state -> causal features (<= t) -> RiskModel -> smoother -> calibrator -> risk(t)
```

What the harness actually does, per video (`run_submission.py`):

1. `solution.py` is imported **once**, before the video loop (line 146). Anything loaded at import time is outside every video's time budget.
2. The per-video timer starts (line 167), then `detect_events(path)` runs to completion (line 175).
3. Only after Part A returns, a fresh `RiskEstimator()` is constructed and fed every frame (line 188).
4. Part A + Part B together must finish within 3 × video duration. If the total overruns, **both** the events and the risk curve for that video are discarded (lines 196–198).

Consequences for the design:

- **Model loading lives at module import** (`src/models_registry.py`), never inside `detect_events` or `RiskEstimator.__init__`.
- **Part A cannot read Part B's curve at runtime**, because Part B has not run yet. When Part A wants risk evidence (accident candidates, Stage 8.3), it runs the same causal risk code on its own decoded frames. Reusing Part B *code* inside Part A is allowed; reusing Part A *output* inside Part B is not.
- Part B may never use Part A output, the video file, or frames after t.
- Runtime is one budget: a slow Part B also wipes Part A's events for that video.

## Time convention

All code and all annotations use the harness convention:

- `fps = cv2.CAP_PROP_FPS` (fallback 25.0), `n_frames = cv2.CAP_PROP_FRAME_COUNT`, `duration = n_frames / fps` (`run_submission.py` lines 55–66).
- Frame `i` has timestamp `t = i / fps` (line 119). Do not use `CAP_PROP_POS_MSEC`.
- The harness rounds event times to 3 decimals and clamps `end` to `duration` (lines 91, 95). An event shorter than about 1 ms can collapse to `start == end` after rounding and make the file invalid, so our segments are always at least one frame long.

## Module layout

| Path | Contents | Owner |
|---|---|---|
| `solution.py` | Official interface; thin wrapper over `src/` | Iko |
| `run_submission.py`, `evaluate.py`, `examples/` | Starter kit, byte-identical | Organizers |
| `src/contracts.py` | Dataclasses and Protocols below | Iko |
| `src/models_registry.py` | Loads every weight file once at import from explicit local paths; exposes read-only model handles to Part A and Part B | Iko |
| `src/perception/detector.py` | `Detector` implementations (YOLO11m default, RT-DETR-L challenger) | Iko (benchmark: Jalol J3) |
| `src/perception/tracker.py` | `Tracker` implementations (ByteTrack default) | Iko (stress test: Jalol J4) |
| `src/scene/geometry.py` | Load `camera.yaml`, point-in-polygon, lane lookup, line crossing | Iko (map draft: Javohir V1) |
| `src/scene/signal.py` | Wraps `classify_signal` | Javohir module (V4), integrated by Iko |
| `src/features/tracks.py` | Per-track kinematics | Iko |
| `src/features/interactions.py` | Pairwise interaction features | Iko |
| `src/atlas/` | Flow-atlas builder and lookup | Iko |
| `src/events/base.py` | State-machine base | Iko |
| `src/events/<class>.py` | One engine per class | Iko |
| `src/events/nm_classifier.py` | Wraps `NMClassifier` | Javohir module (V6), integrated by Iko |
| `src/events/hazard.py` | Wraps `HazardDetector` for `fire_smoke` | Jalol module (J5), integrated by Iko |
| `src/anticipation/` | Causal risk pipeline, `RiskModel` implementations | Iko |
| `src/postprocess/segments.py` | Merge, drop blips, same-class union, clamp | Iko |
| `src/adapter.py` | Rich `Event` to official `[start, end, label]` | Iko |
| `training/` | Training code per model (`nm/`, `hazard/`, `risk/`, `detector/`); never imported by `solution.py` | Iko, teammates' code ported from `contrib/` |
| `weights/download.sh` | Fetches release assets and checks SHA256 | Iko |
| `scripts/` | Cache builder, dev_eval, renders, EDA | Iko |
| `tests/` | Invariants, causality, determinism | Iko |
| `contrib/<name>/` | Teammate deliverables (on `team/<name>` branches) | Jalol, Javohir |

Teammate code is delivered in `contrib/` and ported into `src/` by Iko; `solution.py` never imports from `contrib/`.

## Models registry

`src/models_registry.py` runs at import of `solution.py`:

- Loads detector, hazard detector, near-miss classifier and any risk model weights from explicit paths under `weights/`. A missing file raises a clear error naming `weights/download.sh`. Nothing may trigger a network download (Ultralytics fetches missing weights by name, so always pass the full path).
- Model objects are shared read-only by Part A and Part B.
- **Tracker state is never shared.** Part A and each `RiskEstimator` instance create their own tracker.
- Seeds are fixed here (Python, NumPy, torch) and inference runs in deterministic mode.

## Interfaces (in `src/contracts.py`)

```python
class Detector(Protocol):
    def predict(self, frames: list[np.ndarray]) -> list[Detections]: ...

@dataclass
class Detections:           # one frame
    xyxy: np.ndarray        # (N,4) float32, pixels
    cls: np.ndarray         # (N,) int, internal class ids below
    score: np.ndarray       # (N,) float32

class Tracker(Protocol):
    def update(self, dets: Detections, frame_idx: int) -> list[Track]: ...
    def reset(self) -> None: ...

class RiskModel(Protocol):  # causal
    def reset(self) -> None: ...
    def step(self, feats_t: dict[str, float]) -> float: ...   # features from frames <= t only
```

Internal classes: 0 person, 1 bicycle, 2 motorcycle, 3 car, 4 bus, 5 truck.

The rich event already exists: `Event(label, start, end, confidence, track_ids, evidence)`. The adapter strips it to the official format only at output.

### Teammate module interfaces (exactly as in `docs/TEAM_BRIEF.md`)

```python
# contrib/javohir/signal/signal_classifier.py  (V4)
def classify_signal(crop_bgr) -> str:
    # "red", "yellow", "green" or "unknown"; deterministic; no network

# contrib/jalol/hazard/hazard_detector.py  (J5)
class HazardDetector:
    def __init__(self, weights_path: str, conf: float, device: str = "cuda"): ...
    def predict(self, frame_bgr) -> list[dict]:
        # [{"label": "fire" or "smoke", "xyxy": [x1, y1, x2, y2], "score": float}, ...], pixel coords

# contrib/javohir/nm/nm_classifier.py  (V6)
class NMClassifier:
    def __init__(self, model_path: str): ...
    def predict_proba(self, df) -> np.ndarray:
        # df has the f_* columns of windows.parquet; returns (n_rows, 3) for [none, near_miss, accident]
```

## Cache (`cache/<video_id>/`, gitignored)

| File | Columns |
|---|---|
| `meta.json` | fps, n_frames, width, height, duration, coords (`pixels` for perception), request (source identity, pipeline fingerprints, stride), schema_version, complete, row_counts |
| `frames.parquet` | frame, t; sampled inference frames, including frames with zero detections |
| `detections.parquet` | frame, t, x1, y1, x2, y2, cls, score |
| `tracks.parquet` | frame, t, track_id, cls, score, x1, y1, x2, y2, fx, fy (foot point = bottom-center) |
| `world.parquet` | frame, t, track_id, cls, gx, gy, vx, vy, ax, ay, speed, heading, lane_id, in_road, in_crosswalk, in_intersection, lane_dir_err, stationary_s, age_s |
| `pairs.parquet` | frame, t, id_a, id_b, dist, closing_speed, tca, dmin, heading_diff, path_conflict, lane_conflict, brake_a, brake_b, swerve_a, swerve_b |

`t` is always `frame / fps`. `gx, gy` are homography ground coordinates only if real correspondences exist, else normalized image coordinates; world-cache metadata must record that basis when Stage 4 is implemented. The current perception cache contains pixel coordinates only. Velocities come from a smoothed estimate over a short window, never adjacent-frame differences.

Pair features: `r = p_b - p_a`, `v = v_b - v_a`, `tca = max(0, -(r·v) / (|v|² + eps))`, `dmin = |r + tca·v|`. Keep only pairs with lane/path conflict or among each object's k nearest neighbours.

### `windows.parquet` (handed to Javohir for V6)

One row per candidate window from the near-miss/accident rule engine.

| Column | Type | Meaning |
|---|---|---|
| `window_id` | str | Unique id, `<video_id>:<id_a>:<id_b>:<start_frame>` |
| `video` | str | Video file name |
| `t_start`, `t_end` | float | Window bounds in seconds |
| `label` | str | `accident`, `near_miss` or `none`, from locked dev labels |
| `f_*` | float | Aggregated features computed only from frames inside the window (for example `f_min_tca`, `f_min_dmin`, `f_max_closing_speed`, `f_max_decel`, `f_heading_change`, `f_duration`) |

## Flow atlas (Stage 5)

Built offline from cached sample tracks by `src/atlas/`, stored in `configs/atlas.json`, loaded read-only at import. Part B may use it because it is a static prior computed before any test video.

| Key | Content |
|---|---|
| `grid` | Cell size and frame size the atlas was built on |
| `flow` | Per cell: dominant heading, heading spread, speed percentiles, sample count |
| `movements` | Origin region to destination region clusters, with counts; legal/unknown flag cross-checked against `camera.yaml` |
| `dwell` | Per cell: stop frequency and stop-duration percentiles; marked signal-queue zones |
| `interaction_baseline` | Percentiles (for example p1, p5, p50) of `tca`, `dmin`, `closing_speed` over normal traffic |
| `coverage` | Cells and movements with too few samples, marked low-confidence |

The atlas estimates normal movement only where coverage is adequate; `camera.yaml` supplies independently verified scene facts. No `camera.md` was provided with the samples. Lane directions can be estimated from frames and observed flow, but legal permissions and prohibitions need markings, signs or organizer guidance. Frequent movement is not proof of legality; unsupported restrictions remain unknown.

## Reference points per rule (from the official start/end conventions)

| Use | Point |
|---|---|
| Lane, region, crosswalk, road membership | Foot point (bottom-center) |
| `red_light` start | Front edge of the box in the direction of travel crossing the stop line |
| `solid_line_crossing` | Bottom-left and bottom-right corners (wheel proxy); ends when the whole bottom edge is in the new lane |
| `stopped_vehicle` | Speed below threshold for ≥ 10 s, excluding signal-queue zones and red phase |
| `stop_line` | Front edge stopped past the stop line on red, outside the intersection; ends when the signal turns green |

## `configs/camera.yaml`

```yaml
frame: {width: , height: }
coords: normalized            # all points in [0,1]
road: [[x,y], ...]
lanes:
  - {id: , polygon: [...], direction: [dx,dy], movements: [straight,left,right,u]}
stop_lines: [{id: , points: [[x,y],[x,y]], lanes: [...]}]
crosswalks: [{id: , polygon: [...]}]
solid_lines: [{id: , polyline: [...]}]
intersection: [...]
signal: {visible: true|false, roi: [x1,y1,x2,y2], controls: [lane ids]}
allowed_movements: [{from: lane_id, to: exit_id}]
unknown_movements: [{from: lane_id, to: exit_id, note: }]
homography: {image_points: [...], ground_points_m: [...]}   # omit if no real correspondences
```

## `configs/events.yaml`

```yaml
wrong_way: {enabled: false, min_persist_s: , angle_deg: , merge_gap_s: , min_dur_s: }
```

Every class defaults to `enabled: false` and is switched on only after it shows near-zero false fires on all sample footage (the Score A formula adds any class we predict to the average, and the test set has classes the samples do not).

Postprocess rules: same-class segments never overlap (two simultaneous same-class events become one covering segment; the harness would otherwise **drop** the later-starting one, not merge it); events running past the end get `end = duration`; clamp to `0 <= start < end <= duration`; minimum length one frame.

## Part B contract (official, from `evaluate.py`)

- Interface: `reset(meta)` with `video_id, fps, width, height, n_frames`; `step(frame_bgr, t_sec) -> float in [0, 1]`, called for every frame.
- Frame labels (`frame_label`, lines 222–233), checked in this order: inside any accident `[s, e]` → ignored; in `[s-5, s)` of any accident → positive; in `[s-5, e]` of any near-miss → ignored; else negative. A pre-accident frame stays positive even if it also falls near a near-miss.
- `Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·min(1, mTTA/10)`. AP is chance-normalised. An alarm is a run of score ≥ 0.5; runs whose gap is under 2 s merge; an alarm matches if it starts in `[s-10, s)`; alarms starting on ignored frames are discarded.
- If the ground truth has no accidents at all, Part B is not scored and `M = Score_A`.
- Implications: an alarm starting 5–10 s early still matches and earns TTA, but those frames are AP negatives, so calibration is tuned on full Score B; near-miss periods are free for AP; every false alarm outside accidents lowers alarm precision.
- Pipeline order: features → `RiskModel.step` → causal smoother (fast rise, slow decay) → monotonic calibrator fit on dev folds. Causality test: two clips with identical prefix and different suffix give identical risk over the prefix.

## Validation protocol

Split by whole video only (leave-one-video-out when possible). Never split frames of one video across train and val. Report mean, std and worst video.

## Packaging

`.gitignore` blocks `*.pt/*.onnx/*.engine`. Weights go to a GitHub Release; `weights/download.sh` (run once, with internet, before evaluation) fetches them and verifies SHA256. Total ≤ 5 GB. Python ≥ 3.10.

The kit requires `opencv-python-headless`; Ultralytics pulls in `opencv-python`. Both provide `cv2` and can conflict in one environment. `requirements.txt` must resolve to exactly one OpenCV build; verify in the clean-install test.

## Default model choices

All open weights shippable in the package; no hosted APIs at inference.

- Detector: Ultralytics YOLO11m primary, Ultralytics RT-DETR-L challenger (Stage 3.1, from Jalol's J3 benchmark).
- Tracker: Ultralytics ByteTrack; BoT-SORT only if ID switches break events (Stage 3.2).
- Signal: HSV thresholds on fixed ROI first; small CNN only if HSV is below 98% on any video (V4).
- Near-miss/accident: XGBoost on window features, ships only if it beats the rule baseline (Stage 8.4).
- Fire/smoke: YOLO11s fine-tuned on licence-cleared data, ships only with zero false events on sample footage (Stage 8.5).
- Part B learned: logistic regression, then GRU or causal TCN, only if it beats the analytic baseline (Stage 9.1).
