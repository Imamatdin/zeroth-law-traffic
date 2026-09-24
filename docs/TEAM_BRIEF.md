# Zeroth Law Traffic: team brief for Jalol and Javohir

**How to use this file:** paste the whole file into your AI assistant, then tell it: "I am Jalol" or "I am Javohir". Do only your own section, in order. Every task says exactly what to produce, where to put it, and how to check it.

## 1. The project in one paragraph

WIUT Hackathon 2026, Computer Vision track. A fixed road camera. Part A: given an .mp4, return every traffic event as `[start_sec, end_sec, label]` using 14 official labels. Part B: at every frame, return the probability that an accident starts within 5 seconds, using only past frames. We are scored on a hidden test set from the same camera, plus a website and code quality. The full organizer spec is in `docs/task_spec.md` in the repo. Read the "Event classes" section before annotating.

## 2. How the team works

Iko owns the single main workflow: the codebase, the pipeline, all rules, Part B, the website, and the final submission. Your work is separate. It feeds his workflow through files he pulls in when he is ready. You never change his code.

Rules (these keep us from blocking each other):

- Work only inside `contrib/<your_name>/` on your own branch `team/<your_name>`. Never push to `main`. Never edit files outside your folder.
- Big files (videos, extracted frames, rendered clips, model weights) go to the shared drive folder `zlt-share/<your_name>/`, never into Git.
- Deliver exactly the files named in each task, with exactly the formats shown. If the format is different, Iko cannot use it.
- When you finish a task, post one line in the team chat and nothing else: `delivered: <path> | status: done or partial | note: <one short line>`
- If you are stuck or unsure: write the question in `contrib/<your_name>/QUESTIONS.md`, apply the default given in the task, and continue with your next task. Do not wait for an answer and do not message Iko directly for it. He answers the file in batches.
- Never make up results. If you did not measure something, write "not measured".
- Tasks marked "wait for Iko" start only when Iko posts the trigger line quoted in that task.

## 3. Setup (both of you)

1. Clone: `git clone https://github.com/Imamatdin/zeroth-law-traffic.git` then `cd zeroth-law-traffic`.
2. Create your branch: `git checkout -b team/<your_name>` (use `jalol` or `javohir`, lowercase).
3. Create your folder: `contrib/<your_name>/` with an empty `QUESTIONS.md`.
4. Python 3.11 virtual environment:
   - Windows: `py -3.11 -m venv .venv` then `.venv\Scripts\activate`
   - Linux/Mac: `python3.11 -m venv .venv` then `source .venv/bin/activate`
5. PyTorch with GPU. RTX 50-series cards (like the 5060) need a CUDA 12.8 build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
6. Other packages: `pip install ultralytics opencv-python pandas pyarrow numpy scikit-learn xgboost pyyaml matplotlib`
7. Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print `True` and your GPU name.
8. Download the sample videos and `camera.md` from the shared drive into `data/samples/` in the repo. `data/` is ignored by Git, so they will not be committed.
9. Write `contrib/<your_name>/env.md` with: GPU model, VRAM in GB, OS, Python version, torch version, output of step 7.
10. Commit only your folder: `git add contrib/<your_name>` then `git commit -m "<your_name>: env"` then `git push -u origin team/<your_name>`.

Deliverable: `contrib/<your_name>/env.md`. Post the delivered line.

## 4. Shared formats

### 4.1 The 14 labels (use exactly these strings)

`accident`, `near_miss`, `red_light`, `wrong_way`, `illegal_u_turn`, `stopped_vehicle`, `jaywalking`, `failure_to_yield`, `illegal_turn`, `solid_line_crossing`, `stop_line`, `congestion`, `road_obstacle`, `fire_smoke`

### 4.2 Event start and end rules (from the organizer spec; follow exactly)

| Label | Start | End |
|---|---|---|
| accident | first frame where contact is visible | all involved objects stop moving or leave the frame |
| near_miss | onset of braking or swerving to avoid a collision (no contact) | road users are clear of each other |
| red_light | front of the vehicle crosses the stop line while its signal is red | vehicle leaves the intersection or the frame |
| wrong_way | vehicle enters the opposing lane or moves against its lane direction | vehicle returns to a correct lane or leaves the frame |
| illegal_u_turn | vehicle starts turning (where U-turns are prohibited) | vehicle completes the turn |
| stopped_vehicle | vehicle stops on the carriageway (only if it then stays 10 s or more, and is not queued at a signal) | vehicle moves again or is removed |
| jaywalking | pedestrian steps onto the road outside a crossing | pedestrian leaves the road |
| failure_to_yield | vehicle enters a crossing while a pedestrian is on it or stepping onto it | vehicle leaves the crossing |
| illegal_turn | vehicle starts a turn from the wrong lane or in a prohibited direction | vehicle completes the turn |
| solid_line_crossing | wheel crosses a solid line | vehicle is fully in the new lane |
| stop_line | vehicle stops past the stop line on red, without entering the intersection | signal turns green |
| congestion | queue stops moving, traffic at standstill or crawling across all lanes of a direction | queue clears |
| road_obstacle | debris, animal or fallen object appears on the carriageway | obstacle is removed |
| fire_smoke | first visible smoke or fire | smoke clears or the video ends |

Extra rules:

- Two events of the same label at the same time: write ONE segment covering both.
- Events of different labels may overlap. A wrong-way car that then crashes is two events.
- If an event runs past the end of the video, end = the video duration.
- Times are seconds from the first frame, 2 decimals.
- Mark the frame the condition becomes true, not the moment it became obvious to you.

### 4.3 Event annotation file

Path: `contrib/<your_name>/events/<video_file_name_without_.mp4>.json`

```json
{
  "video": "sample_01.mp4",
  "annotator": "jalol",
  "duration": 312.48,
  "fps": 25.0,
  "events": [
    {"start": 12.40, "end": 18.92, "label": "wrong_way", "who": "white sedan, left lane", "note": ""}
  ],
  "uncertain": [
    {"t": 44.10, "question": "stopped_vehicle or queue at the signal?"}
  ]
}
```

Get duration and fps with OpenCV: `fps = cap.get(cv2.CAP_PROP_FPS)`, `duration = frame_count / fps`.

### 4.4 Tool: frame-accurate player (ask your AI to write it once)

Write `contrib/<your_name>/tools/player.py`: OpenCV window that plays a video, shows the current time in seconds (2 decimals) and frame number on screen, and supports: space = pause/play, d = next frame, a = previous frame, l = +1 s, j = -1 s, s = print current time to the terminal. Use `cap.set(cv2.CAP_PROP_POS_FRAMES, n)` for seeking.

### 4.5 Video split

Sort the sample file names alphabetically. Jalol annotates positions 1, 3, 5, ... Javohir annotates positions 2, 4, 6, ... Then each reviews the other's files.

## 5. JALOL (RTX 5060)

Do these in order.

### J1. Annotate your videos

- Input: your videos from 4.5, rules in 4.2.
- Steps: watch each video once fully at normal speed and list every candidate event with rough times. Then go back to each candidate with the player and find exact start and end frames per 4.2. Anything unclear goes into `uncertain`, not into `events`.
- Output: one file per video in `contrib/jalol/events/` (format 4.3).
- Check: every label is one of the 14; start < end; no two events of the same label overlap; end <= duration.

### J2. Review Javohir's videos

- Input: Javohir's event files for his videos.
- Steps: for each of his events, check start and end at ±15 s with the player. Scan the rest of the video at 2x for missed events.
- Output: `contrib/jalol/review/<video>.md` listing: agree / disagree per event with your times and reason, and any missed events with times. Do not edit his files.

### J3. Detector benchmark

Goal: help Iko choose the object detector. You measure, Iko decides.

Steps:

1. From all sample videos, save 150 frames spread evenly over time and videos. Include frames with dense traffic, the far end of the road, pedestrians, bikes and motorbikes, and low light if any. Save to `zlt-share/jalol/bench_frames/`.
2. Models: `yolo11m.pt` and `rtdetr-l.pt` (Ultralytics downloads them). Image sizes: 640 and 960. That is 4 settings.
3. For each setting, draw boxes on all 150 frames (confidence 0.25), save to `zlt-share/jalol/bench_out/<setting>/`.
4. Look at each drawn frame and count: missed cars/buses/trucks, missed pedestrians, missed bicycles/motorbikes, false boxes. Note whether misses are near or far.
5. Speed: on one full sample video, run `model.track(source=video, persist=True, tracker="bytetrack.yaml", imgsz=<size>, half=True, stream=True)` and measure total wall time including decoding. Compute FPS = frames / seconds. Record peak VRAM with `torch.cuda.max_memory_allocated() / 1e9`.

Output: `contrib/jalol/det_benchmark.md` with this table and 5 worst example frame names per setting:

| setting | missed vehicles (near/far) | missed persons | missed 2-wheelers | false boxes | FPS on 5060 | peak VRAM GB |
|---|---|---|---|---|---|---|

Do not train anything in this task.

### J4. Tracker stress renders

- Input: all event files from J1 and Javohir's (use times as given).
- Steps: for each annotated event, render a clip from 5 s before start to 5 s after end with `yolo11m.pt`, ByteTrack, track IDs drawn on boxes. Save to `zlt-share/jalol/track_clips/`. Watch each clip and write down whether the objects involved in the event keep the same ID the whole time.
- Output: `contrib/jalol/tracker_report.md`, one row per event: video, label, times, involved IDs, ID switches (count), notes (occlusion, crowding, far distance).
- Default if unsure whether something is an ID switch: count it as one and describe it.

### J5. Fire and smoke detector

Goal: a detector for fire_smoke that almost never fires on normal road footage.

Steps:

1. Licence check first. Open the licence files of FASDD_CV and D-Fire (dataset home pages). Write the exact licence name and link in `contrib/jalol/hazard/LICENCES.md`. If a licence is non-commercial or unclear, write that and do not use that dataset. Ask in `QUESTIONS.md` if both are unusable.
2. Train `yolo11s.pt` on the allowed dataset(s): classes 0 = fire, 1 = smoke. Check that class ids are consistent across datasets before merging (they can be swapped between datasets). imgsz 640, epochs 50, `seed=0`, `deterministic=True`, batch as large as fits in 8 GB.
3. False-fire test: run the trained model on every 5th frame of all sample videos. Count frames with any detection at confidence ≥ 0.5 and save those frames to `zlt-share/jalol/hazard_fp/`.
4. Find the lowest confidence threshold where the false-fire count on sample videos is 0.
5. Write the module `contrib/jalol/hazard/hazard_detector.py`:

```python
class HazardDetector:
    def __init__(self, weights_path: str, conf: float, device: str = "cuda"): ...
    def predict(self, frame_bgr) -> list[dict]:
        # returns [{"label": "fire" or "smoke", "xyxy": [x1, y1, x2, y2], "score": float}, ...]
        # pixel coordinates, deterministic, no network access
```

Output: the module, `contrib/jalol/hazard/report.md` (dataset, licence, training settings, val mAP, false-fire count on samples at each threshold, chosen threshold), weights to `zlt-share/jalol/hazard/best.pt`.

### J6. Detector fine-tune (wait for Iko: "go J6")

Only if Iko posts `go J6`. He decides from your J3 results.

Label 300 to 500 sample frames in CVAT or Label Studio with classes person, bicycle, motorcycle, car, bus, truck, YOLO export. Split train/val by whole video, never frames of the same video in both. Fine-tune the chosen checkpoint with `fliplr=0` (flipping reverses lane directions). Report per-class recall on val before and after in `contrib/jalol/det_ft.md`. Weights to the drive.

### J7. Bio and QA (bio anytime; QA wait for Iko: "QA time")

- Bio: `contrib/jalol/bio.md` with full name, role, what you did on this project, GitHub, LinkedIn, portfolio links, 1 to 3 previous projects with one line each, a photo in the drive.
- QA: on `QA time`, fresh clone of the tag Iko names into a new folder, follow only the README, run the two official commands. Write every command and every error into `contrib/jalol/qa_report.md`. Do not fix anything. Then open the website on your phone, upload a short video in the demo, and report what broke.

## 6. JAVOHIR

Do these in order. First write your GPU model and VRAM in `env.md` (setup step 9).

### V1. Camera map draft

Goal: a first version of the scene map Iko's rules use.

- Input: `camera.md` and one clear frame from any sample (confirm all samples have the same view; note it if not).

Steps:

1. Write `contrib/javohir/tools/click_points.py`: shows a frame, left-click adds a point, n finishes the current shape and asks for its name in the terminal, s saves all shapes. Coordinates saved as fractions of width and height (0 to 1), 4 decimals.
2. Draw: road outline; each lane as a polygon; each stop line (2 points); each pedestrian crossing polygon; each solid line (polyline); the intersection area; the traffic light box if visible.
3. Lane direction: watch normal traffic and set an arrow `[dx, dy]` per lane from real movement, not guessed.
4. Allowed movements: list which lane can go to which exit, from `camera.md` and road markings. Mark unknown ones as unknown.

Output: `contrib/javohir/camera_draft.yaml` in this structure, plus `zlt-share/javohir/camera_overlay.png` with everything drawn:

```yaml
frame: {width: 1920, height: 1080}
coords: normalized
road: [[x, y], ...]
lanes:
  - {id: north_1, polygon: [[x, y], ...], direction: [dx, dy], movements: [straight, right]}
stop_lines:
  - {id: north_stop, points: [[x, y], [x, y]], lanes: [north_1]}
crosswalks:
  - {id: cross_1, polygon: [[x, y], ...]}
solid_lines:
  - {id: solid_1, polyline: [[x, y], ...]}
intersection: [[x, y], ...]
signal: {visible: true, roi: [x1, y1, x2, y2], controls: [north_1]}
allowed_movements:
  - {from: north_1, to: east_exit}
unknown_movements:
  - {from: north_1, to: west_exit, note: "no sign visible"}
```

### V2. Annotate your videos

Same as J1, for your videos from 4.5. Output in `contrib/javohir/events/`.

### V3. Review Jalol's videos

Same as J2, reversed. Output in `contrib/javohir/review/`.

### V4. Traffic light state (only if camera.md says the signal is visible)

If the light is not visible: write "signal not visible, V4 skipped" in `QUESTIONS.md` and go to V5.

Steps:

1. Using the signal box from V1, crop that region from every frame of every sample. Keep crops in the drive.
2. Label the state as time ranges per video (the light changes rarely): `contrib/javohir/signal/signal_states.json` `{"sample_01.mp4": [[0.00, 31.20, "red"], [31.20, 34.20, "yellow"], [34.20, 70.00, "green"]]}`
3. Build the classifier with HSV color thresholds per lamp first. Only if HSV is below 98% accuracy on any video, fine-tune mobilenet_v3_small on the crops (split by video).
4. Glare, occlusion or unclear frames must return "unknown", never a guess.
5. Module `contrib/javohir/signal/signal_classifier.py`:

```python
def classify_signal(crop_bgr) -> str:
    # returns "red", "yellow", "green" or "unknown"; deterministic; no network
```

Output: module, labels, `contrib/javohir/signal/report.md` with accuracy per video and a confusion matrix. Target: at least 98% on every video separately.

### V5. Dataset licence table and accident data index

Goal: the README must list every dataset with its licence, and Iko may use extra accident data later.

Steps:

1. For each dataset: CADP, DoTA, CCD, DAD, UA-DETRAC, BDD100K. Find the official page. Read the actual licence or terms text (not a blog, not the code licence of a GitHub repo, which can differ from the data licence).
2. Fill `contrib/javohir/DATASETS.md`:

| dataset | official URL | data licence (exact name) | commercial use allowed? | registration needed? | camera type (CCTV or dashcam) | size | notes |
|---|---|---|---|---|---|---|---|

3. For CADP only (fixed CCTV, closest to our camera): if its licence allows use, download it to the drive and write `contrib/javohir/cadp_index.csv` with columns `clip_id, file, duration_s, fps, accident_start_s, accident_end_s, notes`, filling times from the dataset's own annotations. If a time is not provided by the dataset, leave it empty. Do not guess.

### V6. Near-miss / accident classifier (wait for Iko: "features ready")

Starts when Iko posts `features ready` and puts `windows.parquet` on the drive.

The file has one row per candidate window: `window_id, video, t_start, t_end, label` (accident, near_miss or none) and feature columns whose names start with `f_`.

Steps:

1. Train XGBoost (`random_state=0`, `n_jobs=1` for determinism) using only `f_` columns. Class weights for imbalance.
2. Validation: leave-one-video-out. Never put windows from the same video in train and validation.
3. Report per fold: precision, recall, F1 per class, and the threshold with the best F1.
4. Module `contrib/javohir/nm/nm_classifier.py`:

```python
class NMClassifier:
    def __init__(self, model_path: str): ...
    def predict_proba(self, df) -> "np.ndarray":
        # df has the same f_ columns; returns shape (n_rows, 3) for [none, near_miss, accident]
```

5. Save the model with `model.save_model("nm_xgb.json")` into the drive.

Output: module, `contrib/javohir/nm/train.py`, `contrib/javohir/nm/report.md`.

### V7. Bio and QA

Same as J7, with your own paths.

## 7. Checklist before posting "delivered"

- File is at the exact path, in the exact format.
- Only files inside `contrib/<your_name>/` are committed; no videos, frames or weights in Git.
- Every number was measured; unmeasured things say "not measured".
- Open questions are in `QUESTIONS.md`, and you applied the default.
