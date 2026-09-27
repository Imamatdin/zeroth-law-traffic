# Zeroth Law Traffic

Traffic event detection and causal accident anticipation for the WIUT Hackathon 2026 Computer Vision track.

## Current status

The official starter kit remains unchanged. `solution.py` runs local YOLO11m perception, ByteTrack, an optional Part A stitching hook, the world model, and the enabled event engines with segment postprocessing. The checked-in event configuration currently disables every engine. Part B runs its own detector and tracker and feeds a causal interaction-risk model (`src/anticipation/risk.py`) with its own tracks and the signal read from each frame; it never uses Part A output. No detection accuracy or anticipation performance is claimed.

Development infrastructure includes typed data contracts, a detector/tracker cache interface, an evaluator for saved predictions, exact-frame review sheets, and geometry-overlay tooling. Camera verification and dev-label adjudication are in progress. No detection accuracy or anticipation performance is claimed yet.

## Install and run

Python 3.11 is the tested development version. Install the pinned dependencies, then place the provided YOLO11m checkpoint at `weights/yolo11m.pt` before importing the solution. Model loading never downloads weights. Its SHA256 must be `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`. The submission archive must include this file; it is not stored in Git.

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python run_submission.py --videos data/samples --out outputs/predictions.json --team zeroth-law
python evaluate.py --pred outputs/predictions.json --validate-only
```

Videos, weights, caches and generated outputs are excluded from Git. `imageio-ffmpeg` supplies the CPU FFmpeg binary through its platform wheel; OpenCV is the fallback. `run_submission.py` and `evaluate.py` must remain byte-identical to the organizer kit. See `docs/runtime_integration.md` for budget assumptions, preprocessing differences, validation and remaining judge-machine checks.

For development tools and checks:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

See `scripts/README.md` for review, cache and evaluation commands. `--pred` in the dev evaluator scores an existing file without rerunning inference; `--videos` explicitly runs the official harness. Organizer example files are test fixtures, not labels for the sample videos.

## Architecture

```text
video -> detections -> tracks -> scene state -> event engines
      -> rich evidence -> segment postprocessing -> official predictions

streamed past/current frames -> independent causal state -> risk(t)
```

The proposed system combines scene rules, trajectory/interaction features, and a flow atlas of normal traffic. These model components are planned, not implemented performance claims. Observed movement frequency does not establish legal permission.

The harness runs Part A before Part B and applies one shared per-video time budget. Part B never reads the video file, future frames, or Part A output. Model weights may be shared for inference; trackers and other temporal state must remain independent.

The organizer rules are in `docs/task_spec.md`; module and data contracts are in `docs/ARCHITECTURE.md`; teammate deliverables are specified in `docs/TEAM_BRIEF.md`.

## Team workflow

Iko owns the main workflow, architecture, integration, website and submission. Jalol and Javohir contribute through the separate lanes defined in the team brief; these responsibilities are assignments, not claims of completed contributions. Use short branches and keep `main` runnable. Teammate work stays under `contrib/<name>/` on `team/<name>` until integrated.

Before submission, add the selected model and dataset licences, training/inference settings and seeds, weights with checksums, measured runtime, reproducibility evidence, team contributions, and final `predictions_samples.json`. The website and final submission package remain pending.
