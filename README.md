# Zeroth Law Traffic

Traffic event detection and causal accident anticipation for the WIUT Hackathon 2026 Computer Vision track.

## Current status

The official starter kit is integrated unchanged. Its zero-event, zero-risk baseline passed format validation on the available sample. The solution is still that baseline: no trained perception or event engine is active.

Development infrastructure includes typed data contracts, a detector/tracker cache interface, an evaluator for saved predictions, exact-frame review sheets, and geometry-overlay tooling. Camera verification and dev-label adjudication are in progress. No detection accuracy or anticipation performance is claimed yet.

## Install and run

Python 3.10+; development currently uses Python 3.11.

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python run_submission.py --videos data/samples --out outputs/predictions.json --team zeroth-law
python evaluate.py --pred outputs/predictions.json --validate-only
```

Videos, weights, caches and generated outputs are excluded from Git. The current baseline has no weights to download. `run_submission.py` and `evaluate.py` must remain byte-identical to the organizer kit.

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
