# Zeroth Law Traffic

Traffic event detection and anticipation from video. This repository is the team workspace for a hackathon submission and, later, a reusable traffic analytics project.

## Status

**Scaffold only (2026-09-24).** The organizer starter kit, sample videos, exact output schema, scoring equation, constraints, and deadline have not been supplied here. No current file is claimed to be a valid submission or to earn a score. Add the official starter kit unchanged in its first commit, then adapt this scaffold to its contract.

## Intended pipeline

```text
video -> detections -> tracks -> scene state -> event/risk logic
      -> official predictions -> official evaluator -> submission
```

Cache detections and tracks under ignored `cache/` so event thresholds can be evaluated without rerunning perception. Keep evidence-rich internal events; strip them to the official schema only at the final adapter.

## Repository layout

- `solution.py`: provisional zero-event, zero-risk interface; signatures must be reconciled with the starter kit.
- `run_submission.py`: entry point reserved for the official runner contract.
- `configs/`: camera geometry, event, and risk settings; sample values are intentionally absent.
- `src/`: perception, scene, features, events, anticipation, and postprocessing modules.
- `scripts/`: video inspection, rendering, benchmarking, development evaluation, and submission verification.
- `annotations/`: human labels and camera maps, once samples arrive.
- `experiments/`: experiment log template and results.
- `web/`: visualization consuming the same rich pipeline output.
- `docs/`: decisions and handoffs.

## Check the scaffold

```bash
python -m compileall -q solution.py run_submission.py src
python run_submission.py --help
```

After the starter kit arrives, the first milestone is a valid zero-score baseline: connect `solution.py` and `run_submission.py` to the **official** interface, run the official evaluator in validation mode, and commit the resulting working baseline. The commands and output shape must come from that kit; do not assume the example commands in the planning note are authoritative.

## Team workflow

`main` must remain runnable. Use short branches such as `feat/tracking`, `feat/annotation-tooling`, `feat/risk`, and `feat/website`; run the relevant checks before merging. Keep experimental video, model weights, caches, and generated output out of Git.

Suggested ownership: lead owns architecture, integration, scoring, and final packaging; one teammate owns annotations and camera geometry; another owns the website and visualization. Assign actual people and checkpoints once their GitHub handles and availability are known.

## Next information needed

Official starter kit and evaluator; task constraints, scoring equation, and deadline; sample videos and license; teammate GitHub handles. Record each in `docs/handoff-2026-09-24.md` as it becomes available.
