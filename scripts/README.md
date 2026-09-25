# Development scripts

Run these commands from the repository root using the project's Python environment.

## Exact-frame review

```bash
python scripts/review_frames.py --video data/samples/C3905.MP4 --out data/inspection/C3905/window --start-frame 300 --end-frame 330 --stride 3
```

Sheets carry actual frame indices and `frame / fps` timestamps. Identical requests reuse the completed manifest without decoding again. A changed source or request requires a new output directory. Review sheets contain real frames; they are not ground-truth annotations.

## Development evaluation

```bash
python scripts/dev_eval.py --pred outputs/predictions.json --gt annotations/dev_ground_truth.json --config configs/events.yaml
python scripts/dev_eval.py --videos data/samples --gt annotations/dev_ground_truth.json
```

`--pred` scores existing output without running inference. Only `--videos` runs the unchanged organizer harness. Both call the unchanged evaluator, display its per-class report and write a numbered experiment containing scores, predictions, hashes, Git state and recorded runtime/errors. Missing labels fail before inference starts. Harness failures are rejected unless explicitly included for diagnosis with `--allow-run-errors`.

The real dev ground truth has not been locked yet. Organizer `examples/` are format fixtures, not sample labels or evidence of model accuracy. Cached-rule replay will be connected when real perception and event engines exist.

## Perception cache interface

Install `requirements-dev.txt` for Parquet support. `build_cache.py` accepts a Stage 3 factory under `src/perception/`, specified as `module:function`; it receives `(video_meta, config)` and returns a `Detector` and a fresh `Tracker`. No model factory has been selected or implemented yet.

Supply `--video`, `--out cache/<video_id>`, `--factory`, `--config` (JSON parameters), every `--weights` file, and optional `--stride`. The factory must remap detector classes into the six internal classes in `src/contracts.py`; tracker timestamps must be `frame_idx / meta['fps']`.

The cache contains `meta.json`, `frames.parquet`, `detections.parquet`, and `tracks.parquet`. `frames` records inference frames even when no objects were detected. Matching caches are reused before model construction or video decoding. Source size/mtime/path, perception parameters (including precision/device), weight hashes, perception/contract/model-registry code, package versions and stride determine reuse. Event rule changes do not invalidate perception caches. A stale cache fails with a request for a new output directory. A failed build remains in a partial directory and cannot be mistaken for a completed cache.

## Geometry overlay

`render_camera_map.py --frame <image> --map <draft.json> --out <image>` draws normalized inspection shapes. It does not establish legal movements or turn the draft into an inference configuration.
