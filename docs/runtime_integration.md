# Submission runtime integration

## Execution and isolation

`src/models_registry.py` requires `weights/yolo11m.pt` at an explicit repository-relative location. It reads that checkpoint once when imported and prepares two inference backends, one for each part, before the harness's per-video timer. It then exercises three synthetic frames through each real input shape, including preprocessing, forward and NMS, at import. No video is read during import. Ultralytics online checks and automatic package installation are disabled. Missing weights fail clearly. CUDA device 0 uses FP16; otherwise inference uses CPU FP32. Dependencies include the headless Ultralytics distribution to avoid a second OpenCV installation.

Each consumer gets its own predictor state and each video gets a fresh tracker. The two parts have separate inference backends. ByteTrack's global ID allocator is isolated during tracker operations. It updates only on inference observations and rescales its time buffer, elapsed track clocks and Kalman velocity/covariance when the observed stride changes. Unconfirmed new objects therefore survive skipped inference frames. Sparse observations still reduce motion/association accuracy.

Part A CPU decoding uses OpenCV by default, exactly as the cache builder reads video: it grabs every source frame, retrieves every third, and resizes frames wider than 1920 to width 1920 and height `round(h * 1920 / w)` with INTER_AREA (the detector's prescale). Runtime detections therefore see the same pixels as the caches. This replaced FFmpeg as the default after a Colab T4 run on C3905 took 349.5 s in Part A, while the harness's OpenCV decode of every frame in Part B finished within 44.2 s. Source indices are 0,3,6,... and times are index/fps. Boxes are restored to native dimensions before tracking and scene rules. Signals use scaled lamp cells at every decoded sample, including inference-skipped samples. H.264 interframes still require decoding.

`ZLT_DECODE=ffmpeg` selects the FFmpeg binary bundled in the pinned imageio-ffmpeg wheel: eight decoder and filter threads, `select=not(mod(n\,3))`, then scaling to width 1920 (even height) with bicubic/bt601/centered horizontal chroma. Its pixels differ from the caches. On failure it falls back to OpenCV from the last emitted index, without duplicating observations, and logs the transition. Any other `ZLT_DECODE` value is an error.

Part A invokes `src.features.stitching.prepare_tracks(tracks, width=width, height=height)` when that module exists, expecting `(tracks, info)`. Coordinates and dimensions are native pixels; rich `info` is logged. Absence is a logged pass-through; errors in a present hook propagate. The stitching hook is included in the integrated submission. The world model runs after stitching, followed by `run_engines(..., only_enabled=True)`, which already performs segment postprocessing, then `to_official`. No sample-specific answers or perception cache reads occur in submission inference.

Part B only consumes the current BGR frame and metadata supplied by the harness. It maintains independent causal tracks, starts at stride three, and returns its last score immediately for skipped frames. It feeds the analytic causal risk model with lifetime class votes, current-frame rider exclusion and a one-second last-known signal hold. Risk is no longer a zero placeholder.

## Budget and reproducibility limits

Part A has 1.25 times video duration; Part B has 1.55 times duration. The remaining 0.20 times duration is a combined overhead/uncertainty margin. Each part observes its own elapsed wall time and a rolling mean of the last five processed frames, reserves predicted remaining CPU decode time, and widens stride in multiples of three when necessary. It waits for five observations before changing stride and **never exceeds stride 15** in either part. Every widening logs its reason, rolling cost, requested stride and applied stride; hitting the cap logs the explicit quality-over-budget decision. Final part summaries are written as JSON to stderr. The initial decode priors (0.85x A, 1.40x B) come from local laptop experiments, **not** the judges' CPU. They are revised upward from observed non-inference wall time. The 7.7 GB laptop suffers heavy paging; local wall times are indicative only.

This guard cannot guarantee 3x on arbitrary machines: the harness must decode every frame in B, and A must decode interframes even when sampling. Mandatory decode, a single slow call, fallback restarts, or expensive event logic can exhaust the budget. CPU-only functional execution is not evidence of quality-preserving T4 performance. Profile both detector paths, host decode and full harness on a real T4 before acceptance.

Adaptive inference indices depend on wall-clock timing; enabled event predictions can therefore differ between runs. The model registry seeds Python/NumPy/PyTorch to zero and requests deterministic algorithms, but this does not eliminate guard timing sensitivity. The official harness also writes run-specific timing logs into `predictions.json`: compare the `team` and `videos` payload separately and report full-file differences honestly. No timing data is stripped from the original harness outputs.

Event context retains the base stride three, with actual source timestamps on observations; terminal segment extension remains the base step. Sparse tracks after guard widening need event-quality review. No thresholds are changed to conceal preprocessing or sampling differences.

## Read-only comparison against an existing cache

```powershell
python scripts/validate_runtime_decode.py --video 'C:\Users\imama\Projects\zeroth-law-traffic\data\samples\C3905.MP4' --cache ../zeroth-law-traffic/cache/C3905_yolo11m960_s3 --out private/integration/comparison
```

The script checks the exact source string, size, mtime, checkpoint digest and perception config before comparison. It runs real inference at 32 fixed, uniformly spaced base-stride indices; reads signal colours at all base-stride indices; and writes counts, same-class one-to-one box matches (IoU >=0.5), matched IoU, small-box cache-match recall and signal disagreements. Small/far means cache box height <=3% of native height. This is agreement with a prior detector, **not recall against human labels**. Neither the old cache nor its metadata is changed. Adding the registry changes the cache builder's fingerprint, so do not invoke that builder on the existing cache to perform this comparison.

## Remaining integration gates

- Measure FP16 YOLO11m 960 with `scripts/t4_profile.py`, and the full harness with `scripts/t4_end_to_end.py` on a single T4, with its eight-core CPU and clean dependencies. The detector-only profiler is copied unchanged from `eng/runtime`; its prescaled input is OpenCV INTER_AREA, whereas the end-to-end script exercises the actual FFmpeg path. No branch merge is implied.
- Validate each enabled event on locked labels; stitching and causal risk are already integrated.
- Include the local checkpoint in the offline submission archive and validate installation on the actual judge OS/Python/CUDA combination. The development environment is not a clean judge installation.

## Kaggle full harness

Use a fresh subprocess after installing dependencies and placing the checkpoint inside the repository. This script pins one visible GPU, verifies it is a T4, runs two full unchanged-harness passes at the official 3x limit, validates both outputs, and saves all stdout/stderr plus per-video ratios and equality checks. It rejects an existing output directory. It also reports whole-process elapsed time, which includes import/warm-up, separately from the official per-video time. Empty outputs caused by timeout are treated as failures.

Check `nvidia-smi` before installing. requirements.txt now explicitly pins torch 2.10.0+cu126 and torchvision 0.25.0+cu126. Install with `python -m pip install -r requirements.txt`; see docs/linux_cuda_install.md for the Linux hash lock and driver limits. Do not preinstall the old CUDA 13/default-wheel versions.

Run installation with internet before the offline evaluation. These GPU installation commands are documented from official wheel availability, not locally GPU-tested; verify driver compatibility and the full run on the actual host.

```python
!python /kaggle/working/zeroth-law-traffic/scripts/t4_end_to_end.py --videos /kaggle/input/traffic --out /kaggle/working/zlt-full-t4
```

Local CPU functional checks may use the harness's explicit `--time-factor 30` override to finish both capped-stride passes. This does not change the solution's 1.25x/1.55x plan, and is **not** a 3x acceptance test. Never use that override for the T4 budget gate.
