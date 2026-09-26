# Submission runtime integration

## Execution and isolation

`src/models_registry.py` requires `weights/yolo11m.pt` at an explicit repository-relative location. It reads that checkpoint once when imported and prepares two inference backends, one for each part, before the harness's per-video timer. It then exercises three synthetic frames through each real input shape, including preprocessing, forward and NMS, at import. No video is read during import. Ultralytics online checks and automatic package installation are disabled. Missing weights fail clearly. CUDA device 0 uses FP16; otherwise inference uses CPU FP32. Dependencies include the headless Ultralytics distribution to avoid a second OpenCV installation.

Each consumer gets its own predictor state and each video gets a fresh tracker. The two parts have separate inference backends. ByteTrack's global ID allocator is isolated during tracker operations. Its clock advances at the base stride of three, including empty updates for guard-skipped observations, so the two-second lost-track buffer does not accidentally become minutes long. Large strides still lose continuity; this is a quality cost, not a tracking solution.

Part A CPU decoding uses the FFmpeg binary bundled in the pinned imageio-ffmpeg wheel, eight decoder and filter threads, `select=not(mod(n\,3))`, then scaling to width 1920 with bicubic/bt601/centered horizontal chroma. Source indices are 0,3,6,... and times are index/fps. Boxes are restored to native dimensions before tracking and scene rules. Signals use scaled lamp cells at every decoded sample, including inference-skipped samples. Native-resolution equivalence with OpenCV does **not** imply scaled-pixel equivalence.

The OpenCV fallback sequentially grabs every source frame and retrieves selected frames. It resumes after the last successfully emitted FFmpeg index, without duplicating observations. It uses INTER_AREA, so the fallback has a different preprocessing signature. Failure and transition are logged; it may be slower. H.264 interframes still require decoding.

Part A invokes `src.features.stitching.prepare_tracks(tracks, width=width, height=height)` when that module exists, expecting `(tracks, info)`. Coordinates and dimensions are native pixels; rich `info` is logged. Absence is a logged pass-through; errors in a present hook propagate. The hook is not present on this branch's base and belongs to Claude's main commit 893e89a. The world model runs after stitching, followed by `run_engines(..., only_enabled=True)`, which already performs segment postprocessing, then `to_official`. No sample-specific answers or perception cache reads occur in submission inference.

Part B only consumes the current BGR frame and metadata supplied by the harness. It maintains independent causal tracks, starts at stride three, and returns its last score immediately for skipped frames. The score is deliberately 0.0 until risk logic is integrated.

## Budget and reproducibility limits

Part A has 1.25 times video duration; Part B has 1.55 times duration. The remaining 0.20 times duration is a combined overhead/uncertainty margin. Each part observes its own elapsed wall time and a rolling mean of the last five processed frames, reserves predicted remaining CPU decode time, and widens stride in multiples of three when necessary. It waits for five observations before changing stride and **never exceeds stride 15** in either part. Every widening logs its reason, rolling cost, requested stride and applied stride; hitting the cap logs the explicit quality-over-budget decision. Final part summaries are written as JSON to stderr. The initial decode priors (0.85x A, 1.40x B) come from local laptop experiments, **not** the judges' CPU. They are revised upward from observed non-inference wall time. The 7.7 GB laptop suffers heavy paging; local wall times are indicative only.

This guard cannot guarantee 3x on arbitrary machines: the harness must decode every frame in B, and A must decode interframes even when sampling. Mandatory decode, a single slow call, fallback restarts, or expensive event logic can exhaust the budget. CPU-only functional execution is not evidence of quality-preserving T4 performance. Profile both detector paths, host decode and full harness on a real T4 before acceptance.

Adaptive inference indices depend on wall-clock timing; enabled event predictions can therefore differ between runs. Current all-disabled events and zero risk can give identical prediction payloads without establishing general reproducibility. The official harness also writes run-specific timing logs into `predictions.json`: compare the `team` and `videos` payload separately and report full-file differences honestly. No timing data is stripped from the original harness outputs.

Event context retains the base stride three, with actual source timestamps on observations; terminal segment extension remains the base step. Sparse tracks after guard widening need event-quality review. No thresholds are changed to conceal preprocessing or sampling differences.

## Read-only comparison against an existing cache

```powershell
python scripts/validate_runtime_decode.py --video 'C:\Users\imama\Projects\zeroth-law-traffic\data\samples\C3905.MP4' --cache ../zeroth-law-traffic/cache/C3905_yolo11m960_s3 --out private/integration/comparison
```

The script checks the exact source string, size, mtime, checkpoint digest and perception config before comparison. It runs real inference at 32 fixed, uniformly spaced base-stride indices; reads signal colours at all base-stride indices; and writes counts, same-class one-to-one box matches (IoU >=0.5), matched IoU, small-box cache-match recall and signal disagreements. Small/far means cache box height <=3% of native height. This is agreement with a prior detector, **not recall against human labels**. Neither the old cache nor its metadata is changed. Adding the registry changes the cache builder's fingerprint, so do not invoke that builder on the existing cache to perform this comparison.

## Remaining integration gates

- Measure FP16 YOLO11m 960 with `scripts/t4_profile.py`, and the full harness with `scripts/t4_end_to_end.py` on a single T4, with its eight-core CPU and clean dependencies. The detector-only profiler is copied unchanged from `eng/runtime`; its prescaled input is OpenCV INTER_AREA, whereas the end-to-end script exercises the actual FFmpeg path. No branch merge is implied.
- Merge Claude's completed stitching/event/risk work separately and repeat enabled-event quality and runtime checks.
- Include the local checkpoint in the offline submission archive and validate installation on the actual judge OS/Python/CUDA combination. The development environment is not a clean judge installation.

## Kaggle full harness

Use a fresh subprocess after installing dependencies and placing the checkpoint inside the repository. This script pins one visible GPU, verifies it is a T4, runs two full unchanged-harness passes at the official 3x limit, validates both outputs, and saves all stdout/stderr plus per-video ratios and equality checks. It rejects an existing output directory. It also reports whole-process elapsed time, which includes import/warm-up, separately from the official per-video time. Empty outputs caused by timeout are treated as failures.

Check `nvidia-smi` before installing. The default Linux PyPI resolution selects CUDA 13, whose driver compatibility starts at R580 ([NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)). For a CUDA 12 driver environment, PyTorch publishes the same pinned versions as CUDA 12.6 wheels ([torch](https://download.pytorch.org/whl/cu126/torch/), [torchvision](https://download.pytorch.org/whl/cu126/torchvision/)). An explicit pre-install avoids silently retaining a CPU-only wheel:

```bash
python -m pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Run installation with internet before the offline evaluation. These GPU installation commands are documented from official wheel availability, not locally GPU-tested; verify driver compatibility and the full run on the actual host.

```python
!python /kaggle/working/zeroth-law-traffic/scripts/t4_end_to_end.py --videos /kaggle/input/traffic --out /kaggle/working/zlt-full-t4
```

Local CPU functional checks may use the harness's explicit `--time-factor 30` override to finish both capped-stride passes. This does not change the solution's 1.25x/1.55x plan, and is **not** a 3x acceptance test. Never use that override for the T4 budget gate.
