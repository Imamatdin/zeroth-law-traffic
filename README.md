# Zeroth Law Traffic

Traffic events (Part A) and causal accident anticipation (Part B) for the WIUT Hackathon 2026 CV track. The [organizer specification](docs/task_spec.md) is authoritative.

**Website:** https://zeroth-law-traffic.vercel.app · **Weights:** [weights-v1 release](https://github.com/Imamatdin/zeroth-law-traffic/releases/tag/weights-v1) · **Licence:** [AGPL-3.0](LICENSE)

**Status:** Part A enables seven classes in `configs/events.yaml`: jaywalking, failure_to_yield, red_light, stop_line, stopped_vehicle, wrong_way and near_miss. All 16 non-accident detections they and solid_line_crossing produced on C3905 were confirmed correct by human review. Off: accident, solid_line_crossing (its only C3905 detection is a 0.1 s segment that cannot match the official boundaries), congestion, illegal_turn, illegal_u_turn, road_obstacle and fire_smoke. This validation covers one sample video, not general accuracy. Part B produces analytic risk. Accuracy on real accidents is unverified. The full pipeline fits the 3x budget on an 8-core GPU pod; on a Colab T4 it went over budget before the decode fix, and the rerun after the fix is pending (see [Runtime evidence](#runtime-evidence)). Final sample predictions must be generated from the final tag on a GPU machine.

## Install and official commands

Target: Linux x86_64, Python 3.11, single T4, 8 CPU cores, 32 GB RAM. Install Bash, curl and sha256sum. Internet is required during installation and weight retrieval; inference uses local files only.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash weights/download.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0
python run_submission.py --videos /data/test --out predictions.json --team zeroth-law
python evaluate.py --pred predictions.json --validate-only
```

For a fully resolved Linux/Python 3.11 environment, replace the pip line with `python -m pip install --require-hashes -r requirements-linux-cu126.lock`. Torch 2.10.0+cu126 and torchvision 0.25.0+cu126 target CUDA 12.6; headless Ultralytics is 8.4.163. See [driver limits](docs/linux_cuda_install.md). CPU FP32 is supported but is not promised to meet the 3x budget.

To score annotated footage: `python evaluate.py --pred predictions.json --gt /path/to/ground_truth.json`. Organizer examples are format fixtures, not sample-video labels. Official files are unchanged. Format validation can pass after a timed-out harness run; also use `python scripts/check_harness_output.py --pred predictions.json --videos /data/test` to reject errors and incomplete risk output.

## Weights

Only `weights/yolo11m.pt` is shipped: **40,684,120 bytes**, below 5 GB. This is the Ultralytics COCO-pretrained YOLO11m, with no team fine-tuning. SHA256:

```text
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
```

Published in the [weights-v1 release](https://github.com/Imamatdin/zeroth-law-traffic/releases/tag/weights-v1) ([publication steps](docs/weights_release.md)). On 2026-09-27, `bash weights/download.sh` in an empty folder downloaded it from that release and passed the SHA256 check; a second run reported it already verified. `download.sh` uses curl, verifies SHA256 before atomic installation, and skips an already verified file. No inference-time downloads. RT-DETR-L is an unused local challenger and is not shipped. Camera and atlas JSON files are tracked configuration/prior data, not extra neural checkpoints.

## Approach

```text
A: CPU decode/sample -> YOLO11m -> ByteTrack -> rider fusion/stitching
   -> scene geometry + kinematics + atlas -> enabled event rules
   -> segment merging/clamping -> official events
B: current harness frame -> independent detector/tracker
   -> causal class vote + rider exclusion + signal hold
   -> conflict features -> analytic risk -> smoother/calibrator
```

- **Learned:** COCO-pretrained YOLO11m, selecting person, bicycle, motorcycle, car, bus and truck. No team-trained detector or accident network.
- **Tracking:** Ultralytics ByteTrack association and Kalman filtering; inference-frame updates, stride-adjusted buffer, independent A/B IDs/state.
- **Rules:** manual camera geometry, signal lamp colours, trajectory features, event state machines and segment postprocessing. The sample-fitted flow atlas is an empirical prior, not proof of legal movement.
- **Risk:** closest approach, closing speed, braking, red-light/wrong-way modifiers; rider duplicates excluded. Class votes use history so far; unknown signals hold the last known state for at most one second. Calibration is configured using synthetic scenarios, not fitted to labelled real accidents. B never reads a video, future frames or A outputs.
- **Decode:** Part A decodes on the CPU with OpenCV (grab every frame, retrieve every third, INTER_AREA to width 1920, as the cache builder does) before 960-input inference. Native indices/timestamps/coordinates are restored. `ZLT_DECODE=ffmpeg` selects the bundled imageio-ffmpeg pipe instead. The samples' H.264 10-bit 4:2:2 is not decoded on T4 hardware.

See [architecture](docs/ARCHITECTURE.md), [runtime details](docs/runtime_integration.md), [data](DATASETS.md), and [licences](LICENSES.md). No custom neural training was performed; no custom training recipe is claimed.

## Seeds and nondeterminism

The registry seeds Python, NumPy and PyTorch to **0** before model construction, disables cuDNN benchmarking, enables deterministic cuDNN and requests deterministic PyTorch algorithms with warnings for unsupported operations. Scripts set `PYTHONHASHSEED=0`; CUBLAS workspace configuration defaults to `:4096:8`.

Unsupported CUDA operations, hardware/library changes, FP16 rounding and equal-score NMS may vary. The wall-clock runtime guard can select different strides (3 through 15) under different loads, changing predictions even with fixed seeds. Final same-machine reproducibility needs the T4 run. Official JSON contains variable timing logs; compare prediction payloads separately without changing the submitted harness output.

## Runtime evidence

Official harness, unchanged, on C3905 (127.63 s, 4K 10-bit 4:2:2), 3x budget 382.9 s. Ratio is total harness seconds over video duration.

| Run | Hardware | Part A | Part B | Total | Ratio | Result |
|---|---|---|---|---|---|---|
| All event classes off | GPU pod (RTX PRO 4500 Blackwell), `taskset` to 8 cores | 53.1 s | 67.6 s | 120.8 s | 0.95x | Within budget, format valid |
| Non-visual classes on (incl. accident) | Same pod, 8 cores | 102.4 s | 67.2 s | 169.5 s | 1.33x | Within budget, format valid |
| Before the decode fix (FFmpeg pipe in Part A) | Colab T4 | 349.5 s | 44.2 s | 393.7 s | 3.08x | **Over budget**: scored as empty |
| After the decode fix (OpenCV in Part A) | Colab T4 | – | – | – | – | Pending |

The T4 failure came from Part A's FFmpeg decode: the harness's own OpenCV decode of every 4K frame in Part B took under 44.2 s. Part A now decodes with OpenCV exactly as the cache builder does; `ZLT_DECODE=ffmpeg` restores the old path. The pod is not the judges' T4, so its ratios show headroom, not compliance. Budget split: A 1.25x, B 1.55x, 0.20x margin; the guard widens inference stride (3 to 15) when a part runs slow. Import/warm-up is outside the per-video timer. Reproduce with `python scripts/t4_end_to_end.py --videos /path/to/videos --out outputs/t4-final`; never relax the time factor for acceptance.

## Packaging and final predictions

```bash
# Linux, fresh venv, install, download, first 20 seconds, official 3x budget:
bash scripts/clean_install_test.sh /path/to/C3905.MP4
# Checkout FINAL_TAG first, on the GPU machine with all samples:
bash scripts/make_predictions_samples.sh FINAL_TAG /path/to/all-samples
```

Sample generation requires C3896.MP4, C3897.MP4 and C3905.MP4, includes additional videos in that folder, validates and rejects harness errors before writing `predictions_samples.json`. It refuses to overwrite existing predictions. Publish that generated JSON with the final tag; it is not generated by this packaging task. Fresh Linux/CUDA acceptance remains unrun here. Clean-install logs and environment versions stay in `outputs-clean-install.*`.

Development: `python -m pip install -r requirements-dev.txt`, then `python -B -m unittest discover -s tests -v`.

## Team

| Member | GitHub | Contribution |
|---|---|---|
| Iko | [@Imamatdin](https://github.com/Imamatdin) | Architecture, Part A/B pipeline, integration, website |
| Jalol | – | Perception caches, T4 testing, labels |
| Javohir | – | Labels |

See [team brief](docs/TEAM_BRIEF.md). AI tools assisted development; no hosted inference API is used. Website: https://zeroth-law-traffic.vercel.app
