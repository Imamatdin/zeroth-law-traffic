# Demo validation — 2026-09-27

Base: integration-3 (`7f29a61`), merged into `eng/demo` with merge commit
`5901aa7`. No submission files were edited for the demo.

## Checks actually run

```text
python -B -m unittest discover -s demo/tests -v
Ran 8 tests in 23.926s — OK

python -B -m unittest discover -s tests -p test_official_kit.py -v
Ran 1 test in 0.003s — OK

uv pip compile demo/requirements.txt --python-version 3.11 --python-platform x86_64-manylinux_2_28 --only-binary :all: --index-strategy unsafe-best-match --output-file private/demo-linux-resolved.txt
Resolved 54 packages in 12.87s
```

Tests cover upload rejection, exclusive job reservation, progress/results, cleanup,
sanitized processing failures, CORS, result retention, scene mismatch, and generic
detection/tracking/risk output. The web schema fixture was taken from the current
C3905 replay/events/risk/signal exports, including nested track and risk-pair
fields. This is a structural check, not a browser integration test. Official
harness/evaluator hashes pass. The complete submission test suite was not rerun
for these isolated demo additions.

## Measured CPU runs

Command (run sequentially, without caches):

```text
python -B -m demo.benchmark C:/Users/imama/Projects/zeroth-law-traffic/data/samples/clip20.mp4 private/clip20_1080p.mp4 --out private/benchmark-final
```

Windows build 26200, Intel i5-1335U (10 physical cores, 12 logical processors),
7.7 GB usable RAM. Profile: YOLO11m FP32, input 640, stride 9, decode width 960,
two CPU threads. Both Part A and independent causal Part B are included.

| Input | Duration | Processing | Ratio | Tracks | Detections | Scene gate |
|---|---:|---:|---:|---:|---:|---|
| 3840×2160 | 20.053 s | 161.082 s | 8.033× | 61 | 2,608 | Match, similarity 0.9360 |
| 1920×1080 | 20.120 s | 118.233 s | 5.876× | 58 | 2,627 | Resolution mismatch, similarity 0.9694 |

Engine construction/model warm-up took 22.134 s separately; this excludes Python
module import time. Processing includes CPU decoding, both inference passes,
events, risk and export conversion. Upload/network time is excluded. The 1080p
input is a rescaled clip from the same camera; the explicit exact-resolution
gate intentionally disables its scene-dependent events and signals. Generic risk
and detection/tracking remain available, with an uncalibrated-risk warning.

**Indicative local measurements only:** concurrent work and heavy paging were
observed. These numbers do not establish Hugging Face CPU Space performance or
the competition runtime limit. Local runtime used Python 3.11, torch 2.14.0+cpu,
torchvision 0.29.0+cpu, ultralytics 8.4.163 and OpenCV 5.0.0.93. The deployment
requirements instead pin torch 2.10.0+cpu / torchvision 0.25.0+cpu and the
Ultralytics headless distribution 8.4.163. Linux wheel resolution succeeded;
an actual clean Linux installation and container execution remain unverified.

## Deployment status

Not deployed: HF_TOKEN was unset and the public weights-v1 release endpoint
returned HTTP 404. Docker build and hosted smoke tests were not run. The Docker
build deliberately requires the release checkpoint and verifies its SHA256.
Publish that asset, then follow the exact account/origin/token steps in
[README.md](README.md). No token, local cache, sample video, or private output is
included in the deployment upload allowlist.

Full local logs/results are retained under `private/`: demo-tests-complete.log,
official-hashes-final.log, dependency-resolution-final.log,
demo-linux-resolved.txt, benchmark-final.log, and benchmark-final/.
