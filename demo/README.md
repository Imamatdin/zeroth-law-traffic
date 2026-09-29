# CPU live demo

This backend uses the same detector adapter, a ByteTrack tracker,
offline stitching, world features, enabled event registry/postprocessing, and
causal RiskModel as `solution.py`. It does not import the submission model
registry or change the judging profile. The source base includes `integration-3`, including causal class votes and signal holding.

## Local run

From the repository root, with Python 3.11:

```sh
python -m venv .venv-demo
# Activate .venv-demo using your shell's activation command.
python -m pip install -r demo/requirements.txt
python demo/download_weights.py
uvicorn demo.app:app --host 127.0.0.1 --port 7860 --workers 1
```

If you already have the verified repository checkpoint, place it at
`weights/yolo11n.pt` instead of downloading again. The server never downloads
weights. Open http://localhost:7860/docs or GET `/health`.

**Exactly one worker/process/replica.** Reservation begins before multipart
parsing; a second upload receives 409 rather than entering a memory-heavy queue.
Model setup happens once at startup. Uploads are deleted after success/failure;
results expire after one hour, after 20 retained results, or on server restart.
The store is temporary, not durable. Deploying multiple replicas requires a
shared job coordinator, which this single CPU Space does not use.

## API contract for the frontend

All URLs below are relative to `VITE_API_BASE` (no trailing slash).

### POST /jobs

Multipart field **`video`**, one `.mp4` file. Max **120 seconds**, **3 GiB**,
4096 x 2160 pixels (either orientation), 1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“120 fps. Server checks container
signature, metadata and first-frame decode; the full passes also reject corrupt
or truncated media. Network stalls during upload time out after 30 seconds.

Response **202**:

```json
{"job_id":"opaque-id","status_url":"/jobs/opaque-id","result_url":"/jobs/opaque-id/result"}
```

### GET /jobs/{id}

```json
{"job_id":"opaque-id","state":"running","stage":"detecting","progress":0.25,"created":1790000000,"duration":20.05}
```

`state`: `running | done | error`. Stages: `decoding`, `detecting`, `events`,
`risk`, then `done` or `error`. `progress` is a monotonic 0ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“1 work fraction, not
a time estimate. Decode and inference are streamed together during detecting;
risk is evaluated causally from the shared detections, before offline stitching. Poll every 1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2 seconds.
Completion adds `result_url`, `metadata`, and `finished`. Failure adds
`error: {code, message}`. GET `/health` reports `ready`, `busy`, and limits.

### GET /jobs/{id}/result

Response **200**, JSON envelope:

```text
{
  metadata: {profile, scene_check, scene_events_skipped,
             processing_seconds, wall_time_ratio, warnings},
  replay:   <exact replay.json document shape>,
  events:   <exact events.json document shape>,
  risk:     <exact risk.json document shape>,
  signal:   <exact signal.json document shape>,
  detections: {columns: [frame,t,cls,score,x1,y1,x2,y2],
               coords: "native pixels", rows: [...]}
}
```

The four document shapes match `zlt-web/web/public/data/C3905/` and the web
exporter. Extensions are outside those documents. `replay.tracks` holds parallel
arrays `k,x,y,w,h,v,hd,st`, plus `id,cls,raw`; `k` indexes `replay.t`.
Coordinates/sizes are normalized integers with `q=10000`; positions are foot
points. Speed is box heights/second x100; stationary duration is seconds x10.
`events.segments` and `events.raw_events` contain label/start/end/confidence/
track_ids/evidence; `events.submitted` contains official triples.
`risk` contains aligned `t,risk,raw,smoothed` arrays and evidence `pairs`.
`signal.signals` contains raw and smoothed run-length timelines and bridged gaps.
`near_stop_crossings` is empty: that extra offline diagnostic is not computed.
No field/EDA/atlas document or uploaded video is served by this API.

The web client polls status, fetches `result_url` when done, and renders local
video playback with normalized track boxes, event seeking and the risk curve.
Unmatched scenes never receive reference-camera overlays or event rules.

Errors use `detail: {code,message}` for application errors. Standard malformed
multipart/schema requests may use FastAPI's validation error shape. Codes/status:
409 `busy`/`not_ready`, 413 `too_large`, 415 `wrong_format`, 422 `too_long`/
`invalid_video`/`resolution_limit`/`decode_failed`/`processing_failed`,
404 `not_found` (including expiry), 408 `upload_timeout`.

## Profile and camera check

`cpu-yolo11n-640-720p-8fps-shared`: YOLO11n, imgsz 640, CPU FP32,
two PyTorch/FFmpeg threads, CPU decode bounded to 1280 x 720, native coordinates
restored. The stride is round(source_fps / 8), at least 1: 29.97 fps footage is
analyzed at 7.49 fps, while 25 fps footage is analyzed at 8.33 fps. There is one
shared decode/detection/tracking pass. Risk receives raw observations causally
before the offline event analysis and track stitching. This demo profile is not
the official Part B interface; the frozen submission remains untouched.
`demo/events.yaml` is the exact reviewed config from submission-final. Confidence/IoU/tracker settings come from `configs/perception.json`.
The image-size/stride reduction changes accuracy; it is reported in every result.
No adaptive runtime guard or judging-time cutoff is applied to the demo.

The scene gate requires 3840 x 2160, 16:9 and blurred-edge correlation >=0.55
against `demo/assets/reference.jpg` (the C3905 web-export reference frame).
The threshold is a conservative engineering heuristic, not a calibrated
classifier. Different lighting/moving traffic can cause false mismatches.
1080p uploads intentionally fail the exact-resolution gate even if rescaled
from this camera. On mismatch, event engines, signals and the camera atlas are
disabled. Detection/tracking still run; risk uses generic image-space geometry
with the image treated as a possible road area. Such risk is uncalibrated and
must be shown with `metadata.warnings`. Do not draw C3905 lane overlays on an
unmatched upload. All event classes currently disabled by configuration stay
disabled; the demo never silently forces development engines on.

## Hugging Face CPU Space

Build from repo root: `docker build -f demo/Dockerfile -t zlt-demo .`
Run: `docker run --rm -p 7860:7860 zlt-demo`.
The image listens on 7860, runs as UID 1000, and downloads **only**
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
at build time, checking SHA256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
The pinned Ultralytics release asset must be publicly readable by the Space builder.
Missing assets or checksum mismatches fail the build with a clear error.

On 2026-09-28, HF returned HTTP 402 when creating a Docker CPU Space on a
free account. A PRO subscription is now required for this hosting tier. No
Space is claimed deployed until its build and live health check succeed.

Exact deployment steps once the release is available (PowerShell):

```powershell
python -m pip install huggingface-hub==2.0.0
# Set HF_TOKEN privately, or save it with the Hugging Face CLI login.
# demo.deploy also reads ~/.cache/huggingface/token.
python -m demo.deploy --space YOUR_ACCOUNT/zeroth-law-traffic-demo
Remove-Item Env:HF_TOKEN
```

The helper uses huggingface_hub to create a Docker Space and uploads only
src/configs/demo plus solution.py/evaluate.py. It sets the Space root Dockerfile
and README frontmatter automatically; it never uploads local weights or caches.
Wait for the Space build to become Running; test its `/health` and `/docs`.

Local CORS defaults: `http://localhost:5173`, `http://localhost:4173`.
Once the Vercel origin is known, add a Space variable `DEMO_CORS_ORIGINS` with
the exact origin (comma separated for more than one), or redeploy with
`--origin https://YOUR_SITE.vercel.app`. No wildcard/credentials are enabled.
Set the frontend `VITE_API_BASE` to the Space's `https://...hf.space` host and
rebuild the Vercel site. CORS controls browser access, not authentication.

## Verification

```sh
python -m pip install httpx==0.28.1
python -B -m unittest discover -s demo/tests -v
python -B -m unittest discover -s tests -p test_official_kit.py -v
python -m demo.benchmark PATH_TO_4K_CLIP PATH_TO_1080P_CLIP --out private/demo-timing
```

Benchmark includes the shared CPU decode + detection pass, events, risk and
export conversion. Model preparation is measured separately. See
`demo/VALIDATION.md` for measured local results and deployment status.

## Railway deployment (2026-09-28)

Live API: https://zeroth-law-traffic-production.up.railway.app

Railway uses `demo/Dockerfile`; its start command reads `PORT` (default 7860).
Set `RAILWAY_DOCKERFILE_PATH=demo/Dockerfile` and
`DEMO_CORS_ORIGINS=https://zeroth-law-traffic.vercel.app` on the service.
Use `demo/railway.json` as its config file, or copy that file to `railway.json`
in a staged deployment root. Deploy only src/, configs/, demo/, third_party/,
solution.py and evaluate.py; no local caches, uploads or credentials.
Expose the assigned HTTP port and check `/health` before uploading.

This deployment was tested on Railway's trial with two real clips. C3905 was
re-encoded at its native 3840x2160 resolution solely to reduce upload size.
The foreign clip is the first five seconds of OpenCV's samples/data/vtest.avi.
Source: https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi
Neither test establishes 60-second-clip memory or runtime behavior.

| Clip | Duration | Processing | Tracks | Detections | Camera match |
|---|---:|---:|---:|---:|---|
| C3905 excerpt | 4.938 s | 5.692 s | 36 | 1347 | yes |
| OpenCV foreign scene | 5.000 s | 5.577 s | 9 | 428 | no |

The trial remains resource/credit limited. Model output is demonstration output,
not the frozen submission's results. Weights are the pinned upstream YOLO11n
asset documented above; weights-v1 and submission-final have not been changed.
