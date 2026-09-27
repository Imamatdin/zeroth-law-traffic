# CPU live demo

This backend uses the same detector adapter, independent ByteTrack trackers,
Part A stitching, world features, enabled event registry/postprocessing, and
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
`weights/yolo11m.pt` instead of downloading again. The server never downloads
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
4096 x 2160 pixels (either orientation), 1â€“120 fps. Server checks container
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
`risk`, then `done` or `error`. `progress` is a monotonic 0â€“1 work fraction, not
a time estimate. Decode and inference are streamed together during detecting;
risk includes its own second CPU decode/inference pass. Poll every 1â€“2 seconds.
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

The existing web `client.js` slot currently expects a result inline in job
status with simple arrays. Update it to fetch `result_url` on `done`, then pass
`bundle.replay`, `bundle.events`, `bundle.risk`, `bundle.signal` to the existing
export consumers. For a simple result list use `bundle.events.submitted`.

```js
const form = new FormData(); form.append('video', file);
const upload = await fetch(`${base}/jobs`, {method: 'POST', body: form});
if (!upload.ok) throw new Error(JSON.stringify(await upload.json()));
const {job_id} = await upload.json();
for (;;) {
  const response = await fetch(`${base}/jobs/${job_id}`);
  if (!response.ok) throw new Error(`Status HTTP ${response.status}`);
  const job = await response.json();
  onProgress(job);
  if (job.state === 'error') throw new Error(job.error.message);
  if (job.state === 'done') {
    const response = await fetch(`${base}${job.result_url}`);
    if (!response.ok) throw new Error(`Result HTTP ${response.status}`);
    return await response.json();
  }
  await new Promise(resolve => setTimeout(resolve, 1500));
}
```

Errors use `detail: {code,message}` for application errors. Standard malformed
multipart/schema requests may use FastAPI's validation error shape. Codes/status:
409 `busy`/`not_ready`, 413 `too_large`, 415 `wrong_format`, 422 `too_long`/
`invalid_video`/`resolution_limit`/`decode_failed`/`processing_failed`,
404 `not_found` (including expiry), 408 `upload_timeout`.

## Profile and camera check

`cpu-yolo11m-640-s9`: YOLO11m, imgsz 640, stride 9 in both passes, CPU FP32,
two PyTorch/FFmpeg threads, CPU decode scaled to width 960, native coordinates
restored. Confidence/IoU/tracker settings come from `configs/perception.json`.
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
`https://github.com/Imamatdin/zeroth-law-traffic/releases/download/weights-v1/yolo11m.pt`
at build time, checking SHA256
`d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
The GitHub repo/release asset must be publicly readable by the Space builder.
Missing assets or checksum mismatches fail the build with a clear error.

Deployment prerequisites checked on 2026-09-27: HF_TOKEN is unset here and the public weights-v1 release returns HTTP 404. No Space has been deployed. Publish the release asset before the following steps.

Exact deployment steps once the release is available (PowerShell):

```powershell
python -m pip install huggingface-hub==2.0.0
$env:HF_TOKEN = Read-Host 'Hugging Face write token' -MaskInput
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

Benchmark includes CPU decode + detection in both passes, events, risk and
export conversion. Model preparation is measured separately. See
`demo/VALIDATION.md` for measured local results and deployment status.
