# Team website

Vite + React, static build. The site consumes the same rich event, trajectory, and risk output as the pipeline; it never invents data. Anything not backed by pipeline output is marked as a placeholder on the page.

## Data

`public/data/` is generated, and committed so Vercel can build from Git. Regenerate it, never edit it by hand:

One command refreshes every sample that has a perception cache (derived tables if missing, event and risk replays with this checkout's code, export) and fails if a sample in `data/samples` is left out:

```bash
python scripts/refresh_web_data.py                                      # in the repo that holds the caches
python scripts/refresh_web_data.py --source-root ../zeroth-law-traffic  # from a worktree
```

It calls `export_web.py` per video, which can also be run alone:

```bash
python scripts/export_web.py --video data/samples/C3905.MP4 --cache cache/C3905_yolo11m960_s3 \
    --events outputs/web_replay/events/C3905.json --risk outputs/web_replay/risk/C3905.json
```

Inputs are read-only: the perception cache (`tracks_stitched`, `world`, `pairs`, `signal_series`), the `replay_events.py --all` and `replay_risk.py` outputs, `configs/camera.yaml`, `configs/atlas.json`, `configs/events.yaml`, and one decode of the video for the lighting series and `frame.jpg`. From a worktree without caches, add `--source-root ../zeroth-law-traffic`. `--reuse-video` skips the decode and keeps the previous frame and lighting.

Per video (`public/data/<id>/`): `replay.json` (tracks, every 2nd cached sample), `events.json`, `risk.json`, `field.json` (all approaching pairs, safe ones included), `signal.json` (raw readings and the smoothed timeline Part A uses), `scene.json`, `atlas.json`, `eda.json`, `frame.jpg`.

## Run

```bash
npm install
npm run dev        # or: npm run build && npm run preview
```

## Screenshots and interaction checks

With `npm run preview` running:

```bash
node scripts/shoot.mjs http://localhost:4173/ shots/latest
node scripts/interact.mjs http://localhost:4173/ shots/interact
```

Set `PW_CHROMIUM` to a Chromium executable to reuse an existing browser instead of `npx playwright install`.

## What is drawn from where

- Road, sidewalks, median, refuges, crossings, stop line, signal head and lamp cells: `configs/camera.yaml`.
- Lane dashes: straight lines between the camera.yaml carriageway edges.
- Trees, buildings, bus stop, gantry, lamp post, signs: traced by hand on frame 0 (`src/scene/scenery.js`), decoration only.
- Road users: every replayed track at its tracked foot point and box size. Class comes from the tracker; body and clothing colours are seeded per track id and are not measured.

## Deploy (Vercel)

Project root directory `web`; `vercel.json` sets the Vite framework, `npm ci`, `npm run build` and output `dist`. The committed `public/data` ships with the build.

## Demo backend slot

`src/api/client.js` targets `VITE_API_BASE` (FastAPI on a Hugging Face CPU Space, not built yet). Without it the demo section reports the backend as offline.
