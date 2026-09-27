// Loads the JSON written by scripts/export_web.py and decodes it into structures the scene can query per frame.
export const REF_W = 1920;
export const REF_H = 1080;

const base = `${import.meta.env.BASE_URL}data`;

async function getJson(path) {
  const r = await fetch(`${base}/${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

export async function loadIndex() {
  return getJson('index.json');
}

export async function loadVideo(id) {
  const [replay, events, risk, signal, scene, atlas, field, eda] = await Promise.all(
    ['replay', 'events', 'risk', 'signal', 'scene', 'atlas', 'field', 'eda'].map((f) => getJson(`${id}/${f}.json`)),
  );
  return decode({ id, replay, events, risk, signal, scene, atlas, field, eda });
}

// Binary search: last index with arr[i] <= x (or -1).
export function floorIndex(arr, x) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= x) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

function decodeTracks(replay) {
  const T = replay.t;
  const q = replay.q;
  return replay.tracks.map((tr) => {
    const n = tr.k.length;
    const ts = new Float64Array(n), x = new Float32Array(n), y = new Float32Array(n);
    const w = new Float32Array(n), h = new Float32Array(n), v = new Float32Array(n), hd = new Float32Array(n);
    const st = new Float32Array(n), face = new Int8Array(n), walk = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      ts[i] = T[tr.k[i]];
      x[i] = (tr.x[i] / q) * REF_W;
      y[i] = (tr.y[i] / q) * REF_H;
      w[i] = (tr.w[i] / q) * REF_W;
      h[i] = (tr.h[i] / q) * REF_H;
      v[i] = tr.v[i] / 100;
      hd[i] = tr.hd[i];
      st[i] = tr.st[i] / 10;
    }
    // Facing (left/right) from net horizontal travel over about a second; held while standing still, and
    // back-filled before the first decisive movement.
    let last = 0;
    let firstDecided = -1;
    for (let i = 0; i < n; i++) {
      let j0 = i, j1 = i;
      while (j0 > 0 && ts[i] - ts[j0 - 1] <= 0.6) j0--;
      while (j1 < n - 1 && ts[j1 + 1] - ts[i] <= 0.6) j1++;
      const dx = x[j1] - x[j0];
      if (Math.abs(dx) > 0.12 * h[i]) {
        last = dx > 0 ? 1 : -1;
        if (firstDecided < 0) firstDecided = i;
      }
      face[i] = last;
    }
    const initial = firstDecided < 0 ? 1 : face[firstDecided];
    for (let i = 0; i < n && face[i] === 0; i++) face[i] = initial;
    for (let i = 1; i < n; i++) walk[i] = walk[i - 1] + Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / Math.max(h[i], 1);
    return { id: tr.id, cls: tr.cls, raw: tr.raw, t0: ts[0], t1: ts[n - 1], ts, x, y, w, h, v, hd, st, face, walk };
  });
}

// Pose of a track at time t, linearly interpolated; null outside its lifetime.
export function poseAt(tr, t) {
  if (t < tr.t0 || t > tr.t1) return null;
  const i = floorIndex(tr.ts, t);
  const j = Math.min(i + 1, tr.ts.length - 1);
  const u = j === i ? 0 : (t - tr.ts[i]) / (tr.ts[j] - tr.ts[i]);
  const L = (a) => a[i] + (a[j] - a[i]) * u;
  let dh = tr.hd[j] - tr.hd[i];
  if (dh > 180) dh -= 360;
  if (dh < -180) dh += 360;
  return {
    x: L(tr.x), y: L(tr.y), w: L(tr.w), h: L(tr.h), v: L(tr.v), hd: tr.hd[i] + dh * u,
    st: u < 0.5 ? tr.st[i] : tr.st[j], face: tr.face[u < 0.5 ? i : j], walk: L(tr.walk), i,
  };
}

function decodeField(field, T, q) {
  const n = field.k.length;
  const byK = new Map();
  for (let i = 0; i < n; i++) {
    const k = field.k[i];
    if (!byK.has(k)) byK.set(k, []);
    byK.get(k).push({
      a: field.a[i], b: field.b[i],
      x: (field.px[i] / q) * REF_W, y: (field.py[i] / q) * REF_H,
      tca: field.tca[i] / 10, dmin: field.dmin[i] / 100,
    });
  }
  const ks = [...byK.keys()].sort((a, b) => a - b);
  return { byK, ks, kt: ks.map((k) => T[k]), filter: field.filter, count: ks.map((k) => byK.get(k).length) };
}

function decode(d) {
  const { replay } = d;
  const tracks = decodeTracks(replay);
  const byId = new Map(tracks.map((t) => [t.id, t]));
  const q = replay.q;
  const risk = {
    ...d.risk,
    pairs: d.risk.pairs.map((p) => ({ ...p, t: d.risk.t[p.i], x: (p.px / q) * REF_W, y: (p.py / q) * REF_H })),
  };
  const events = d.events.raw_events.map((e, i) => ({ ...e, key: `${e.label}-${i}` }));
  const labels = [...new Set(events.map((e) => e.label))];
  const sig = Object.entries(d.signal.signals)[0];
  return {
    id: d.id, replay, tracks, byId, events, labels, eventsMeta: d.events, risk,
    signalId: sig?.[0], signal: sig?.[1], crossings: d.signal.near_stop_crossings,
    camera: d.scene.camera, atlas: d.atlas, field: decodeField(d.field, replay.t, q), eda: d.eda,
    duration: replay.duration, T: replay.t, frameUrl: `${base}/${d.id}/frame.jpg`,
  };
}

export function stateAt(runs, t) {
  for (const [a, b, s] of runs) if (t >= a && t < b) return s;
  return runs.length ? runs[runs.length - 1][2] : null;
}
