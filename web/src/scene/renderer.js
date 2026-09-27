// Draws one frame of the miniature: static ground, depth-sorted props and road users, and the analytical
// layers (trails, interaction field, risk contours, flow atlas, event notes). Owns the camera.
import { REF_W, REF_H, poseAt, floorIndex, stateAt } from '../data/load.js';
import { palette, eventColor, FONTS } from '../design/tokens.js';
import { buildProps, renderStatic, grainTile } from './staticScene.js';
import { drawActor, headlight, looks } from './actors.js';
import { SCENERY } from './scenery.js';
import { inkLine, ellipsePts, handText, withAlpha, clamp, rng, blob } from './ink.js';

const MAX_FOLLOW_Z = 1.45;
const LAMP_COLOUR = { red: 'sig-red', yellow: 'sig-yellow', green: 'sig-green' };
const CLASS_NAMES = ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck'];

export function eventCaption(e, t) {
  switch (e.label) {
    case 'stopped_vehicle': return t >= e.start ? `stopped for ${(Math.min(t, e.end) - e.start).toFixed(0)} s` : 'about to stop';
    case 'jaywalking': return 'on the carriageway';
    case 'stop_line': return 'stopped past the line on red';
    case 'red_light': return 'crossed on red';
    default: return e.label.replace(/_/g, ' ');
  }
}

export class Diorama {
  constructor(canvas, data, { reducedMotion = false } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = data;
    this.reduced = reducedMotion;
    this.pal = palette();
    this.props = buildProps(data.camera, this.pal);
    this.static = null;
    this.view = { cx: REF_W / 2, cy: REF_H / 2, z: 1 };
    this.target = { ...this.view };
    this.override = false;
    this.focus = null;
    this.boxes = [];
    this.vw = 1;
    this.vh = 1;
    this.dpr = 1;
    this.lastZoomChange = 0;
    this.dusk = (data.eda?.lighting?.mean_luma ?? []).length
      ? data.eda.lighting.mean_luma.reduce((a, b) => a + b, 0) / data.eda.lighting.mean_luma.length < 70 : false;
    this.photo = null;
    this.grain = this.ctx.createPattern(grainTile({ ink: this.pal.ink }), 'repeat');
  }

  loadPhoto() {
    if (this.photo) return;
    const img = new Image();
    img.src = this.data.frameUrl;
    img.onload = () => { this.photo = img; };
  }

  resize(w, h, dpr) {
    this.vw = w;
    this.vh = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.clampView(this.view);
    this.clampView(this.target);
    this.ensureStatic(true);
  }

  get labelScale() {
    return clamp(this.vw / 1000, 0.72, 1);
  }

  get s0() {
    return Math.max(this.vw / REF_W, this.vh / REF_H);
  }

  home() {
    const portrait = this.vw / this.vh < 1.2;
    return { cx: portrait ? REF_W * 0.42 : REF_W / 2, cy: portrait ? REF_H * 0.5 : REF_H / 2, z: 1 };
  }

  clampView(v) {
    v.z = clamp(v.z, 1, 3);
    const k = this.s0 * v.z;
    const hw = this.vw / (2 * k), hh = this.vh / (2 * k);
    v.cx = clamp(v.cx, Math.min(hw, REF_W / 2), Math.max(REF_W - hw, REF_W / 2));
    v.cy = clamp(v.cy, Math.min(hh, REF_H / 2), Math.max(REF_H - hh, REF_H / 2));
    return v;
  }

  // The ground is rendered once, sharp up to the follow camera's zoom; a manual zoom past that re-renders
  // it once the view has settled.
  ensureStatic(force = false) {
    const need = (z) => clamp(Math.ceil(this.s0 * z * this.dpr * 4) / 4, 0.5, 2);
    const have = this.static?.scale ?? 0;
    if (force) {
      const want = need(Math.max(this.view.z, MAX_FOLLOW_Z));
      if (want > have * 1.05 || want < have / 1.6) this.static = renderStatic(this.data.camera, this.pal, this.props, want);
    } else if (need(this.view.z) > have * 1.25) {
      this.static = renderStatic(this.data.camera, this.pal, this.props, need(this.view.z));
    }
  }

  // ---------- camera ----------
  panBy(dx, dy) {
    const k = this.s0 * this.view.z;
    this.override = true;
    this.view.cx -= dx / k;
    this.view.cy -= dy / k;
    this.clampView(this.view);
    this.target = { ...this.view };
  }

  zoomBy(f, sx = this.vw / 2, sy = this.vh / 2) {
    const k0 = this.s0 * this.view.z;
    const wx = this.view.cx + (sx - this.vw / 2) / k0, wy = this.view.cy + (sy - this.vh / 2) / k0;
    this.override = true;
    this.view.z = clamp(this.view.z * f, 1, 3);
    const k1 = this.s0 * this.view.z;
    this.view.cx = wx - (sx - this.vw / 2) / k1;
    this.view.cy = wy - (sy - this.vh / 2) / k1;
    this.clampView(this.view);
    this.target = { ...this.view };
    this.lastZoomChange = performance.now();
  }

  resetView() {
    this.override = false;
    this.focus = null;
    this.target = this.clampView(this.home());
  }

  focusEvent(e) {
    this.override = false;
    this.focus = e;
  }

  get maxFollowZ() {
    return this.vw / this.vh < 1.2 ? 1.15 : MAX_FOLLOW_Z;
  }

  frameBox(poses, pad = 5) {
    const xs = poses.flatMap((p) => [p.x - (p.w ?? 0) / 2, p.x + (p.w ?? 0) / 2]);
    const ys = poses.flatMap((p) => [p.y - (p.h ?? 0), p.y]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const bw = Math.max(x1 - x0, 90), bh = Math.max(y1 - y0, 60);
    const z = clamp(Math.min(this.vw / (bw * pad), this.vh / (bh * pad)) / this.s0, 1, this.maxFollowZ);
    return this.clampView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, z });
  }

  followTarget(t, ui) {
    if (this.override) return null;
    const d = this.data;
    if (this.focus) {
      const e = this.focus;
      if (t > e.end + 1.5 || t < e.start - 4) this.focus = null;
      else {
        const poses = e.track_ids.map((id) => d.byId.get(id)).filter(Boolean).map((tr) => poseAt(tr, clamp(t, tr.t0, tr.t1)));
        if (poses.length) return this.frameBox(poses);
      }
    }
    if (ui.selected != null) {
      const tr = d.byId.get(ui.selected);
      const p = tr && poseAt(tr, t);
      if (p) return this.frameBox([p], 7);
    }
    if (!ui.layers.follow) return null;
    if (ui.layers.risk) {
      const now = this.riskAt(t);
      if (now?.pair && now.smoothed >= 0.05) {
        const q = now.pair;
        const poses = [q.a, q.b].map((id) => (id != null ? d.byId.get(id) : null)).map((tr) => tr && poseAt(tr, t)).filter(Boolean);
        return this.frameBox([...poses, { x: q.x, y: q.y, w: 160, h: 120 }], 4);
      }
    }
    const active = d.events
      .filter((e) => e.end - e.start <= 20 && t >= e.start - 1 && t <= Math.min(e.end, e.start + 10))
      .sort((a, b) => a.start - b.start);
    for (const e of active) {
      const poses = e.track_ids.map((id) => d.byId.get(id)).map((tr) => tr && poseAt(tr, t)).filter(Boolean);
      if (poses.length) return this.frameBox(poses, 6);
    }
    return null;
  }

  // ---------- frame ----------
  frame(t, dt, ui) {
    const { ctx, pal, data: d } = this;
    const ft = this.followTarget(t, ui);
    if (ft) this.target = ft;
    else if (!this.override) this.target = this.clampView(this.home());
    const a = this.reduced ? 1 : 1 - Math.exp(-dt * 2.4);
    const prevZ = this.view.z;
    for (const key of ['cx', 'cy', 'z']) this.view[key] += (this.target[key] - this.view[key]) * a;
    this.clampView(this.view);
    if (Math.abs(this.view.z - prevZ) > 1e-3) this.lastZoomChange = performance.now();
    else if (performance.now() - this.lastZoomChange > 250) this.ensureStatic();

    const k = this.s0 * this.view.z;
    const ox = this.vw / 2 - this.view.cx * k, oy = this.vh / 2 - this.view.cy * k;
    const dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = pal.paper;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.static) {
      const s = this.static.scale;
      ctx.setTransform((dpr * k) / s, 0, 0, (dpr * k) / s, dpr * ox, dpr * oy);
      ctx.drawImage(this.static.ground, 0, 0);
    }
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * ox, dpr * oy);
    const px = 1 / k;
    const boil = this.reduced ? 0 : Math.floor(t * 4);

    if (ui.layers.photo && this.photo) {
      ctx.globalAlpha = 0.7;
      ctx.drawImage(this.photo, 0, 0, REF_W, REF_H);
      ctx.globalAlpha = 1;
    }

    const actors = [];
    for (const tr of d.tracks) {
      const p = poseAt(tr, t);
      if (p) actors.push({ tr, p });
    }
    const poseOf = new Map(actors.map((x) => [x.tr.id, x.p]));

    if (ui.layers.atlas) this.drawAtlas(px);
    if (ui.layers.field) this.drawField(t, poseOf, px);
    if (ui.layers.trails) this.drawTrails(t, actors, px);
    if (this.dusk) for (const { tr, p } of actors) if (tr.cls >= 2 && p.v > 0.3) headlight(ctx, p, pal);

    const layout = ui.layers.notes ? this.layoutNotes(t, poseOf, px, k, ox, oy) : { notes: [], boxes: [] };
    this.drawPlaces(layout.boxes, px, k, ox, oy);

    const items = [
      ...actors.map((x) => ({ y: x.p.y, actor: x })),
      ...this.props.map((pr, i) => ({ y: pr.baseY, prop: i })),
    ].sort((u, v) => u.y - v.y);
    const signalState = d.signal ? stateAt(d.signal.raw, t) : null;
    this.boxes = [];
    for (const it of items) {
      if (it.actor) {
        const { tr, p } = it.actor;
        drawActor(ctx, tr, p, pal, { lw: 1.05 * px, amp: 0.35 * px * 2, seed: tr.id * 101 + boil });
        this.boxes.push({ id: tr.id, x0: ox + (p.x - p.w / 2) * k, y0: oy + (p.y - p.h) * k, x1: ox + (p.x + p.w / 2) * k, y1: oy + p.y * k, fx: ox + p.x * k, fy: oy + p.y * k });
      } else if (this.static) {
        const pr = this.props[it.prop];
        const [x0, y0, x1, y1] = pr.bbox;
        ctx.drawImage(this.static.sprites[it.prop], x0, y0, x1 - x0, y1 - y0);
        if (pr.lamps && signalState && LAMP_COLOUR[signalState]) {
          const [lx, ly, lr] = pr.lamps[signalState];
          const cx = (pr.bbox[0] + pr.bbox[2]) / 2;
          const col = pal[LAMP_COLOUR[signalState]];
          const g = ctx.createRadialGradient(cx, ly, 0, cx, ly, lr * 4);
          g.addColorStop(0, withAlpha(col, 0.55));
          g.addColorStop(1, withAlpha(col, 0));
          ctx.fillStyle = g;
          ctx.fillRect(cx - lr * 4, ly - lr * 4, lr * 8, lr * 8);
          blob(ctx, ellipsePts(cx, ly, lr, lr, 12), { fill: col, stroke: pal.ink, width: px, seed: 3, amp: 0.2 });
        }
      }
    }

    if (ui.selected != null) this.drawSelection(t, ui.selected, poseOf, px);
    if (ui.layers.risk) this.drawRisk(t, poseOf, px);
    if (ui.layers.notes) this.drawNotes(layout, px);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = this.grain;
    ctx.fillRect(0, 0, this.vw, this.vh);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    return { actors, k };
  }

  drawTrails(t, actors, px) {
    const { ctx, pal } = this;
    for (const { tr, p } of actors) {
      if (p.v < 0.15 && p.st > 3) continue;
      const i1 = p.i;
      const i0 = Math.max(0, floorIndex(tr.ts, t - 4));
      if (i1 - i0 < 2) continue;
      const col = tr.cls === 0 ? looks(tr).coat : pal['ink-soft'];
      for (let i = i0; i <= i1; i++) {
        const age = (t - tr.ts[i]) / 4;
        ctx.globalAlpha = clamp(0.55 * (1 - age), 0, 0.55);
        ctx.fillStyle = col;
        const r = (tr.cls === 0 ? 1.6 : 2.2) * px * (1 + (tr.h[i] / 120));
        ctx.beginPath();
        ctx.arc(tr.x[i], tr.y[i] - 2 * px, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  drawField(t, poseOf, px) {
    const { ctx, pal } = this;
    const f = this.data.field;
    const j = floorIndex(f.kt, t);
    if (j < 0) return;
    const j1 = Math.min(j + 1, f.ks.length - 1);
    const u = j1 === j ? 0 : clamp((t - f.kt[j]) / (f.kt[j1] - f.kt[j]), 0, 1);
    const pairs = new Map();
    for (const q of f.byK.get(f.ks[j])) pairs.set(`${q.a}-${q.b}`, { p0: q, p1: null });
    for (const q of f.byK.get(f.ks[j1])) {
      const key = `${q.a}-${q.b}`;
      if (pairs.has(key)) pairs.get(key).p1 = q;
      else pairs.set(key, { p0: null, p1: q });
    }
    ctx.save();
    ctx.lineCap = 'round';
    for (const { p0, p1 } of pairs.values()) {
      const w = p0 && p1 ? 1 : p0 ? 1 - u : u;
      const q = p0 && p1 ? { ...p0, x: p0.x + (p1.x - p0.x) * u, y: p0.y + (p1.y - p0.y) * u, tca: p0.tca + (p1.tca - p0.tca) * u, dmin: p0.dmin + (p1.dmin - p0.dmin) * u } : p0 ?? p1;
      // display mapping: nearer predicted passes and sooner closest approaches draw stronger
      const I = Math.exp(-q.dmin) * Math.max(0, 1 - q.tca / 5);
      const alpha = w * (0.1 + 0.45 * I);
      if (alpha < 0.03) continue;
      const pa = poseOf.get(q.a), pb = poseOf.get(q.b);
      const scale = pa && pb ? (pa.h + pb.h) / 2 : 50;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = pal.field;
      ctx.lineWidth = px * 1.1;
      ctx.setLineDash([px * 2, px * 5]);
      for (const pp of [pa, pb]) {
        if (!pp) continue;
        ctx.beginPath();
        ctx.moveTo(pp.x, pp.y - pp.h * 0.3);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      const rx = scale * (0.35 + 0.45 * I);
      ctx.beginPath();
      ctx.ellipse(q.x, q.y, rx, rx * 0.42, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  riskAt(t) {
    const r = this.data.risk;
    const k = floorIndex(r.t, t);
    if (k < 0) return null;
    let pair = null;
    for (let i = r.pairs.length - 1; i >= 0; i--) {
      if (r.pairs[i].t <= t) {
        if (t - r.pairs[i].t <= 6) pair = r.pairs[i];
        break;
      }
    }
    return { k, risk: r.risk[k], smoothed: r.smoothed[k], raw: r.raw[k], pair };
  }

  drawRisk(t, poseOf, px) {
    const { ctx, pal } = this;
    const now = this.riskAt(t);
    if (!now?.pair || now.smoothed < 0.015) return;
    const { pair, smoothed } = now;
    const pa = pair.a != null ? poseOf.get(pair.a) : null, pb = pair.b != null ? poseOf.get(pair.b) : null;
    const base = pa && pb ? (pa.h + pb.h) * 0.9 : 90;
    const n = clamp(Math.ceil(smoothed / 0.015), 1, 14);
    const lift = base * 0.09;
    for (let i = 0; i < n; i++) {
      const u = i / n;
      const rx = base * (1.25 - u * 1.05);
      const pts = ellipsePts(pair.x, pair.y - i * lift, rx, rx * 0.42, 28);
      inkLine(ctx, pts, { color: pal.risk, width: px * (i === n - 1 ? 2 : 1.2), seed: 900 + i, amp: rx * 0.04, closed: true, alpha: clamp(0.25 + 0.6 * u, 0.25, 0.9), double: false });
    }
    ctx.save();
    ctx.setLineDash([px * 4, px * 4]);
    ctx.strokeStyle = withAlpha(pal.risk, 0.7);
    ctx.lineWidth = px * 1.3;
    for (const pp of [pa, pb]) {
      if (!pp) continue;
      ctx.beginPath();
      ctx.moveTo(pp.x, pp.y - pp.h * 0.4);
      ctx.lineTo(pair.x, pair.y);
      ctx.stroke();
    }
    ctx.restore();
    const topY = pair.y - n * lift - base * 0.25;
    handText(ctx, `closest approach in ${pair.tca.toFixed(1)} s`, pair.x, topY, { size: 22 * px * this.labelScale, color: pal.risk, halo: pal.paper, align: 'center', font: FONTS.hand, weight: 700 });
    handText(ctx, `pair score ${this.data.risk.raw[pair.i].toFixed(2)} at ${pair.t.toFixed(1)} s · P now ${now.risk.toFixed(3)}`, pair.x, topY + 18 * px * this.labelScale, { size: 18 * px * this.labelScale, color: pal['ink-soft'], halo: pal.paper, align: 'center', font: FONTS.hand });
  }

  drawSelection(t, id, poseOf, px) {
    const { ctx, pal } = this;
    const tr = this.data.byId.get(id);
    if (!tr) return;
    const pts = [];
    for (let i = 0; i < tr.ts.length; i++) pts.push([tr.x[i], tr.y[i]]);
    inkLine(ctx, pts, { color: pal.ink, width: 1.6 * px, seed: 77, amp: 0.8 * px, dash: [6 * px, 5 * px], alpha: 0.75, double: false });
    const [sx, sy] = pts[0];
    blob(ctx, ellipsePts(sx, sy, 5 * px, 5 * px, 10), { fill: pal.paper, stroke: pal.ink, width: 1.4 * px, seed: 5 });
    handText(ctx, `#${tr.id} from here`, sx + 8 * px, sy - 8 * px, { size: 20 * px * this.labelScale, color: pal.ink, halo: pal.paper, font: FONTS.hand });
    const p = poseOf.get(id);
    if (p) {
      const rx = p.w * 0.7 + 10 * px, ry = p.h * 0.62 + 8 * px;
      inkLine(ctx, ellipsePts(p.x, p.y - p.h / 2, rx, ry, 30), { color: pal.ink, width: 1.8 * px, seed: 78, amp: 1.2 * px, closed: true });
    }
  }

  // Event notes are laid out first (lasso + a label spot that stays on canvas and off other labels), so place
  // names can step aside for them; they are drawn last.
  layoutNotes(t, poseOf, px, k, ox, oy) {
    const { ctx } = this;
    const ls = this.labelScale;
    const boxes = [];
    const notes = [];
    const overlaps = (b) => boxes.some((q) => b[0] < q[2] && b[2] > q[0] && b[1] < q[3] && b[3] > q[1]);
    const active = this.data.events.filter((e) => t >= e.start - 0.2 && t <= e.end + 0.6);
    for (const e of active) {
      for (const id of e.track_ids) {
        const p = poseOf.get(id);
        if (!p) continue;
        const rx = p.w * 0.75 + 12 * px, ry = p.h * 0.66 + 10 * px;
        const prog = this.reduced ? 1 : clamp((t - e.start + 0.2) / 0.7, 0, 1);
        const note = { e, id, p, rx, ry, prog, label: null };
        notes.push(note);
        const sx = ox + p.x * k, sy = oy + p.y * k;
        if (prog < 0.6 || sx < 0 || sx > this.vw || sy < 0 || sy > this.vh + p.h * k) continue;
        const title = `${e.label.replace(/_/g, ' ')} · #${id}`;
        const caption = eventCaption(e, t);
        ctx.font = `700 ${25 * ls}px ${FONTS.hand}`;
        const tw = Math.max(ctx.measureText(title).width, ctx.measureText(caption).width * 0.8) + 6;
        const hTop = 24 * ls, hBot = 26 * ls;
        const up = p.y - p.h - 22 * px, down = p.y + 34 * px;
        const rightX = p.x + rx * 0.8 + 20 * px, leftX = p.x - rx * 0.8 - 20 * px;
        const cands = [];
        for (let lift = 0; lift < 4; lift++) {
          const dy = lift * 44 * px;
          cands.push([rightX, up - dy, false], [leftX, up - dy, true], [rightX, down + dy, false], [leftX, down + dy, true]);
        }
        let pick = null;
        for (const [lx, ly, left] of cands) {
          const bx = ox + lx * k, by = oy + ly * k;
          const b = left ? [bx - tw, by - hTop, bx, by + hBot] : [bx, by - hTop, bx + tw, by + hBot];
          if (b[0] < 6 || b[2] > this.vw - 54 || b[1] < 6 || b[3] > this.vh - 6) continue;
          if (overlaps(b)) continue;
          pick = { lx, ly, left, b };
          break;
        }
        if (!pick) continue;
        boxes.push(pick.b);
        note.label = { ...pick, title, caption };
      }
    }
    return { notes, boxes };
  }

  drawPlaces(boxes, px, k, ox, oy) {
    const { ctx, pal } = this;
    const ls = this.labelScale;
    ctx.save();
    ctx.globalAlpha = 0.8;
    for (const pl of SCENERY.places) {
      const x = pl.at[0] * REF_W, y = pl.at[1] * REF_H;
      ctx.font = `600 ${22 * ls}px ${FONTS.hand}`;
      const tw = ctx.measureText(pl.text).width;
      const sx = ox + x * k, sy = oy + y * k;
      const x0 = pl.align === 'right' ? sx - tw : sx;
      const b = [x0 - 4, sy - 22 * ls, x0 + tw + 4, sy + 6];
      if (b[0] < 4 || b[2] > this.vw - 4 || b[1] < 4 || b[3] > this.vh - 4) continue;
      if (boxes.some((q) => b[0] < q[2] && b[2] > q[0] && b[1] < q[3] && b[3] > q[1])) continue;
      handText(ctx, pl.text, x, y, { size: 22 * px * ls, color: pal['ink-soft'], halo: pal.paper, align: pl.align ?? 'left', rot: pl.rot ?? 0, font: FONTS.hand });
    }
    ctx.restore();
  }

  drawNotes(layout, px) {
    const { ctx, pal } = this;
    const ls = this.labelScale;
    for (const { e, id, p, rx, ry, prog, label } of layout.notes) {
      const col = eventColor(e.label);
      const a0 = -2.2 + (id % 5) * 0.2;
      const pts = [];
      const total = Math.PI * 2 + 0.55;
      const steps = Math.max(2, Math.round(40 * prog));
      for (let i = 0; i <= steps; i++) {
        const a = a0 + (total * prog * i) / steps;
        const wob = 1 + 0.05 * Math.sin(a * 3 + id);
        pts.push([p.x + Math.cos(a) * rx * wob, p.y - p.h / 2 + Math.sin(a) * ry * wob]);
      }
      inkLine(ctx, pts, { color: col, width: 2.2 * px, seed: id, amp: 0.8 * px, double: true });
      if (!label) continue;
      const { lx, ly, left, title, caption } = label;
      const below = ly > p.y;
      const from = [p.x + (left ? -rx * 0.62 : rx * 0.62), below ? p.y - p.h * 0.2 : p.y - p.h * 0.75];
      inkLine(ctx, [from, [lx + (left ? 4 * px : -4 * px), ly - (below ? 16 : -4) * px * ls]], { color: col, width: 1.3 * px, seed: id + 3, amp: 1.2 * px, double: false });
      handText(ctx, title, lx, ly, { size: 25 * px * ls, color: col, halo: pal.paper, align: left ? 'right' : 'left', font: FONTS.hand, weight: 700 });
      handText(ctx, caption, lx, ly + 20 * px * ls, { size: 19 * px * ls, color: pal['ink-soft'], halo: pal.paper, align: left ? 'right' : 'left', font: FONTS.hand });
    }
  }

  drawAtlas(px) {
    const { ctx, pal } = this;
    const at = this.data.atlas;
    const { nx, ny } = at.grid;
    const cw = REF_W / nx, ch = REF_H / ny;
    const minN = at.params.min_samples;
    ctx.save();
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const still = at.dwell.still_samples[j][i], veh = at.dwell.vehicle_samples[j][i];
        if (veh >= minN && still / veh > 0.5) {
          ctx.fillStyle = withAlpha(pal['sig-yellow'], 0.28);
          const r = rng(i * 97 + j);
          for (let s = 0; s < 14; s++) {
            ctx.beginPath();
            ctx.arc((i + r()) * cw, (j + r()) * ch, 2.4 * px + 1.2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        const n = at.flow.count[j][i], res = at.flow.resultant[j][i];
        if (n < minN || res < 0.5) continue;
        const h = (at.flow.heading_deg[j][i] * Math.PI) / 180;
        const cx = (i + 0.5) * cw, cy = (j + 0.5) * ch;
        const L = cw * 0.36 * res;
        const dx = Math.cos(h), dy = Math.sin(h);
        const tip = [cx + dx * L, cy + dy * L];
        const alpha = clamp(0.25 + Math.log10(n) * 0.2, 0.25, 0.85);
        inkLine(ctx, [[cx - dx * L, cy - dy * L], tip], { color: pal.ink, width: 1.4 * px + 0.6, seed: i * 31 + j, amp: 0.6, alpha, double: false });
        const hl = L * 0.45;
        inkLine(ctx, [[tip[0] - Math.cos(h - 0.5) * hl, tip[1] - Math.sin(h - 0.5) * hl], tip, [tip[0] - Math.cos(h + 0.5) * hl, tip[1] - Math.sin(h + 0.5) * hl]], { color: pal.ink, width: 1.4 * px + 0.6, seed: i * 31 + j + 1, amp: 0.4, alpha, double: false });
      }
    }
    ctx.restore();
  }

  hit(mx, my) {
    let best = null;
    let bestArea = Infinity;
    for (const b of this.boxes) {
      if (mx >= b.x0 && mx <= b.x1 && my >= b.y0 && my <= b.y1) {
        const area = (b.x1 - b.x0) * (b.y1 - b.y0);
        if (area < bestArea) {
          bestArea = area;
          best = b.id;
        }
      }
    }
    if (best != null) return best;
    let bd = 22;
    for (const b of this.boxes) {
      const d = Math.hypot(b.fx - mx, b.fy - my);
      if (d < bd) {
        bd = d;
        best = b.id;
      }
    }
    return best;
  }

  visibleIds() {
    return this.boxes
      .filter((b) => b.x1 > 0 && b.x0 < this.vw && b.y1 > 0 && b.y0 < this.vh)
      .sort((a, b) => a.fx - b.fx)
      .map((b) => b.id);
  }
}

export { CLASS_NAMES };
