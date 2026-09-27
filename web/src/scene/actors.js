// Road users as small illustrated figures, fitted to the tracked box (foot point = bottom-centre).
// Classes: 0 person, 1 bicycle, 2 motorcycle, 3 car, 4 bus, 5 truck.
import { BODY_COLOURS, COAT_COLOURS } from '../design/tokens.js';
import { blob, ellipsePts, roundRectPts, rectPts, inkLine, rng, shade, withAlpha } from './ink.js';

const SKIN = ['#d9b49a', '#c89a7c', '#e3c2a6', '#a97c5f'];

export function looks(tr) {
  const r = rng(tr.id * 7919);
  return {
    body: BODY_COLOURS[Math.floor(r() * BODY_COLOURS.length)],
    coat: COAT_COLOURS[Math.floor(r() * COAT_COLOURS.length)],
    trousers: r() < 0.6 ? '#3a3e4c' : '#6d7385',
    skin: SKIN[Math.floor(r() * SKIN.length)],
    cargo: r(),
    hat: r() < 0.2,
    bag: r() < 0.35,
  };
}

function shadow(ctx, x, y, w, h, pal) {
  blob(ctx, ellipsePts(x + w * 0.04, y, w * 0.55, Math.max(2, h * 0.09), 18), { fill: withAlpha(pal.ink, 0.16), amp: 0.3, seed: 1 });
}

function wheel(ctx, cx, cy, rx, ry, pal, o) {
  blob(ctx, ellipsePts(cx, cy, rx, ry, 12), { fill: '#2b2d38', stroke: pal.ink, width: o.lw, seed: o.seed + cx, amp: o.amp * 0.5 });
  blob(ctx, ellipsePts(cx, cy, rx * 0.45, ry * 0.45, 8), { fill: '#8d909c', seed: o.seed + cx + 1, amp: 0 });
}

function car(ctx, p, lk, pal, o) {
  const { x, y, w, h, face } = p;
  const x0 = x - w / 2, y0 = y - h;
  const F = (u) => (face > 0 ? x0 + w * u : x0 + w * (1 - u));
  shadow(ctx, x, y - h * 0.04, w, h, pal);
  const body = [[F(0.02), y0 + h * 0.5], [F(0.12), y0 + h * 0.42], [F(0.9), y0 + h * 0.44], [F(0.99), y0 + h * 0.55],
    [F(0.99), y0 + h * 0.82], [F(0.01), y0 + h * 0.82]];
  blob(ctx, body, { fill: lk.body, stroke: pal.ink, width: o.lw, seed: o.seed, amp: o.amp });
  const cabin = [[F(0.14), y0 + h * 0.44], [F(0.26), y0 + h * 0.08], [F(0.66), y0 + h * 0.08], [F(0.84), y0 + h * 0.45]];
  blob(ctx, cabin, { fill: shade(lk.body, -0.08), stroke: pal.ink, width: o.lw, seed: o.seed + 2, amp: o.amp });
  const glass = [[F(0.2), y0 + h * 0.41], [F(0.29), y0 + h * 0.15], [F(0.63), y0 + h * 0.15], [F(0.78), y0 + h * 0.42]];
  blob(ctx, glass, { fill: pal['wash-glass'], seed: o.seed + 3, amp: o.amp * 0.6 });
  inkLine(ctx, [[F(0.47), y0 + h * 0.15], [F(0.49), y0 + h * 0.42]], { color: pal.ink, width: o.lw * 0.8, seed: o.seed + 4, amp: o.amp * 0.3, double: false });
  inkLine(ctx, [[F(0.33), y0 + h * 0.19], [F(0.4), y0 + h * 0.19]], { color: '#ffffff', width: o.lw * 1.2, seed: o.seed + 5, amp: 0, double: false, alpha: 0.7 });
  wheel(ctx, F(0.22), y0 + h * 0.84, w * 0.085, h * 0.12, pal, o);
  wheel(ctx, F(0.79), y0 + h * 0.84, w * 0.085, h * 0.12, pal, o);
  blob(ctx, ellipsePts(F(0.955), y0 + h * 0.6, w * 0.03, h * 0.05, 8), { fill: pal['wash-window-lit'], seed: o.seed + 6, amp: 0 });
  blob(ctx, ellipsePts(F(0.035), y0 + h * 0.6, w * 0.022, h * 0.05, 8), { fill: pal['sig-red'], seed: o.seed + 7, amp: 0 });
}

function bus(ctx, p, lk, pal, o) {
  const { x, y, w, h, face } = p;
  const x0 = x - w / 2, y0 = y - h;
  const F = (u) => (face > 0 ? x0 + w * u : x0 + w * (1 - u));
  shadow(ctx, x, y - h * 0.03, w, h, pal);
  blob(ctx, roundRectPts(x0 + w * 0.01, y0 + h * 0.06, w * 0.98, h * 0.78, h * 0.07), { fill: pal['line-paint'], stroke: pal.ink, width: o.lw, seed: o.seed, amp: o.amp });
  blob(ctx, [[F(0.01), y0 + h * 0.56], [F(0.99), y0 + h * 0.5], [F(0.99), y0 + h * 0.82], [F(0.01), y0 + h * 0.82]], { fill: pal['wash-bus'], seed: o.seed + 1, amp: o.amp * 0.6 });
  blob(ctx, [[F(0.4), y0 + h * 0.82], [F(0.75), y0 + h * 0.5], [F(0.85), y0 + h * 0.5], [F(0.55), y0 + h * 0.82]], { fill: pal['line-paint'], seed: o.seed + 9, amp: 0.3, alpha: 0.9 });
  const win = [[F(0.05), y0 + h * 0.17], [F(0.9), y0 + h * 0.15], [F(0.9), y0 + h * 0.43], [F(0.05), y0 + h * 0.46]];
  blob(ctx, win, { fill: pal['wash-glass'], stroke: pal.ink, width: o.lw * 0.8, seed: o.seed + 2, amp: o.amp * 0.5 });
  for (let k = 1; k < 6; k++) inkLine(ctx, [[F(0.05 + k * 0.142), y0 + h * 0.16], [F(0.05 + k * 0.142), y0 + h * 0.45]], { color: pal.ink, width: o.lw * 0.7, seed: o.seed + 10 + k, amp: 0.2, double: false, alpha: 0.7 });
  blob(ctx, [[F(0.92), y0 + h * 0.12], [F(0.985), y0 + h * 0.14], [F(0.985), y0 + h * 0.46], [F(0.92), y0 + h * 0.44]], { fill: pal['wash-glass'], stroke: pal.ink, width: o.lw * 0.8, seed: o.seed + 3, amp: 0.3 });
  blob(ctx, rectPts(Math.min(F(0.8), F(0.9)), y0 + h * 0.075, w * 0.1, h * 0.05), { fill: pal['wash-window-lit'], seed: o.seed + 4, amp: 0 });
  wheel(ctx, F(0.17), y0 + h * 0.85, w * 0.045, h * 0.1, pal, o);
  wheel(ctx, F(0.8), y0 + h * 0.85, w * 0.045, h * 0.1, pal, o);
}

function truck(ctx, p, lk, pal, o) {
  const { x, y, w, h, face } = p;
  const x0 = x - w / 2, y0 = y - h;
  const F = (u) => (face > 0 ? x0 + w * u : x0 + w * (1 - u));
  shadow(ctx, x, y - h * 0.04, w, h, pal);
  const cab = [[F(0.62), y0 + h * 0.82], [F(0.62), y0 + h * 0.12], [F(0.88), y0 + h * 0.1], [F(0.99), y0 + h * 0.45], [F(0.99), y0 + h * 0.82]];
  blob(ctx, cab, { fill: lk.body, stroke: pal.ink, width: o.lw, seed: o.seed, amp: o.amp });
  blob(ctx, [[F(0.66), y0 + h * 0.18], [F(0.86), y0 + h * 0.16], [F(0.94), y0 + h * 0.42], [F(0.66), y0 + h * 0.42]], { fill: pal['wash-glass'], seed: o.seed + 1, amp: 0.4 });
  blob(ctx, [[F(0.01), y0 + h * 0.55], [F(0.62), y0 + h * 0.55], [F(0.62), y0 + h * 0.74], [F(0.01), y0 + h * 0.74]], { fill: shade(lk.body, -0.15), stroke: pal.ink, width: o.lw, seed: o.seed + 2, amp: o.amp });
  if (lk.cargo < 0.45) {
    blob(ctx, roundRectPts(Math.min(F(0.12), F(0.52)), y0 + h * 0.22, w * 0.4, h * 0.34, h * 0.08), { fill: pal['sig-yellow'], stroke: pal.ink, width: o.lw, seed: o.seed + 3, amp: o.amp });
  } else if (lk.cargo < 0.75) {
    blob(ctx, rectPts(Math.min(F(0.04), F(0.6)), y0 + h * 0.1, w * 0.56, h * 0.46), { fill: pal['wash-concrete'], stroke: pal.ink, width: o.lw, seed: o.seed + 3, amp: o.amp });
    for (let k = 1; k < 7; k++) {
      inkLine(ctx, [[F(0.04 + k * 0.08), y0 + h * 0.13], [F(0.04 + k * 0.08), y0 + h * 0.53]], { color: pal['ink-soft'], width: o.lw * 0.7, seed: o.seed + 20 + k, amp: 0.3, double: false, alpha: 0.45 });
    }
    inkLine(ctx, [[F(0.06), y0 + h * 0.2], [F(0.58), y0 + h * 0.2]], { color: '#ffffff', width: o.lw * 1.4, seed: o.seed + 30, amp: 0.3, double: false, alpha: 0.5 });
  }
  wheel(ctx, F(0.16), y0 + h * 0.84, w * 0.07, h * 0.12, pal, o);
  wheel(ctx, F(0.36), y0 + h * 0.84, w * 0.07, h * 0.12, pal, o);
  wheel(ctx, F(0.84), y0 + h * 0.84, w * 0.07, h * 0.12, pal, o);
  blob(ctx, ellipsePts(F(0.97), y0 + h * 0.62, w * 0.02, h * 0.05, 8), { fill: pal['wash-window-lit'], seed: o.seed + 6, amp: 0 });
}

function person(ctx, p, lk, pal, o, { riding = false } = {}) {
  const { x, y, h, face } = p;
  const moving = p.v > 0.25 && !riding;
  const ph = Math.sin(p.walk * Math.PI * 1.35) * (moving ? 1 : 0);
  const hy = y - h;
  const s = h / 10;
  if (!riding) blob(ctx, ellipsePts(x + s * 0.5, y, s * 2.1, s * 0.5, 12), { fill: withAlpha(pal.ink, 0.16), amp: 0, seed: 2 });
  const hip = [x, hy + s * 5.6];
  if (!riding) {
    for (const k of [-1, 1]) {
      const fx = x + k * ph * s * 1.3;
      inkLine(ctx, [hip, [fx, y - s * 0.2]], { color: lk.trousers, width: s * 0.95, seed: o.seed + k, amp: 0, double: false });
      inkLine(ctx, [[fx - s * 0.1, y], [fx + face * s * 0.8, y]], { color: pal.ink, width: s * 0.7, seed: o.seed + k + 3, amp: 0, double: false });
    }
  }
  const coat = [[x - s * 1.25, hy + s * 2.2], [x + s * 1.25, hy + s * 2.2], [x + s * 1.55, hy + s * 6.1], [x - s * 1.55, hy + s * 6.1]];
  blob(ctx, coat, { fill: lk.coat, stroke: pal.ink, width: o.lw, seed: o.seed + 5, amp: o.amp * 0.4 });
  for (const k of [-1, 1]) {
    const sw = riding ? 0.9 * face : -k * ph * 0.9;
    inkLine(ctx, [[x + k * s * 1.2, hy + s * 2.6], [x + k * s * 1.3 + sw * s * 1.4, hy + s * 5.2]], { color: lk.coat, width: s * 0.8, seed: o.seed + 7 + k, amp: 0, double: false });
  }
  if (lk.bag && !riding) blob(ctx, rectPts(x - face * s * 2.1, hy + s * 3.4, s * 1.2, s * 1.6), { fill: '#6e5d52', stroke: pal.ink, width: o.lw * 0.7, seed: o.seed + 9, amp: 0.2 });
  blob(ctx, ellipsePts(x, hy + s * 1.15, s * 1.05, s * 1.15, 12), { fill: lk.skin, stroke: pal.ink, width: o.lw, seed: o.seed + 10, amp: o.amp * 0.3 });
  if (riding) blob(ctx, ellipsePts(x, hy + s * 0.85, s * 1.25, s * 0.95, 12), { fill: pal['sig-yellow'], stroke: pal.ink, width: o.lw, seed: o.seed + 11, amp: 0.2 });
  else if (lk.hat) blob(ctx, ellipsePts(x, hy + s * 0.4, s * 1.2, s * 0.45, 10), { fill: '#3a3e4c', seed: o.seed + 12, amp: 0.2 });
  else blob(ctx, ellipsePts(x - face * s * 0.2, hy + s * 0.55, s * 1.05, s * 0.55, 10), { fill: '#2f2a2a', seed: o.seed + 13, amp: 0.2 });
}

function twoWheeler(ctx, p, lk, pal, o, motor) {
  const { x, y, w, h, face } = p;
  const x0 = x - w / 2;
  const F = (u) => (face > 0 ? x0 + w * u : x0 + w * (1 - u));
  const r = Math.min(w * 0.2, h * 0.16);
  shadow(ctx, x, y - r * 0.2, w * 0.9, h * 0.5, pal);
  const back = [F(0.2), y - r], front = [F(0.8), y - r];
  for (const c of [back, front]) {
    blob(ctx, ellipsePts(c[0], c[1], r, r, 16), { stroke: pal.ink, width: o.lw * 1.6, seed: o.seed + c[0], amp: 0.2 });
    if (motor) blob(ctx, ellipsePts(c[0], c[1], r * 0.8, r * 0.8, 12), { fill: '#2b2d38', seed: o.seed + 1, amp: 0 });
  }
  const seat = [F(0.42), y - r * 2.4];
  inkLine(ctx, [back, seat, [F(0.7), y - r * 2.6], front], { color: motor ? '#3a3e4c' : lk.body === '#f1f0ea' ? '#c9544f' : lk.body, width: o.lw * 2, seed: o.seed + 3, amp: 0.2, double: false });
  if (motor) {
    blob(ctx, [[F(0.3), y - r * 1.4], [F(0.72), y - r * 1.6], [F(0.66), y - r * 2.6], [F(0.36), y - r * 2.5]], { fill: lk.body === '#f1f0ea' ? '#3a3e4c' : lk.body, stroke: pal.ink, width: o.lw, seed: o.seed + 4, amp: 0.3 });
    if (lk.cargo < 0.5) blob(ctx, rectPts(Math.min(F(0.0), F(0.28)), y - r * 4.2, w * 0.28, r * 1.9), { fill: pal['sig-yellow'], stroke: pal.ink, width: o.lw, seed: o.seed + 5, amp: 0.3 });
  }
  person(ctx, { ...p, x: F(0.47), y: y - r * 1.9, h: h * 0.82 }, { ...lk, coat: lk.cargo < 0.5 && motor ? pal['sig-yellow'] : lk.coat }, pal, o, { riding: true });
}

export function drawActor(ctx, tr, p, pal, o) {
  const lk = looks(tr);
  switch (tr.cls) {
    case 0: return person(ctx, p, lk, pal, o);
    case 1: return twoWheeler(ctx, p, lk, pal, o, false);
    case 2: return twoWheeler(ctx, p, lk, pal, o, true);
    case 4: return bus(ctx, p, lk, pal, o);
    case 5: return truck(ctx, p, lk, pal, o);
    default: return car(ctx, p, lk, pal, o);
  }
}

// A soft warm pool of light ahead of moving vehicles (the footage is at dusk).
export function headlight(ctx, p, pal) {
  const { x, y, w, h, face } = p;
  const fx = x + face * w * 0.5;
  const g = ctx.createRadialGradient(fx + face * w * 0.25, y - h * 0.15, 0, fx + face * w * 0.25, y - h * 0.15, w * 0.45);
  g.addColorStop(0, withAlpha(pal['wash-window-lit'], 0.35));
  g.addColorStop(1, withAlpha(pal['wash-window-lit'], 0));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(fx + face * w * 0.25, y - h * 0.15, w * 0.45, h * 0.22, 0, 0, Math.PI * 2);
  ctx.fill();
}
