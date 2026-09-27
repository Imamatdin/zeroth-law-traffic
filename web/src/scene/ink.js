// Hand-drawn primitives on Canvas 2D: seeded wobble, watercolour washes, hatching. Deterministic per seed so
// the drawing does not shimmer between frames unless we ask it to "boil".

export function rng(seed) {
  let a = (seed | 0) ^ 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const hashSeed = (...xs) => xs.reduce((h, x) => Math.imul(h ^ Math.round(x * 1000), 16777619) >>> 0, 2166136261);

// Smooth 1-D noise from a few seeded sines; cheap and continuous along a stroke.
function noise1(seed) {
  const r = rng(seed);
  const f = [r() * 0.9 + 0.35, r() * 2.1 + 1.2, r() * 4.3 + 2.9];
  const p = [r() * 6.28, r() * 6.28, r() * 6.28];
  return (s) => (Math.sin(s * f[0] + p[0]) * 0.55 + Math.sin(s * f[1] + p[1]) * 0.3 + Math.sin(s * f[2] + p[2]) * 0.15);
}

// Densify a polyline and push each point sideways by smooth noise. `step` and `amp` are in current units.
export function wobble(pts, { seed = 1, amp = 1.2, step = 10, closed = false, freq = 0.08 } = {}) {
  const n = noise1(seed);
  const src = closed ? [...pts, pts[0]] : pts;
  const out = [];
  let s = 0;
  for (let i = 0; i < src.length - 1; i++) {
    const [x0, y0] = src[i];
    const [x1, y1] = src[i + 1];
    const len = Math.hypot(x1 - x0, y1 - y0) || 1e-6;
    const nx = -(y1 - y0) / len;
    const ny = (x1 - x0) / len;
    const k = Math.max(1, Math.ceil(len / step));
    for (let j = 0; j < k; j++) {
      const u = j / k;
      const d = n((s + u * len) * freq) * amp;
      out.push([x0 + (x1 - x0) * u + nx * d, y0 + (y1 - y0) * u + ny * d]);
    }
    s += len;
  }
  if (!closed) {
    const [x, y] = src[src.length - 1];
    out.push([x, y]);
  }
  return out;
}

export function tracePath(ctx, pts, closed = false) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (closed) ctx.closePath();
}

// An ink line: a main stroke plus a fainter second pass slightly off, like a pen going over it again.
export function inkLine(ctx, pts, { color, width = 1.4, seed = 1, amp = 1, step = 10, closed = false, double = true, alpha = 1, dash = null } = {}) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  tracePath(ctx, wobble(pts, { seed, amp, step, closed }), closed);
  ctx.stroke();
  if (double) {
    ctx.globalAlpha = alpha * 0.35;
    ctx.lineWidth = width * 0.7;
    tracePath(ctx, wobble(pts, { seed: seed + 77, amp: amp * 1.4, step, closed }), closed);
    ctx.stroke();
  }
  ctx.restore();
}

// Watercolour wash: a few translucent, slightly shifted fills plus pigment pooling at the edge.
export function wash(ctx, pts, { color, seed = 1, amp = 2.2, step = 14, passes = 3, alpha = 0.42, edge = 0.28 } = {}) {
  const r = rng(seed);
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < passes; i++) {
    const dx = (r() - 0.5) * amp;
    const dy = (r() - 0.5) * amp;
    const w = wobble(pts.map(([x, y]) => [x + dx, y + dy]), { seed: seed + i * 13, amp, step, closed: true });
    ctx.globalAlpha = alpha;
    tracePath(ctx, w, true);
    ctx.fill();
  }
  if (edge > 0) {
    ctx.globalAlpha = edge;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, amp * 0.9);
    tracePath(ctx, wobble(pts, { seed: seed + 5, amp: amp * 0.7, step, closed: true }), true);
    ctx.stroke();
  }
  ctx.restore();
}

// Flat fill with a wobbly outline, for small objects drawn every frame.
export function blob(ctx, pts, { fill, stroke, width = 1.2, seed = 1, amp = 0.6, step = 6, alpha = 1 } = {}) {
  const w = wobble(pts, { seed, amp, step, closed: true });
  ctx.save();
  ctx.globalAlpha = alpha;
  tracePath(ctx, w, true);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }
  ctx.restore();
}

export function ellipsePts(cx, cy, rx, ry, n = 24, rot = 0) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const x = Math.cos(a) * rx;
    const y = Math.sin(a) * ry;
    out.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
  }
  return out;
}

export function rectPts(x, y, w, h) {
  return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
}

export function roundRectPts(x, y, w, h, r, n = 4) {
  r = Math.min(r, w / 2, h / 2);
  const out = [];
  const corner = (cx, cy, a0) => {
    for (let i = 0; i <= n; i++) {
      const a = a0 + (i / n) * (Math.PI / 2);
      out.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
  };
  corner(x + w - r, y + r, -Math.PI / 2);
  corner(x + w - r, y + h - r, 0);
  corner(x + r, y + h - r, Math.PI / 2);
  corner(x + r, y + r, Math.PI);
  return out;
}

// Parallel wobbly hatching clipped to a polygon.
export function hatch(ctx, pts, { color, angle = -0.6, gap = 6, width = 0.8, alpha = 0.35, seed = 3 } = {}) {
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const R = Math.hypot(x1 - x0, y1 - y0) / 2 + gap;
  const dx = Math.cos(angle), dy = Math.sin(angle);
  ctx.save();
  tracePath(ctx, pts, true);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = alpha;
  ctx.lineCap = 'round';
  let i = 0;
  for (let o = -R; o <= R; o += gap) {
    const ox = cx - dy * o, oy = cy + dx * o;
    const line = [[ox - dx * R, oy - dy * R], [ox + dx * R, oy + dy * R]];
    tracePath(ctx, wobble(line, { seed: seed + i++, amp: gap * 0.12, step: 12 }));
    ctx.stroke();
  }
  ctx.restore();
}

// A small tile of paper grain and fibres; callers repeat it as a pattern.
export function grainTile({ ink, size = 192, seed = 11, speck = 0.07 }) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const r = rng(seed);
  const img = g.createImageData(size, size);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = 150 + r() * 105;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  g.strokeStyle = ink;
  g.lineWidth = 0.7;
  for (let i = 0; i < size / 5; i++) {
    const x = r() * size, y = r() * size, a = r() * 6.28, l = 5 + r() * 16;
    g.globalAlpha = speck + r() * speck;
    g.beginPath();
    g.moveTo(x, y);
    g.quadraticCurveTo(x + Math.cos(a + 0.7) * l * 0.5, y + Math.sin(a + 0.7) * l * 0.5, x + Math.cos(a) * l, y + Math.sin(a) * l);
    g.stroke();
  }
  return c;
}

// Pigment blooms: irregular paler and darker patches that break up a flat wash, clipped to its shape.
export function blooms(ctx, pts, { color, paper, seed = 1, count = 18, size = 60 } = {}) {
  const r = rng(seed);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  ctx.save();
  tracePath(ctx, pts, true);
  ctx.clip();
  for (let i = 0; i < count; i++) {
    const cx = x0 + r() * (x1 - x0), cy = y0 + r() * (y1 - y0);
    const s = size * (0.5 + r());
    const shape = wobble(ellipsePts(cx, cy, s, s * (0.4 + r() * 0.4), 14, r() * 3), { seed: seed + i, amp: s * 0.18, step: s * 0.3, closed: true });
    ctx.globalAlpha = 0.035 + r() * 0.05;
    ctx.fillStyle = r() < 0.55 ? paper : color;
    tracePath(ctx, shape, true);
    ctx.fill();
  }
  ctx.restore();
}

// Handwritten label with a paper-coloured halo so it stays legible over washes.
export function handText(ctx, text, x, y, { size = 18, color, halo, align = 'left', rot = 0, weight = 600, font }) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  ctx.font = `${weight} ${size}px ${font}`;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  if (halo) {
    ctx.strokeStyle = halo;
    ctx.lineWidth = size * 0.28;
    ctx.lineJoin = 'round';
    ctx.globalAlpha = 0.85;
    ctx.strokeText(text, 0, 0);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = color;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

export const lerp = (a, b, u) => a + (b - a) * u;
export const lerp2 = (p, q, u) => [lerp(p[0], q[0], u), lerp(p[1], q[1], u)];
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function withAlpha(hex, a) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function shade(hex, k) {
  const h = hex.replace('#', '');
  const n = parseInt(h, 16);
  const f = (v) => Math.round(k < 0 ? v * (1 + k) : v + (255 - v) * k);
  return `rgb(${f((n >> 16) & 255)},${f((n >> 8) & 255)},${f(n & 255)})`;
}
