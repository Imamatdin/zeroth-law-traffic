// The unmoving part of the miniature, drawn once per resolution: ground layer (one canvas) and props (one
// sprite each, depth-sorted against road users every frame). Geometry: configs/camera.yaml via scene.json;
// decoration: scenery.js.
import { REF_W, REF_H } from '../data/load.js';
import { SCENERY, ZEBRA } from './scenery.js';
import {
  wash, inkLine, hatch, blob, ellipsePts, rectPts, roundRectPts, wobble, tracePath, lerp2, rng,
  grainTile, blooms, shade, withAlpha,
} from './ink.js';

const P = ([x, y]) => [x * REF_W, y * REF_H];
const poly = (pts) => pts.map(P);

function unit([dx, dy]) {
  const l = Math.hypot(dx, dy) || 1;
  return [dx / l, dy / l];
}

function dashedLine(ctx, a, b, { color, width, seed, dash0 = 26, gapRatio = 1.4 }) {
  // Perspective: dashes shrink towards the far (upper) end, scaled by image y.
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  let s = 0;
  let i = 0;
  while (s < len) {
    const u = s / len;
    const y = a[1] + (b[1] - a[1]) * u;
    const scale = 0.25 + (y / REF_H) * 1.6;
    const d = dash0 * scale;
    const u1 = Math.min(1, (s + d) / len);
    inkLine(ctx, [lerp2(a, b, u), lerp2(a, b, u1)], { color, width: width * scale, seed: seed + i++, amp: 0.3, double: false, alpha: 0.85 });
    s += d * (1 + gapRatio);
  }
}

function zebra(ctx, pts, dir, stripes, pal, seed) {
  const d = unit([dir[0] * REF_W, dir[1] * REF_H]);
  const n = [-d[1], d[0]];
  const proj = pts.map(([x, y]) => x * n[0] + y * n[1]);
  const along = pts.map(([x, y]) => x * d[0] + y * d[1]);
  const m0 = Math.min(...proj), m1 = Math.max(...proj);
  const a0 = Math.min(...along) - 40, a1 = Math.max(...along) + 40;
  const period = (m1 - m0) / stripes;
  ctx.save();
  tracePath(ctx, pts, true);
  ctx.clip();
  for (let i = 0; i < stripes; i++) {
    const o0 = m0 + (i + 0.22) * period, o1 = m0 + (i + 0.72) * period;
    const q = (o, a) => [n[0] * o + d[0] * a, n[1] * o + d[1] * a];
    const band = [q(o0, a0), q(o0, a1), q(o1, a1), q(o1, a0)];
    blob(ctx, band, { fill: pal['line-paint'], seed: seed + i, amp: 0.8, step: 8, alpha: 0.92 });
    inkLine(ctx, [q(o0, a0), q(o0, a1)], { color: pal['ink-soft'], width: 0.6, seed: seed + 50 + i, amp: 0.5, double: false, alpha: 0.35 });
  }
  ctx.restore();
}

export function drawGround(ctx, cam, pal) {
  const r = rng(5);
  // pavement everywhere first; the road and furniture are laid over it
  wash(ctx, rectPts(-20, -20, REF_W + 40, REF_H + 40), { color: pal['wash-pavement'], seed: 2, alpha: 0.5, passes: 2, edge: 0 });

  for (const [i, b] of SCENERY.buildings.entries()) {
    const pts = poly(b.poly);
    wash(ctx, pts, { color: b.glass ? pal['wash-building-deep'] : pal['wash-building'], seed: 30 + i, alpha: 0.55 });
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), yb = Math.max(...ys);
    if (b.fins) {
      for (let k = 0; k <= b.fins; k++) {
        const x = x0 + ((x1 - x0) * k) / b.fins;
        inkLine(ctx, [[x, 0], [x, yb - 6 - k * 1.2]], { color: pal['ink-soft'], width: 1, seed: 90 + k, amp: 0.6, alpha: 0.55 });
      }
    }
    const rows = b.floors * 2;
    for (let row = 0; row < rows; row++) {
      for (let x = x0 + 14; x < x1 - 14; x += 26) {
        const y = 10 + row * 26;
        if (y > yb - 30) continue;
        const lit = r() < 0.38;
        blob(ctx, rectPts(x, y, 12, 14), { fill: lit ? pal['wash-window-lit'] : withAlpha(pal.ink, 0.18), seed: x * 7 + y, amp: 0.5, alpha: lit ? 0.85 : 1 });
      }
    }
    inkLine(ctx, pts, { color: pal.ink, width: 1.2, seed: 40 + i, amp: 1.2, closed: true, alpha: 0.6 });
  }

  for (const [i, g] of SCENERY.grass.entries()) {
    const pts = poly(g);
    wash(ctx, pts, { color: pal['wash-grass'], seed: 60 + i, alpha: 0.6 });
    inkLine(ctx, pts, { color: pal['ink-soft'], width: 1, seed: 61 + i, closed: true, alpha: 0.5 });
  }

  const road = poly(cam.road);
  wash(ctx, road, { color: pal['wash-asphalt'], seed: 7, alpha: 0.5, passes: 3, amp: 3, edge: 0.18 });
  // atmospheric perspective: the far carriageway sits in a paler wash
  ctx.save();
  tracePath(ctx, road, true);
  ctx.clip();
  const haze = ctx.createLinearGradient(0, 0, 0, REF_H * 0.55);
  haze.addColorStop(0, withAlpha(pal.paper, 0.45));
  haze.addColorStop(1, withAlpha(pal.paper, 0));
  ctx.fillStyle = haze;
  ctx.fillRect(0, 0, REF_W, REF_H);
  ctx.restore();
  blooms(ctx, road, { color: pal['wash-asphalt'], paper: pal.paper, seed: 13, count: 160, size: 38 });
  hatch(ctx, road, { color: pal.ink, angle: 0.29, gap: 9, width: 0.6, alpha: 0.07, seed: 12 });
  inkLine(ctx, road, { color: pal.ink, width: 1.6, seed: 8, amp: 1.4, closed: true, alpha: 0.75 });

  for (const [i, s] of (cam.sidewalks ?? []).entries()) {
    const pts = poly(s.polygon);
    wash(ctx, pts, { color: pal['wash-plaza'], seed: 70 + i, alpha: 0.6 });
    blooms(ctx, pts, { color: pal['wash-plaza'], paper: pal.paper, seed: 74 + i, count: 14, size: 40 });
    hatch(ctx, pts, { color: pal.ink, angle: 0.5, gap: 11, width: 0.6, alpha: 0.12, seed: 71 + i });
    hatch(ctx, pts, { color: pal.ink, angle: -1.1, gap: 11, width: 0.6, alpha: 0.1, seed: 72 + i });
    inkLine(ctx, pts, { color: pal.ink, width: 1.5, seed: 73 + i, closed: true, amp: 1.1 });
  }

  // lane dashes: straight lines between the carriageway edges of camera.yaml (perspective keeps them straight)
  const med = cam.median?.[0]?.polygon;
  const plaza = cam.sidewalks?.find((s) => s.id === 'left_plaza')?.polygon;
  const stop = cam.stop_lines?.[0];
  if (med && plaza && stop) {
    const [sa, sb] = stop.points.map(P);
    const nearFar = [P(plaza[0]), P(med[med.length - 1])];
    for (const f of [0.25, 0.5, 0.75]) dashedLine(ctx, lerp2(nearFar[0], nearFar[1], f), lerp2(sa, sb, f), { color: pal['line-paint'], width: 3, seed: 100 + f * 10 });
    const farNear = [P(med[2]), P(cam.road[5])];
    const farFar = [P(med[0]), P(cam.road[0])];
    for (const f of [1 / 3, 2 / 3]) dashedLine(ctx, lerp2(farFar[0], farFar[1], f), lerp2(farNear[0], farNear[1], f), { color: pal['line-paint'], width: 3, seed: 120 + f * 10 });
  }

  for (const [i, m] of (cam.median ?? []).entries()) {
    const pts = poly(m.polygon);
    const kerb = pts.map(([x, y]) => [x, y + 7]);
    wash(ctx, kerb, { color: pal['wash-kerb'], seed: 80 + i, alpha: 0.7, edge: 0 });
    wash(ctx, pts, { color: pal['wash-concrete'], seed: 81 + i, alpha: 0.75 });
    blooms(ctx, pts, { color: pal['wash-kerb'], paper: pal.paper, seed: 79 + i, count: 16, size: 30 });
    inkLine(ctx, pts, { color: pal.ink, width: 1.6, seed: 82 + i, closed: true });
    const lower = [pts[7], pts[6], pts[5]].map(([x, y]) => [x, y + 11]);
    inkLine(ctx, lower, { color: pal['line-yellow'], width: 2.6, seed: 83, amp: 0.8, double: false, alpha: 0.9 });
    const upper = [pts[0], pts[1], pts[2]].map(([x, y]) => [x, y - 5]);
    inkLine(ctx, upper, { color: pal['line-yellow'], width: 2.2, seed: 84, amp: 0.8, double: false, alpha: 0.8 });
  }

  for (const [i, isl] of (cam.islands ?? []).entries()) {
    const pts = poly(isl.polygon);
    if (isl.id === 'median_nose') {
      wash(ctx, pts, { color: pal['wash-concrete'], seed: 85, alpha: 0.8 });
      const outline = wobble(pts, { seed: 86, amp: 0.4, step: 5, closed: true });
      for (let k = 0; k < outline.length; k++) {
        const a = outline[k], b = outline[(k + 1) % outline.length];
        ctx.strokeStyle = k % 2 ? pal.ink : pal['line-paint'];
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
      continue;
    }
    const kerb = pts.map(([x, y]) => [x, y + 6]);
    wash(ctx, kerb, { color: pal['wash-kerb'], seed: 87 + i, alpha: 0.75, edge: 0 });
    wash(ctx, pts, { color: pal['wash-refuge'], seed: 88 + i, alpha: 0.7 });
    hatch(ctx, pts, { color: shade(pal['wash-refuge'], -0.45), angle: 0.4, gap: 7, width: 0.6, alpha: 0.25, seed: 89 + i });
    hatch(ctx, pts, { color: shade(pal['wash-refuge'], -0.45), angle: -1.2, gap: 7, width: 0.6, alpha: 0.2, seed: 90 + i });
    inkLine(ctx, pts, { color: pal.ink, width: 1.6, seed: 91 + i, closed: true });
  }

  const dirs = Object.fromEntries((cam.approaches ?? []).map((a) => [a.id, a.direction]));
  for (const [i, c] of (cam.crosswalks ?? []).entries()) {
    const cfg = ZEBRA[c.id];
    const dir = dirs[cfg?.approach] ?? [1, 0.3];
    zebra(ctx, poly(c.polygon), dir, cfg?.stripes ?? 20, pal, 200 + i * 40);
  }

  for (const s of cam.stop_lines ?? []) {
    const pts = s.points.map(P);
    inkLine(ctx, pts, { color: pal['line-paint'], width: 7, seed: 150, amp: 0.8, double: false });
    inkLine(ctx, pts.map(([x, y]) => [x, y + 4]), { color: pal.ink, width: 0.8, seed: 151, amp: 0.8, double: false, alpha: 0.6 });
  }

  for (const [i, c] of SCENERY.cracks.entries()) inkLine(ctx, poly(c), { color: pal.ink, width: 0.8, seed: 160 + i, amp: 2.2, step: 6, alpha: 0.35, double: false });

}

// ---------- props: each returns {baseY, bbox, draw(ctx)} in reference pixels ----------

function tree(t, i, pal) {
  const [bx, by] = P([t.x, t.y]);
  const R = t.r * REF_W;
  const cy = by - t.lift * REF_H;
  return {
    baseY: by,
    bbox: [bx - R * 1.4, cy - R * 1.3, bx + R * 1.4, by + 8],
    draw(ctx) {
      const r = rng(300 + i);
      const trunkTop = cy + R * 0.4;
      inkLine(ctx, [[bx - 2, by], [bx - 1.5, trunkTop]], { color: '#6e5d52', width: Math.max(4, R * 0.12), seed: 301 + i, amp: 1 });
      inkLine(ctx, [[bx - 2, by], [bx - 2, by - R * 0.35]], { color: pal['line-paint'], width: Math.max(4, R * 0.12), seed: 302 + i, amp: 0.5, double: false });
      for (let k = 0; k < 5; k++) {
        const ox = (r() - 0.5) * R * 1.1, oy = (r() - 0.5) * R * 0.7;
        const rr = R * (0.55 + r() * 0.35);
        wash(ctx, ellipsePts(bx + ox, cy + oy, rr, rr * 0.86, 20), { color: k < 2 ? pal['wash-leaf-deep'] : pal['wash-leaf'], seed: 310 + i * 9 + k, alpha: 0.5, amp: 3.5 });
      }
      const outline = ellipsePts(bx, cy, R * 1.05, R * 0.95, 26).slice(2, 22);
      inkLine(ctx, outline, { color: pal.ink, width: 1.3, seed: 330 + i, amp: R * 0.06, step: 7, alpha: 0.7 });
      for (let k = 0; k < 6; k++) {
        const a = r() * 6.28, d = r() * R * 0.7;
        const x = bx + Math.cos(a) * d, y = cy + Math.sin(a) * d * 0.8;
        inkLine(ctx, [[x, y], [x + 6, y - 4], [x + 12, y + 1]], { color: pal['ink-soft'], width: 0.9, seed: 340 + k + i, amp: 1, alpha: 0.4, double: false });
      }
    },
  };
}

function busStop(pal) {
  const s = SCENERY.busStop;
  const [x0, yb] = P([s.x0, s.base]);
  const [x1] = P([s.x1, s.base]);
  const yt = s.top * REF_H;
  return {
    baseY: yb,
    bbox: [x0 - 10, yt - 30, x1 + 10, yb + 6],
    draw(ctx) {
      wash(ctx, rectPts(x0 + 6, yt + 6, x1 - x0 - 12, yb - yt - 8), { color: pal['wash-glass'], seed: 401, alpha: 0.55 });
      inkLine(ctx, [[x0 + 4, yb], [x0 + 4, yt]], { color: pal.ink, width: 2, seed: 402 });
      inkLine(ctx, [[x1 - 4, yb], [x1 - 4, yt]], { color: pal.ink, width: 2, seed: 403 });
      blob(ctx, roundRectPts(x0 - 8, yt - 8, x1 - x0 + 16, 12, 4), { fill: pal['wash-building-deep'], stroke: pal.ink, width: 1.4, seed: 404 });
      blob(ctx, rectPts(x0 + 14, yt - 30, 46, 20), { fill: pal['line-paint'], stroke: pal.ink, width: 1.2, seed: 405 });
      ctx.fillStyle = pal['wash-bus'];
      ctx.fillRect(x0 + 20, yt - 25, 10, 10);
      inkLine(ctx, [[x0 + 12, yb - 12], [x1 - 12, yb - 12]], { color: pal['ink-soft'], width: 3, seed: 406, alpha: 0.8 });
    },
  };
}

function gantry(pal, cam) {
  const g = SCENERY.gantry;
  const posts = g.posts.map(P);
  const [a, b] = g.arm.map(P);
  const top = g.top * REF_H;
  const ped = cam.signals?.find((s) => s.kind === 'pedestrian');
  return {
    baseY: Math.max(...posts.map((p) => p[1])),
    bbox: [posts[0][0] - 30, a[1] - 60, b[0] + 40, posts[0][1] + 10],
    draw(ctx) {
      for (const [i, [x, y]] of posts.entries()) {
        inkLine(ctx, [[x, y], [x, top]], { color: '#8d97a3', width: 7, seed: 501 + i, amp: 0.6, double: false });
        inkLine(ctx, [[x + 3.5, y], [x + 3.5, top]], { color: pal.ink, width: 1.1, seed: 503 + i, amp: 0.6, alpha: 0.8 });
      }
      inkLine(ctx, [a, b], { color: '#8d97a3', width: 7, seed: 505, amp: 0.6, double: false });
      inkLine(ctx, [[a[0], a[1] + 4], [b[0], b[1] + 4]], { color: pal.ink, width: 1.1, seed: 506, amp: 0.6, alpha: 0.8 });
      for (const [i, h] of g.heads.entries()) {
        const [x, y] = P(h);
        blob(ctx, roundRectPts(x - 9, y, 18, 44, 4), { fill: '#2b2d38', stroke: pal.ink, width: 1.2, seed: 510 + i });
        for (let k = 0; k < 3; k++) blob(ctx, ellipsePts(x, y + 8 + k * 13, 4, 4, 10), { fill: '#595c68', seed: 520 + k + i * 3, amp: 0.3 });
      }
      const [sx, sy] = P(g.sign);
      blob(ctx, [[sx, sy - 22], [sx + 22, sy], [sx, sy + 22], [sx - 22, sy]], { fill: pal['line-paint'], stroke: pal.ink, width: 1.3, seed: 530 });
      if (ped) {
        const [x0, y0, x1, y1] = ped.roi;
        const [px0, py0] = P([x0, y0]);
        const [px1, py1] = P([x1, y1]);
        blob(ctx, rectPts(px0, py0, px1 - px0, py1 - py0), { fill: '#2b2d38', stroke: pal.ink, width: 1.2, seed: 531 });
      }
    },
  };
}

function lamp(pal) {
  const l = SCENERY.lamp;
  const [bx, by] = P(l.base);
  const [, ty] = P(l.top);
  const [p0x, p0y] = P([l.panel[0], l.panel[1]]);
  const [p1x, p1y] = P([l.panel[2], l.panel[3]]);
  return {
    baseY: by,
    bbox: [p0x - 10, p0y - 10, p1x + 60, by + 6],
    draw(ctx) {
      inkLine(ctx, [[bx, by], [bx, ty]], { color: '#8d97a3', width: 4.5, seed: 601, amp: 0.8, double: false });
      inkLine(ctx, [[bx + 2.2, by], [bx + 2.2, ty]], { color: pal.ink, width: 0.9, seed: 602, amp: 0.8, alpha: 0.8 });
      blob(ctx, [[p0x, p1y], [p1x, p0y + 10], [p1x, p0y], [p0x, p0y + 6]], { fill: '#56688a', stroke: pal.ink, width: 1.2, seed: 603 });
      inkLine(ctx, [[bx, ty + 40], [bx + 50, ty + 30]], { color: pal.ink, width: 2, seed: 604 });
      blob(ctx, ellipsePts(bx + 54, ty + 33, 9, 5, 12), { fill: pal['wash-window-lit'], stroke: pal.ink, width: 1, seed: 605 });
    },
  };
}

function signPole(pal, spec, kind, seed) {
  const [bx, by] = P(spec.base);
  const [, ty] = P(spec.top);
  return {
    baseY: by,
    bbox: [bx - 26, ty - 26, bx + 26, by + 6],
    draw(ctx) {
      inkLine(ctx, [[bx, by], [bx, ty]], { color: '#8d97a3', width: 4, seed, amp: 0.6, double: false });
      inkLine(ctx, [[bx + 2, by], [bx + 2, ty]], { color: pal.ink, width: 0.9, seed: seed + 1, amp: 0.6, alpha: 0.8 });
      if (kind === 'crossing') {
        blob(ctx, rectPts(bx - 17, ty - 4, 34, 34), { fill: pal['wash-sign'], stroke: pal.ink, width: 1.3, seed: seed + 2 });
        blob(ctx, [[bx, ty + 2], [bx + 13, ty + 25], [bx - 13, ty + 25]], { fill: pal['line-paint'], seed: seed + 3, amp: 0.3 });
        blob(ctx, ellipsePts(bx, ty + 12, 2.5, 2.5, 8), { fill: pal.ink, seed: seed + 4, amp: 0.1 });
        inkLine(ctx, [[bx, ty + 15], [bx - 3, ty + 22]], { color: pal.ink, width: 1.4, seed: seed + 5, amp: 0.2, double: false });
      } else if (kind === 'round') {
        blob(ctx, ellipsePts(bx, ty, 15, 15, 18), { fill: pal['wash-sign'], stroke: pal.ink, width: 1.3, seed: seed + 2 });
        inkLine(ctx, [[bx - 7, ty - 7], [bx + 6, ty + 6], [bx + 6, ty - 1], [bx + 6, ty + 6], [bx - 1, ty + 6]], { color: pal['line-paint'], width: 2.4, seed: seed + 3, amp: 0.2, double: false });
      }
    },
  };
}

// The vehicle head on the median. Lamp cells come from camera.yaml; the lit lamp is drawn per frame.
function medianSignal(pal, cam) {
  const sig = cam.signals?.find((s) => s.kind === 'vehicle');
  const spec = SCENERY.medianPole;
  const [bx, by] = P(spec.base);
  const roi = sig ? [P([sig.roi[0], sig.roi[1]]), P([sig.roi[2], sig.roi[3]])] : null;
  const lamps = sig?.lamps ? Object.fromEntries(Object.entries(sig.lamps).map(([k, [x0, y0, x1, y1]]) => {
    const [ax, ay] = P([x0, y0]);
    const [cx, cy] = P([x1, y1]);
    return [k, [(ax + cx) / 2, (ay + cy) / 2, Math.max(cx - ax, cy - ay) * 0.75]];
  })) : null;
  const ty = roi ? roi[1][1] : P(spec.top)[1];
  return {
    baseY: by,
    signalId: sig?.id,
    lamps,
    bbox: roi ? [roi[0][0] - 12, roi[0][1] - 12, roi[1][0] + 12, by + 6] : [bx - 20, ty - 20, bx + 20, by + 6],
    draw(ctx) {
      inkLine(ctx, [[bx, by], [bx, ty]], { color: '#8d97a3', width: 5, seed: 701, amp: 0.6, double: false });
      inkLine(ctx, [[bx + 2.5, by], [bx + 2.5, ty]], { color: pal.ink, width: 1, seed: 702, amp: 0.6, alpha: 0.8 });
      if (roi) {
        const [[x0, y0], [x1, y1]] = roi;
        const cx = (x0 + x1) / 2;
        blob(ctx, roundRectPts(cx - 16, y0 + 2, 32, y1 - y0 - 4, 6), { fill: '#2b2d38', stroke: pal.ink, width: 1.4, seed: 703 });
        if (lamps) for (const [i, [lx, ly, lr]] of Object.values(lamps).entries()) blob(ctx, ellipsePts(cx, ly, lr, lr, 12), { fill: '#555866', seed: 704 + i, amp: 0.3 });
      }
    },
  };
}

function small(pal, kind, at, seed) {
  const [x, y] = P(at);
  return {
    baseY: y,
    bbox: [x - 14, y - 34, x + 14, y + 4],
    draw(ctx) {
      if (kind === 'bollard') {
        blob(ctx, roundRectPts(x - 4, y - 26, 8, 26, 3), { fill: pal['line-paint'], stroke: pal.ink, width: 1, seed });
        blob(ctx, rectPts(x - 4, y - 18, 8, 6), { fill: pal['sig-red'], seed: seed + 1, amp: 0.2 });
      } else {
        blob(ctx, rectPts(x - 11, y - 22, 22, 22), { fill: pal['wash-concrete'], stroke: pal.ink, width: 1.1, seed });
      }
    },
  };
}

export function buildProps(cam, pal) {
  const props = [
    ...SCENERY.trees.map((t, i) => tree(t, i, pal)),
    busStop(pal),
    gantry(pal, cam),
    lamp(pal),
    signPole(pal, SCENERY.crossingSign, 'crossing', 800),
    signPole(pal, SCENERY.roundSign, 'round', 820),
    medianSignal(pal, cam),
    ...SCENERY.bollards.map((b, i) => small(pal, 'bollard', b, 900 + i * 3)),
    ...SCENERY.bins.map((b, i) => small(pal, 'bin', b, 950 + i * 3)),
  ];
  return props;
}

// Render ground and prop sprites at `scale` device pixels per reference pixel.
export function renderStatic(cam, pal, props, scale) {
  const W = Math.round(REF_W * scale), H = Math.round(REF_H * scale);
  const ground = document.createElement('canvas');
  ground.width = W;
  ground.height = H;
  const g = ground.getContext('2d');
  g.fillStyle = pal.paper;
  g.fillRect(0, 0, W, H);
  g.setTransform(scale, 0, 0, scale, 0, 0);
  drawGround(g, cam, pal);
  const sprites = props.map((p) => {
    const [x0, y0, x1, y1] = p.bbox;
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.ceil((x1 - x0) * scale));
    c.height = Math.max(1, Math.ceil((y1 - y0) * scale));
    const cx = c.getContext('2d');
    cx.setTransform(scale, 0, 0, scale, -x0 * scale, -y0 * scale);
    p.draw(cx);
    return c;
  });
  return { ground, sprites, scale };
}

export { grainTile };
