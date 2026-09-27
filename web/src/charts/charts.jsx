// Small SVG chart kit for the EDA and results sections. One axis per chart, single data hue, hairline grid,
// hover/focus readouts and a table view for every figure (dataviz rules).
import { useEffect, useId, useMemo, useRef, useState } from 'react';

export function useWidth(initial = 600) {
  const ref = useRef(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(120, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const niceMax = (v) => {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * p).find((m) => m >= v);
};

export function TableView({ caption, columns, rows, summary = 'Table' }) {
  return (
    <details className="table-view">
      <summary>{summary}</summary>
      <div className="table-scroll">
        <table>
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>{columns.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>{r.map((v, j) => <td key={j} className="num">{v}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

// Line (or area) over time with a crosshair readout. Arrow keys move the readout when focused.
export function LineChart({ x, y, height = 160, yLabel, xLabel = 's', format = (v) => v.toFixed(1), area = false, yMax, ariaLabel, marks = [] }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const L = 44, R = 12, T = 12, B = 26;
  const pw = width - L - R, ph = height - T - B;
  const x0 = x[0], x1 = x[x.length - 1];
  const top = yMax ?? niceMax(Math.max(...y));
  const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * pw;
  const Y = (v) => T + ph - (v / top) * ph;
  const d = useMemo(() => y.map((v, i) => `${i ? 'L' : 'M'}${X(x[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(''), [x, y, width, top]);
  const ticks = [0, top / 2, top];
  const xticks = [];
  const step = niceMax((x1 - x0) / (width < 500 ? 3 : 6));
  for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) xticks.push(v);
  const pick = (px) => {
    const t = x0 + ((px - L) / pw) * (x1 - x0);
    let i = 0;
    for (let j = 0; j < x.length; j++) if (Math.abs(x[j] - t) < Math.abs(x[i] - t)) i = j;
    return i;
  };
  const onKey = (e) => {
    const s = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (!s) return;
    e.preventDefault();
    setHover((h) => Math.max(0, Math.min(x.length - 1, (h ?? 0) + s * Math.max(1, Math.round(x.length / 60)))));
  };
  return (
    <div className="chart" ref={ref}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
        onPointerMove={(e) => setHover(pick(e.clientX - e.currentTarget.getBoundingClientRect().left))}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={L + pw} y1={Y(v)} y2={Y(v)} className="grid" />
            <text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick num">{format(v)}</text>
          </g>
        ))}
        {xticks.map((v) => (
          <text key={v} x={X(v)} y={height - 8} textAnchor="middle" className="tick num">{`${+v.toFixed(1)}`}</text>
        ))}
        <text x={L + pw} y={height - 8} textAnchor="end" className="tick">{xLabel}</text>
        {yLabel && <text x={L} y={T - 2} className="tick">{yLabel}</text>}
        {marks.map((m, i) => (
          <g key={i}>
            <rect x={X(m.x0)} y={T} width={Math.max(1, X(m.x1) - X(m.x0))} height={ph} fill={m.fill} opacity={0.14} />
          </g>
        ))}
        {area && <path d={`${d}L${X(x1)},${Y(0)}L${X(x0)},${Y(0)}Z`} fill="var(--data)" opacity={0.16} />}
        <path d={d} fill="none" stroke="var(--data)" strokeWidth={2} strokeLinejoin="round" />
        {hover != null && (
          <g className="crosshair">
            <line x1={X(x[hover])} x2={X(x[hover])} y1={T} y2={T + ph} />
            <circle cx={X(x[hover])} cy={Y(y[hover])} r={4} fill="var(--data)" stroke="var(--paper)" strokeWidth={2} />
            <text x={Math.min(X(x[hover]) + 8, L + pw - 90)} y={T + 12} className="readout num">
              {`${+x[hover].toFixed(1)} ${xLabel} · ${format(y[hover])}`}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

// Horizontal bars with the value at the bar end (text in ink, not the series colour).
export function BarList({ items, format = (v) => v, ariaLabel }) {
  const max = Math.max(...items.map((i) => i.value), 1);
  return (
    <ul className="barlist" aria-label={ariaLabel}>
      {items.map((it) => (
        <li key={it.label} title={`${it.label}: ${format(it.value)}`}>
          <span className="bl-label">{it.label}</span>
          <span className="bl-track">
            <span className="bl-bar" style={{ width: `${(it.value / max) * 100}%` }} />
          </span>
          <span className="bl-value num">{format(it.value)}</span>
        </li>
      ))}
    </ul>
  );
}

// Bins past `cap` fold into one overflow bin labelled "≥ cap" so the bulk of the distribution stays readable.
export function Histogram({ edges: allEdges, counts: allCounts, cap, height = 110, marker, markerLabel, ariaLabel, unit = '' }) {
  const [ref, width] = useWidth(240);
  const cut = cap != null ? allEdges.findIndex((e) => e >= cap) : -1;
  const edges = cut > 0 ? [...allEdges.slice(0, cut + 1), cap + (allEdges[1] - allEdges[0])] : allEdges;
  const counts = cut > 0 ? [...allCounts.slice(0, cut), allCounts.slice(cut).reduce((a, b) => a + b, 0)] : allCounts;
  const L = 6, R = 6, T = 16, B = 20;
  const pw = width - L - R, ph = height - T - B;
  const max = Math.max(...counts, 1);
  const lo = edges[0], hi = edges[edges.length - 1];
  const X = (v) => L + ((v - lo) / (hi - lo)) * pw;
  const bw = pw / counts.length;
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        <line x1={L} x2={L + pw} y1={T + ph} y2={T + ph} className="grid" />
        {counts.map((c, i) => (
          <rect key={i} x={X(edges[i]) + 1} y={T + ph - (c / max) * ph} width={Math.max(0.5, bw - 2)} height={(c / max) * ph} rx={1.5} fill="var(--data)" opacity={0.75}>
            <title>{cut > 0 && i === counts.length - 1 ? `≥ ${cap} ${unit}: ${c}` : `${edges[i]}–${edges[i + 1]} ${unit}: ${c}`}</title>
          </rect>
        ))}
        {marker != null && (
          <g>
            <line x1={X(marker)} x2={X(marker)} y1={T - 4} y2={T + ph} stroke="var(--ink)" strokeWidth={1.2} />
            <text x={Math.min(X(marker) + 4, width - 70)} y={T - 5} className="tick num">{markerLabel}</text>
          </g>
        )}
        <text x={L} y={height - 5} className="tick num">{lo}</text>
        <text x={L + pw} y={height - 5} textAnchor="end" className="tick num">{cut > 0 ? `≥ ${cap} ${unit}` : `${hi} ${unit}`}</text>
      </svg>
    </div>
  );
}

// The camera scene as ink outlines in reference pixels (1920 x 1080); children draw in the same space.
export function MiniMap({ camera, children, ariaLabel, legend }) {
  const P = (pts) => pts.map(([x, y]) => `${(x * 1920).toFixed(1)},${(y * 1080).toFixed(1)}`).join(' ');
  const id = useId();
  return (
    <figure className="minimap">
      <svg viewBox="0 0 1920 1080" role="img" aria-label={ariaLabel}>
        <defs>
          <filter id={`${id}-rough`}>
            <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="3" />
            <feDisplacementMap in="SourceGraphic" scale="5" />
          </filter>
        </defs>
        <rect width="1920" height="1080" fill="var(--paper-deep)" />
        <g filter={`url(#${id}-rough)`} fill="none" stroke="var(--ink-faint)" strokeWidth="3">
          <polygon points={P(camera.road)} fill="var(--paper)" />
          {(camera.sidewalks ?? []).map((s) => <polygon key={s.id} points={P(s.polygon)} />)}
          {(camera.median ?? []).map((s) => <polygon key={s.id} points={P(s.polygon)} fill="var(--paper-deep)" />)}
          {(camera.islands ?? []).map((s) => <polygon key={s.id} points={P(s.polygon)} fill="var(--paper-deep)" />)}
          {(camera.crosswalks ?? []).map((s) => <polygon key={s.id} points={P(s.polygon)} strokeDasharray="10 8" />)}
          {(camera.stop_lines ?? []).map((s) => <polyline key={s.id} points={P(s.points)} stroke="var(--ink)" strokeWidth="6" />)}
        </g>
        {children}
      </svg>
      {legend && <figcaption>{legend}</figcaption>}
    </figure>
  );
}

// Sequential single-hue grid over the minimap.
export function HeatCells({ grid, nx, ny, label }) {
  const max = Math.max(...grid.flat(), 1);
  const cw = 1920 / nx, ch = 1080 / ny;
  return (
    <g aria-label={label}>
      {grid.flatMap((row, j) => row.map((v, i) => (v > 0 ? (
        <rect key={`${i}-${j}`} x={i * cw} y={j * ch} width={cw} height={ch} fill="var(--data)" opacity={0.08 + 0.82 * Math.sqrt(v / max)}>
          <title>{`${v} samples`}</title>
        </rect>
      ) : null)))}
    </g>
  );
}

export function RampLegend({ label, lo = 'few', hi = 'many' }) {
  return (
    <span className="ramp-legend">
      <span>{label}</span>
      <span className="num">{lo}</span>
      <span className="ramp" aria-hidden="true" />
      <span className="num">{hi}</span>
    </span>
  );
}

export function Stat({ value, label, note }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}
