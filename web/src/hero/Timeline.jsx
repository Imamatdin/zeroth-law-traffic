import { useEffect, useMemo, useRef, useState } from 'react';
import { eventVar } from '../design/tokens.js';

const SIG_VAR = { red: 'var(--sig-red)', yellow: 'var(--sig-yellow)', green: 'var(--sig-green)' };

function useWidth(ref) {
  const [w, setW] = useState(800);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

const pretty = (s) => s.replace(/_/g, ' ');

export default function Timeline({ data, clock, onEvent, selectedEvent }) {
  const wrap = useRef(null);
  const head = useRef(null);
  const slider = useRef(null);
  const width = useWidth(wrap);
  const narrow = width < 560;
  const LW = narrow ? 78 : 128;
  const PW = Math.max(100, width - LW - 10);
  const D = data.duration;
  const X = (t) => LW + (t / D) * PW;

  const lanes = useMemo(() => {
    const m = new Map();
    for (const label of data.labels) {
      const ends = [];
      const evs = data.events.filter((e) => e.label === label).sort((a, b) => a.start - b.start);
      for (const e of evs) {
        let i = ends.findIndex((end) => end <= e.start);
        if (i < 0) i = ends.push(0) - 1;
        ends[i] = e.end;
        m.set(e.key, i);
      }
      m.set(label, Math.max(1, ends.length));
    }
    return m;
  }, [data]);

  const rows = useMemo(() => {
    let y = 20;
    const r = { axis: 0, signal: y };
    y += 30;
    r.events = data.labels.map((l) => {
      const at = y;
      y += Math.max(24, 8 + lanes.get(l) * 11);
      return [l, at];
    });
    r.field = y + 4;
    y += 30;
    r.risk = y + 4;
    y += 86;
    r.height = y;
    return r;
  }, [data.labels, lanes]);

  const riskPath = useMemo(() => {
    const { t, risk, raw } = data.risk;
    const Y = (v) => rows.risk + 70 - v * 70;
    const line = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(t[i]).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    return { risk: line(risk), raw: line(raw), Y };
  }, [data.risk, rows, PW, LW]);

  const fieldPath = useMemo(() => {
    const { kt, count } = data.field;
    const max = Math.max(1, ...count);
    const Y = (v) => rows.field + 22 - (v / max) * 20;
    let d = `M${X(kt[0])},${rows.field + 22}`;
    kt.forEach((t, i) => (d += `L${X(t).toFixed(1)},${Y(count[i]).toFixed(1)}`));
    d += `L${X(kt[kt.length - 1])},${rows.field + 22}Z`;
    return { d, max };
  }, [data.field, rows, PW, LW]);

  const peak = useMemo(() => {
    const { t, risk } = data.risk;
    let i = 0;
    risk.forEach((v, j) => v > risk[i] && (i = j));
    return { t: t[i], v: risk[i] };
  }, [data.risk]);

  useEffect(() => clock.subscribe((c) => {
    head.current?.setAttribute('transform', `translate(${X(c.t)},0)`);
    slider.current?.setAttribute('aria-valuenow', String(Math.round(c.t)));
    slider.current?.setAttribute('aria-valuetext', `${c.t.toFixed(1)} seconds`);
  }), [clock, PW, LW]);

  const seekFromPointer = (e) => {
    const b = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - b.left;
    clock.seek(((x - LW) / PW) * D);
  };
  const onKey = (e) => {
    const step = { ArrowLeft: -1, ArrowRight: 1, PageDown: -10, PageUp: 10 }[e.key];
    if (step) clock.seek(clock.t + step * (e.shiftKey ? 5 : 1));
    else if (e.key === 'Home') clock.seek(0);
    else if (e.key === 'End') clock.seek(D);
    else return;
    e.preventDefault();
  };

  const ticks = [];
  const every = narrow ? 30 : 10;
  for (let s = 0; s <= D; s += every) ticks.push(s);
  const sig = data.signal;

  return (
    <div className="timeline" ref={wrap}>
      <svg width={width} height={rows.height} role="group" aria-label="Replay timeline">
        <defs>
          <filter id="rough" x="-5%" y="-20%" width="110%" height="140%">
            <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="4" />
            <feDisplacementMap in="SourceGraphic" scale="2.2" />
          </filter>
          <pattern id="hatch-off" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="4" height="4" fill="var(--paper)" />
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-faint)" strokeWidth="1.4" />
          </pattern>
        </defs>

        <g
          className="tl-scrub"
          role="slider"
          tabIndex={0}
          aria-label="Replay time"
          aria-valuemin={0}
          aria-valuemax={Math.round(D)}
          ref={slider}
          aria-valuenow={Math.round(clock.t)}
          aria-valuetext={`${clock.t.toFixed(1)} seconds`}
          onKeyDown={onKey}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            seekFromPointer(e);
          }}
          onPointerMove={(e) => e.buttons && seekFromPointer(e)}
        >
          <rect x={LW} y={0} width={PW} height={rows.height} fill="transparent" />
        </g>
        <g className="tl-axis">
          {ticks.map((s) => (
            <g key={s} transform={`translate(${X(s)},0)`}>
              <line y1={14} y2={rows.height} stroke="var(--rule)" strokeDasharray="1 4" />
              <text y={11} textAnchor="middle" className="num">{s}</text>
            </g>
          ))}
          <text x={LW + PW} y={11} textAnchor="end" className="tl-unit">s</text>
        </g>

        <text x={0} y={rows.signal + 12} className="tl-label">median signal</text>
        <g filter="url(#rough)">
          {sig?.smoothed.map(([a, b, s], i) => (
            <rect key={i} x={X(a)} y={rows.signal} width={Math.max(1, X(b) - X(a))} height={14} fill={SIG_VAR[s] ?? 'url(#hatch-off)'} opacity={0.8}>
              <title>{`${s} ${a.toFixed(1)}–${b.toFixed(1)} s`}</title>
            </rect>
          ))}
        </g>
        {sig?.bridged.map(([a, b], i) => (
          <rect key={i} x={X(a)} y={rows.signal + 3} width={Math.max(2, X(b) - X(a))} height={8} fill="url(#hatch-off)" stroke="var(--ink)" strokeWidth={0.8}>
            <title>{`lamp read dark ${a.toFixed(2)}–${b.toFixed(2)} s; bridged by the smoothing Part A uses`}</title>
          </rect>
        ))}
        {data.crossings.map((c, i) => (
          <line key={i} x1={X(c.t)} x2={X(c.t)} y1={rows.signal + 17} y2={rows.signal + 23} stroke="var(--ink)" strokeWidth={1}>
            <title>{`track #${c.track} crossed the near stop line at ${c.t.toFixed(1)} s`}</title>
          </line>
        ))}

        {rows.events.map(([label, y]) => (
          <g key={label}>
            <text x={0} y={y + 4 + lanes.get(label) * 5.5 + 5} className="tl-label">{pretty(label)}</text>
            <line x1={LW} x2={LW + PW} y1={y + 4 + lanes.get(label) * 5.5} y2={y + 4 + lanes.get(label) * 5.5} stroke="var(--rule)" />
            {data.events.filter((e) => e.label === label).map((e) => {
              const x0 = X(e.start), x1 = Math.max(X(e.end), x0 + 5);
              const sel = selectedEvent?.key === e.key;
              const by = y + 4 + lanes.get(e.key) * 11;
              return (
                <g
                  key={e.key}
                  className="tl-event"
                  role="button"
                  tabIndex={0}
                  aria-label={`${pretty(e.label)}, ${e.start.toFixed(1)} to ${e.end.toFixed(1)} seconds, track ${e.track_ids.join(', ')}. Fly there.`}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    onEvent(e);
                  }}
                  onPointerDown={(ev) => ev.stopPropagation()}
                  onKeyDown={(ev) => {
                    if (ev.key === 'Enter' || ev.key === ' ') {
                      ev.preventDefault();
                      onEvent(e);
                    }
                  }}
                >
                  <rect x={x0} y={by} width={x1 - x0} height={9} rx={2} fill={eventVar(e.label)} opacity={sel ? 0.95 : 0.6} filter="url(#rough)" />
                  <rect x={x0} y={by - 3} width={x1 - x0} height={15} fill="transparent" />
                  <rect x={x0} y={by} width={x1 - x0} height={9} rx={2} fill="none" stroke="var(--ink)" strokeWidth={sel ? 2 : 1} filter="url(#rough)" />
                  <title>{`${pretty(e.label)} ${e.start.toFixed(1)}–${e.end.toFixed(1)} s · track #${e.track_ids.join(', #')}`}</title>
                </g>
              );
            })}
          </g>
        ))}

        <text x={0} y={rows.field + 14} className="tl-label">approaching pairs</text>
        <path d={fieldPath.d} fill="var(--field)" opacity={0.22} />
        <text x={LW + PW} y={rows.field + 6} textAnchor="end" className="tl-unit num">max {fieldPath.max}</text>

        <text x={0} y={rows.risk + 14} className="tl-label">risk</text>
        <text x={0} y={rows.risk + 30} className="tl-sub">P(accident ≤ 5 s)</text>
        <text x={0} y={rows.risk + 44} className="tl-sub tl-sub-raw">pair score</text>
        <rect x={LW} y={rows.risk} width={PW} height={70} fill="none" stroke="var(--rule)" />
        <line x1={LW} x2={LW + PW} y1={riskPath.Y(0.5)} y2={riskPath.Y(0.5)} stroke="var(--ink-soft)" strokeDasharray="5 4" />
        <text x={LW + 4} y={riskPath.Y(0.5) - 4} className="tl-unit">alarm threshold 0.5</text>
        <path d={riskPath.raw} fill="none" stroke="var(--ink-soft)" strokeWidth={1} strokeDasharray="1.5 2.5" />
        <path d={riskPath.risk} fill="none" stroke="var(--risk)" strokeWidth={1.8} />
        <text x={LW + PW} y={rows.risk + 82} textAnchor="end" className="tl-unit num">
          {`peak P ${peak.v.toFixed(3)} at ${peak.t.toFixed(1)} s · ${data.risk.alarm_starts.length} alarms`}
        </text>

        <g ref={head} transform={`translate(${X(clock.t)},0)`} pointerEvents="none">
          <line y1={14} y2={rows.height} stroke="var(--ink)" strokeWidth={1.6} filter="url(#rough)" />
          <path d="M-6,4 L6,4 L0,13 Z" fill="var(--ink)" />
        </g>
      </svg>
    </div>
  );
}
