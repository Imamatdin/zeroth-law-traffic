import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DioramaCanvas from './DioramaCanvas.jsx';
import Timeline from './Timeline.jsx';
import Notebook from './Notebook.jsx';
import { createClock, useClock } from './clock.js';
import { CLASS_NAMES } from '../scene/renderer.js';
import { eventVar } from '../design/tokens.js';
import './hero.css';

const LAYERS = [
  ['notes', 'Event notes'],
  ['trails', 'Trails'],
  ['field', 'Interaction field'],
  ['risk', 'Risk terrain'],
  ['atlas', 'Flow atlas'],
  ['follow', 'Follow events'],
  ['photo', 'Camera frame'],
];
const SPEEDS = [1, 2, 4, 0.5];
const pretty = (s) => s.replace(/_/g, ' ');

function PlayIcon({ playing }) {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      {playing ? (
        <g fill="currentColor">
          <path d="M4.2 2.6c1.1-.2 2.2-.1 3 .1l-.2 12.6c-1 .2-2 .2-2.9 0z" />
          <path d="M10.9 2.7c1-.2 2-.1 2.9 0l.1 12.6c-.9.3-2 .2-3 0z" />
        </g>
      ) : (
        <path d="M4.4 2.3c.4-.3 1-.2 1.4 0l9.2 5.9c.6.4.6 1.2 0 1.6L5.9 15.7c-.5.3-1.2.3-1.6-.1-.3-4.4-.3-8.9.1-13.3z" fill="currentColor" />
      )}
    </svg>
  );
}

function InkIcon({ d }) {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Hero({ data }) {
  const reducedMotion = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, []);
  const clock = useMemo(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.has('t') ? parseFloat(q.get('t')) : 8;
    return createClock(data.duration, { t, playing: !reducedMotion && !q.has('paused') });
  }, [data, reducedMotion]);
  const snap = useClock(clock, 6);
  const [layers, setLayers] = useState({ notes: true, trails: true, field: true, risk: true, atlas: false, follow: !reducedMotion, photo: false });
  const [selected, setSelected] = useState(null);
  const [event, setEvent] = useState(null);
  const [visible, setVisible] = useState([]);
  const rendererRef = useRef(null);
  const uiRef = useRef({ layers, selected });
  uiRef.current = { layers, selected };
  const heroRef = useRef(null);

  useEffect(() => {
    const id = setInterval(() => setVisible(rendererRef.current?.visibleIds() ?? []), 400);
    return () => clearInterval(id);
  }, []);

  const select = useCallback((id) => {
    setSelected(id);
    if (id != null) setEvent(null);
  }, []);
  const flyTo = useCallback((e) => {
    setSelected(null);
    setEvent(e);
    clock.seek(Math.max(0, e.start - 1.5));
    rendererRef.current?.focusEvent(e);
  }, [clock]);
  const clear = useCallback(() => {
    setSelected(null);
    setEvent(null);
  }, []);

  useEffect(() => {
    window.__zlt = { clock, setLayers, select, flyTo, data, renderer: rendererRef };
  }, [clock, select, flyTo, data]);

  const cycle = useCallback((dir) => {
    const ids = rendererRef.current?.visibleIds() ?? [];
    if (!ids.length) return;
    const i = ids.indexOf(selected);
    select(ids[(i + dir + ids.length) % ids.length]);
  }, [selected, select]);

  const onKey = (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = e.target.tagName;
    if (tag === 'TEXTAREA' || tag === 'SELECT' || (tag === 'INPUT' && e.target.type !== 'checkbox') || e.target.isContentEditable) return;
    const inControl = tag === 'INPUT' || tag === 'BUTTON' || e.target.getAttribute?.('role') === 'slider' || e.target.getAttribute?.('role') === 'button';
    const r = rendererRef.current;
    if (e.key === ' ' && !inControl) clock.set({ playing: !clock.playing });
    else if (e.key === 'ArrowRight' && !inControl) clock.seek(clock.t + (e.shiftKey ? 5 : 1));
    else if (e.key === 'ArrowLeft' && !inControl) clock.seek(clock.t - (e.shiftKey ? 5 : 1));
    else if (e.key === ']') cycle(1);
    else if (e.key === '[') cycle(-1);
    else if (e.key === 'Escape') clear();
    else if (e.key === '+' || e.key === '=') r?.zoomBy(1.25);
    else if (e.key === '-') r?.zoomBy(0.8);
    else if (e.key === '0') r?.resetView();
    else return;
    e.preventDefault();
  };

  const keyRef = useRef(onKey);
  keyRef.current = onKey;
  useEffect(() => {
    const h = (e) => keyRef.current(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);

  const sel = selected != null ? data.byId.get(selected) : null;
  const submitted = data.eventsMeta.submitted.length;
  const v = data.eda.video;
  const label = `Illustrated replay of camera ${data.id} at ${snap.t.toFixed(1)} seconds: ${visible.length} tracked road users in view${sel ? `, ${CLASS_NAMES[sel.cls]} #${sel.id} selected` : ''}.`;

  return (
    <section className="hero" ref={heroRef} aria-labelledby="hero-title">
      <div className="hero-intro">
        <h1 id="hero-title">One avenue, redrawn from what our tracker saw.</h1>
        <p className="hero-lede">
          Every figure below is a real track from sample <span className="num">{v.file}</span>{' '}
          (<span className="num">{v.duration_s.toFixed(1)} s, {v.width}×{v.height}, {v.fps} fps</span>),
          drawn where the camera saw it. Scrub the timeline, tap a car or a walker, or jump to an event.
        </p>
      </div>

      <div className="hero-grid">
        <div className="stage">
          <DioramaCanvas
            data={data}
            clock={clock}
            uiRef={uiRef}
            onSelect={select}
            rendererRef={rendererRef}
            reducedMotion={reducedMotion}
            label={label}
          />
          <div className="zoom" role="group" aria-label="View">
            <button type="button" onClick={() => rendererRef.current?.zoomBy(1.25)} aria-label="Zoom in">
              <InkIcon d="M9 3.2c.2 3.9 0 7.7-.1 11.6M3.3 9.1c3.8-.2 7.6-.1 11.4.1" />
            </button>
            <button type="button" onClick={() => rendererRef.current?.zoomBy(0.8)} aria-label="Zoom out">
              <InkIcon d="M3.3 9.1c3.8-.2 7.6-.1 11.4.1" />
            </button>
            <button type="button" onClick={() => rendererRef.current?.resetView()} aria-label="Show the whole scene">
              <InkIcon d="M3 6.5V3.2h3.4M11.6 3.1H15v3.3M15 11.5v3.3h-3.3M6.4 14.9H3v-3.4" />
            </button>
          </div>
        </div>
        <Notebook
          data={data}
          t={snap.t}
          selected={selected}
          event={event}
          visible={visible}
          onSelect={select}
          onEvent={flyTo}
          onClear={clear}
        />
      </div>

      <p className="hero-status">
        <span className="stamp">in-sample · provisional</span>
        Event notes come from a development replay with all engines on; with labels not yet locked the submission reports{' '}
        <span className="num">{submitted}</span> events for this clip.
      </p>

      <div className="deck">
        <div className="transport">
          <button type="button" className="play" onClick={() => clock.set({ playing: !clock.playing })} aria-label={snap.playing ? 'Pause' : 'Play'}>
            <PlayIcon playing={snap.playing} />
          </button>
          <button type="button" className="step" onClick={() => clock.seek(clock.t - 1)} aria-label="Back one second">−1 s</button>
          <button type="button" className="step" onClick={() => clock.seek(clock.t + 1)} aria-label="Forward one second">+1 s</button>
          <span className="clock num" aria-hidden="true">
            {snap.t.toFixed(1)} <span>/ {data.duration.toFixed(1)} s</span>
          </span>
          <button type="button" className="step" onClick={() => clock.set({ speed: SPEEDS[(SPEEDS.indexOf(clock.speed) + 1) % SPEEDS.length] })} aria-label={`Playback speed ${snap.speed} times`}>
            {snap.speed}×
          </button>
        </div>
        <fieldset className="layers">
          <legend className="visually-hidden">Layers</legend>
          {LAYERS.map(([k, name]) => (
            <label key={k} className="layer">
              <input type="checkbox" checked={layers[k]} onChange={() => setLayers((s) => ({ ...s, [k]: !s[k] }))} />
              <span className="box" aria-hidden="true" />
              {name}
            </label>
          ))}
        </fieldset>
      </div>

      <div className="events-row" role="group" aria-label="Jump to events">
        {data.labels.map((label) => {
          const evs = data.events.filter((e) => e.label === label).sort((a, b) => a.start - b.start);
          const next = evs.find((e) => e.start > snap.t + 0.5) ?? evs[0];
          const on = event?.label === label;
          return (
            <button
              type="button"
              key={label}
              className={`ev-chip${on ? ' is-on' : ''}`}
              style={{ '--c': eventVar(label) }}
              onClick={() => flyTo(next)}
              aria-label={`Next ${pretty(label)}: ${next.start.toFixed(1)} seconds, ${evs.length} in this clip`}
            >
              <span className="ev-name">{pretty(label)}</span>
              <span className="num">×{evs.length}</span>
              <span className="ev-next num">next {next.start.toFixed(1)} s</span>
            </button>
          );
        })}
      </div>

      <Timeline data={data} clock={clock} onEvent={flyTo} selectedEvent={event} />

      <p className="keys">
        Keys: <kbd>Space</kbd> play · <kbd>←</kbd>/<kbd>→</kbd> one second · <kbd>[</kbd>/<kbd>]</kbd> previous or next road user ·{' '}
        <kbd>+</kbd>/<kbd>−</kbd> zoom · <kbd>0</kbd> reset view · <kbd>Esc</kbd> clear. Drag the scene to pan.
      </p>
    </section>
  );
}
