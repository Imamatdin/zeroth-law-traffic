import { poseAt, stateAt, floorIndex } from '../data/load.js';
import { eventVar } from '../design/tokens.js';
import { CLASS_NAMES, eventCaption } from '../scene/renderer.js';

const pretty = (s) => s.replace(/_/g, ' ');
const COMPASS = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗'];
const arrow = (deg) => COMPASS[((Math.round(deg / 45) % 8) + 8) % 8];

function fmtValue(key, v) {
  if (v == null) return '—';
  if (Array.isArray(v)) {
    if (v.length && typeof v[0] === 'object') return `${v.length} items`;
    return v.map((x) => (typeof x === 'number' ? +x.toFixed(2) : x)).join(', ');
  }
  if (typeof v === 'object') return `${Object.keys(v).length} fields`;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') {
    if (key === 'cls') return CLASS_NAMES[v] ?? v;
    const d = Math.abs(v) >= 100 ? 0 : 2;
    return `${+v.toFixed(d)}`;
  }
  return String(v);
}

function unitOf(key) {
  if (key.endsWith('_px')) return 'px';
  if (key.endsWith('_s') || ['crossing_t', 'stopped_from', 'green_at'].includes(key)) return 's';
  if (key === 'median_speed_rel') return 'box h/s';
  return '';
}

function atlasHeading(atlas, x, y) {
  const { nx, ny } = atlas.grid;
  const i = Math.floor((x / 1920) * nx), j = Math.floor((y / 1080) * ny);
  if (i < 0 || j < 0 || i >= nx || j >= ny) return null;
  const n = atlas.flow.count[j][i];
  if (n < atlas.params.min_samples || atlas.flow.resultant[j][i] < 0.5) return null;
  return { deg: atlas.flow.heading_deg[j][i], n };
}

function Rows({ rows }) {
  return (
    <dl className="nb-rows">
      {rows.map(([k, v, u]) => (
        <div key={k} className="nb-row">
          <dt>{k}</dt>
          <dd className="num">
            {v}
            {u ? <span className="nb-unit"> {u}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function TrackPage({ data, id, t, onEvent, onClear }) {
  const tr = data.byId.get(id);
  if (!tr) return null;
  const p = poseAt(tr, t);
  const events = data.events.filter((e) => e.track_ids.includes(id));
  const riskPairs = data.risk.pairs.filter((q) => q.a === id || q.b === id);
  const flow = p && tr.cls !== 0 ? atlasHeading(data.atlas, p.x, p.y) : null;
  const dev = flow && p ? Math.abs(((p.hd - flow.deg + 540) % 360) - 180) : null;
  const rows = [
    ['seen', `${tr.t0.toFixed(1)}–${tr.t1.toFixed(1)}`, 's'],
    ['raw tracklets', tr.raw.length > 1 ? `${tr.raw.length} stitched` : '1'],
  ];
  if (p) {
    rows.push(['speed now', p.v.toFixed(2), 'box h/s']);
    rows.push(['heading', `${arrow(p.hd)} ${Math.round(p.hd)}°`, 'image']);
    rows.push(['standing still', p.st.toFixed(1), 's']);
    if (flow) rows.push(['usual flow here', `${arrow(flow.deg)} ${Math.round(flow.deg)}°, off by ${Math.round(dev)}°`, `n=${flow.n}`]);
  } else rows.push(['now', 'not in view']);
  if (riskPairs.length) {
    const m = riskPairs.reduce((a, b) => (b.tca < a.tca ? b : a));
    rows.push(['closest risk pair', `tca ${m.tca.toFixed(1)} s at ${m.t.toFixed(1)}`, 's']);
  }
  return (
    <>
      <header className="nb-head">
        <p className="hand nb-kicker">road user</p>
        <h3>
          {pretty(CLASS_NAMES[tr.cls])} <span className="num">#{tr.id}</span>
        </h3>
        <button type="button" className="nb-clear" onClick={onClear}>
          clear
        </button>
      </header>
      <Rows rows={rows} />
      {events.length ? (
        <ul className="nb-links">
          {events.map((e) => (
            <li key={e.key}>
              <button type="button" onClick={() => onEvent(e)} style={{ '--c': eventVar(e.label) }}>
                {pretty(e.label)} <span className="num">{e.start.toFixed(1)} s</span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="nb-note">No event engine flagged this track.</p>
      )}
      <p className="nb-note">
        Speeds are image speeds divided by the object's own box height: there is no metric calibration for this camera.
      </p>
    </>
  );
}

function EventPage({ data, e, t, onSelect, onClear }) {
  const ev = Object.entries(e.evidence).filter(([k, v]) => k !== 'track_id' && !(Array.isArray(v) && v.length && typeof v[0] === 'object'));
  return (
    <>
      <header className="nb-head">
        <p className="hand nb-kicker" style={{ color: eventVar(e.label) }}>{eventCaption(e, t)}</p>
        <h3>{pretty(e.label)}</h3>
        <button type="button" className="nb-clear" onClick={onClear}>
          clear
        </button>
      </header>
      <Rows rows={[
        ['when', `${e.start.toFixed(2)}–${e.end.toFixed(2)}`, 's'],
        ['confidence', e.confidence.toFixed(2)],
        ...ev.map(([k, v]) => [pretty(k.replace(/_(s|px|t)$/, '')), fmtValue(k, v), unitOf(k)]),
      ]} />
      <ul className="nb-links">
        {e.track_ids.map((id) => (
          <li key={id}>
            <button type="button" onClick={() => onSelect(id)}>
              follow track <span className="num">#{id}</span>
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}

function NowPage({ data, t, visible, onEvent }) {
  const active = data.events.filter((e) => t >= e.start && t <= e.end);
  const raw = data.signal ? stateAt(data.signal.raw, t) : null;
  const smooth = data.signal ? stateAt(data.signal.smoothed, t) : null;
  const k = floorIndex(data.risk.t, t);
  const counts = {};
  for (const id of visible) {
    const c = CLASS_NAMES[data.byId.get(id)?.cls] ?? '?';
    counts[c] = (counts[c] ?? 0) + 1;
  }
  const fj = floorIndex(data.field.kt, t);
  const pairs = fj >= 0 ? data.field.byK.get(data.field.ks[fj]) : [];
  const soonest = pairs.length ? pairs.reduce((a, b) => (b.tca < a.tca ? b : a)) : null;
  return (
    <>
      <header className="nb-head">
        <p className="hand nb-kicker">field notes</p>
        <h3>
          at <span className="num">{t.toFixed(1)} s</span>
        </h3>
      </header>
      <Rows rows={[
        ['median signal', smooth === raw ? smooth : `${smooth} (lamp reads ${raw})`],
        ['in view', Object.entries(counts).map(([c, n]) => `${n} ${c}`).join(', ') || 'nobody'],
        ['approaching pairs', `${pairs.length}`],
        ['soonest closest approach', soonest ? `${soonest.tca.toFixed(1)} s, ${soonest.dmin.toFixed(2)} box h apart` : '—'],
        ['risk P(≤ 5 s)', k >= 0 ? data.risk.risk[k].toFixed(4) : '—'],
      ]} />
      {active.length ? (
        <ul className="nb-links">
          {active.map((e) => (
            <li key={e.key}>
              <button type="button" onClick={() => onEvent(e)} style={{ '--c': eventVar(e.label) }}>
                {pretty(e.label)} <span className="num">#{e.track_ids.join(', #')}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="nb-note">No event is running. Tap any car or walker to read its track.</p>
      )}
    </>
  );
}

export default function Notebook({ data, t, selected, event, visible, onSelect, onEvent, onClear }) {
  return (
    <aside className="notebook" aria-live="polite" aria-label="Field notes">
      {selected != null ? (
        <TrackPage data={data} id={selected} t={t} onEvent={onEvent} onClear={onClear} />
      ) : event ? (
        <EventPage data={data} e={event} t={t} onSelect={onSelect} onClear={onClear} />
      ) : (
        <NowPage data={data} t={t} visible={visible} onEvent={onEvent} />
      )}
    </aside>
  );
}
