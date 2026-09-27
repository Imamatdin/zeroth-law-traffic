import { useMemo } from 'react';
import { stateAt } from '../data/load.js';
import { LineChart, BarList, Histogram, MiniMap, HeatCells, RampLegend, Stat, TableView } from '../charts/charts.jsx';
import { Section, Figure, CLASS_NAMES } from './common.jsx';

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
};

function inPoly([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// Pedestrian samples by where the 64x36 cell centre falls: crossing, refuge/median, carriageway or pavement.
function pedestrianPlaces(heat, cam) {
  const { nx, ny, pedestrians } = heat;
  const out = { crossing: 0, refuge: 0, carriageway: 0, pavement: 0 };
  const safe = [...(cam.islands ?? []), ...(cam.median ?? [])].map((s) => s.polygon);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = pedestrians[j][i];
      if (!v) continue;
      const p = [(i + 0.5) / nx, (j + 0.5) / ny];
      if ((cam.crosswalks ?? []).some((c) => inPoly(p, c.polygon))) out.crossing += v;
      else if (safe.some((s) => inPoly(p, s))) out.refuge += v;
      else if ((cam.sidewalks ?? []).some((s) => inPoly(p, s.polygon)) || !inPoly(p, cam.road)) out.pavement += v;
      else out.carriageway += v;
    }
  }
  return out;
}

function FlowArrows({ atlas, trajectories }) {
  const { nx, ny } = atlas.grid;
  const cw = 1920 / nx, ch = 1080 / ny;
  const min = atlas.params.min_samples;
  return (
    <g>
      {trajectories && atlas.movements.map((m, i) => (
        <line key={i} x1={m.start[0] * 1920} y1={m.start[1] * 1080} x2={m.end[0] * 1920} y2={m.end[1] * 1080} stroke="var(--data)" strokeWidth="2" opacity="0.25" />
      ))}
      {!trajectories && atlas.flow.count.flatMap((row, j) => row.map((n, i) => {
        const res = atlas.flow.resultant[j][i];
        if (n < min || res < 0.5) return null;
        const h = (atlas.flow.heading_deg[j][i] * Math.PI) / 180;
        const cx = (i + 0.5) * cw, cy = (j + 0.5) * ch, L = cw * 0.38 * res;
        const tx = cx + Math.cos(h) * L, ty = cy + Math.sin(h) * L;
        return (
          <g key={`${i}-${j}`} stroke="var(--data)" strokeWidth="4" strokeLinecap="round" fill="none" opacity={Math.min(0.95, 0.3 + Math.log10(n) * 0.2)}>
            <title>{`${Math.round(atlas.flow.heading_deg[j][i])}°, ${n} samples, resultant ${res.toFixed(2)}`}</title>
            <line x1={cx - Math.cos(h) * L} y1={cy - Math.sin(h) * L} x2={tx} y2={ty} />
            <polyline points={`${tx - Math.cos(h - 0.5) * L * 0.5},${ty - Math.sin(h - 0.5) * L * 0.5} ${tx},${ty} ${tx - Math.cos(h + 0.5) * L * 0.5},${ty - Math.sin(h + 0.5) * L * 0.5}`} />
          </g>
        );
      }))}
    </g>
  );
}

export default function EDA({ data }) {
  const e = data.eda;
  const v = e.video;
  const L = e.lighting;
  const lum = L.mean_luma, road = L.road_luma;
  const secs = e.counts_per_second.car.map((_, i) => i + 0.5);
  const cam = data.camera;
  const places = useMemo(() => pedestrianPlaces(e.heatmaps, cam), [e, cam]);
  const pedTotal = Object.values(places).reduce((a, b) => a + b, 0) || 1;
  const junctionShare = useMemo(() => {
    const { nx, ny, vehicles } = e.heatmaps;
    let inside = 0, all = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const n = vehicles[j][i];
      all += n;
      if (n && cam.intersection && inPoly([(i + 0.5) / nx, (j + 0.5) / ny], cam.intersection)) inside += n;
    }
    return all ? inside / all : 0;
  }, [e, cam]);
  const signal = data.signal;
  const byState = useMemo(() => {
    const out = {};
    for (const c of data.crossings) {
      const s = stateAt(signal.smoothed, c.t);
      out[s] = (out[s] ?? 0) + 1;
    }
    return out;
  }, [data]);
  const phases = signal.smoothed.map(([a, b, s]) => ({ s, a, b, d: b - a, cut: a === 0 || Math.abs(b - data.duration) < 0.05 }));
  const full = phases.filter((p) => !p.cut);
  const stops = data.atlas.dwell.stops;
  const longStops = stops.filter((s) => s.duration_s >= 10);
  const covered = data.atlas.coverage;
  const ib = data.atlas.interaction_baseline;
  const speedClasses = Object.entries(e.speed_rel.by_class);
  const meanCount = (c) => e.counts_per_second[c].reduce((a, b) => a + b, 0) / e.counts_per_second[c].length;

  return (
    <Section
      id="eda"
      n="3"
      title="What the sample shows"
      lede={`Exploratory analysis of ${v.file}, computed from our tracker's output on 1 in ${v.cache_stride} frames and one decoded frame per second. More sample clips are added here as their caches are exported.`}
    >
      <div className="stats">
        <Stat value={`${v.width}×${v.height}`} label="resolution" />
        <Stat value={`${v.fps}`} label="frames per second" />
        <Stat value={`${v.duration_s.toFixed(1)} s`} label="duration" note={`${v.n_frames} frames`} />
        <Stat value={`${v.bitrate_mbps} Mbit/s`} label="average bitrate" note={`${(v.size_mb / 1000).toFixed(2)} GB file`} />
        <Stat value={`${data.tracks.length}`} label="tracks replayed" note="after stitching" />
      </div>

      <div className="fig-grid">
        <Figure
          title="Lighting"
          finding={`Mean frame luma runs ${Math.min(...lum)}–${Math.max(...lum)} of 255 (road only: median ${median(road)}). This is a dusk clip, so the scene drawing lights headlights and windows.`}
        >
          <LineChart x={L.t} y={lum} yMax={255} height={150} yLabel="mean luma (0–255)" format={(q) => q.toFixed(0)} ariaLabel="Mean frame luma per second" />
          <TableView caption="Luma per second" columns={['t (s)', 'frame luma', 'road luma']} rows={L.t.map((t, i) => [t, lum[i], road[i]])} />
        </Figure>

        <Figure
          title="Road users in view"
          finding={`On average ${meanCount('car').toFixed(1)} cars and ${meanCount('person').toFixed(1)} people are tracked at any moment; each panel has its own scale.`}
          wide
        >
          <div className="multiples">
            {CLASS_NAMES.map((c) => (
              <div key={c} className="multiple">
                <p className="multiple-h">{c}</p>
                <LineChart x={secs} y={e.counts_per_second[c]} height={96} area format={(q) => q.toFixed(q < 3 ? 1 : 0)} ariaLabel={`${c}: tracked per second`} />
              </div>
            ))}
          </div>
          <TableView
            caption="Mean tracked per second"
            columns={['second', ...CLASS_NAMES]}
            rows={secs.map((s, i) => [Math.floor(s), ...CLASS_NAMES.map((c) => e.counts_per_second[c][i])])}
          />
        </Figure>

        <Figure title="Distinct tracks by class" finding="Counted after stitching; a track broken for longer than the stitching gap still counts twice.">
          <BarList items={CLASS_NAMES.map((c) => ({ label: c, value: e.unique_tracks[c] }))} ariaLabel="Distinct tracks by class" />
        </Figure>

        <Figure
          title="Where pedestrians are"
          finding={`By cell: ${Math.round((100 * places.crossing) / pedTotal)}% of pedestrian samples fall on a marked crossing, ${Math.round((100 * places.refuge) / pedTotal)}% on refuges or the median and ${Math.round((100 * places.carriageway) / pedTotal)}% on open carriageway: the pool the jaywalking rule works on.`}
        >
          <MiniMap camera={cam} ariaLabel="Pedestrian foot-point heatmap over the scene" legend={<RampLegend label="pedestrian samples" />}>
            <HeatCells grid={e.heatmaps.pedestrians} nx={e.heatmaps.nx} ny={e.heatmaps.ny} label="pedestrians" />
          </MiniMap>
        </Figure>

        <Figure title="Where vehicles are" finding={`${Math.round(100 * junctionShare)}% of vehicle samples fall inside the junction box traced in camera.yaml (by cell centre).`}>
          <MiniMap camera={cam} ariaLabel="Vehicle foot-point heatmap over the scene" legend={<RampLegend label="vehicle samples" />}>
            <HeatCells grid={e.heatmaps.vehicles} nx={e.heatmaps.nx} ny={e.heatmaps.ny} label="vehicles" />
          </MiniMap>
        </Figure>

        <Figure
          title="Lane directions, learned"
          finding={`The flow atlas holds a dominant heading in ${covered.covered_cells} of ${covered.total_cells} cells. camera.yaml's approach directions were cross-checked against it, and the wrong-way rule is judged against it.`}
        >
          <MiniMap camera={cam} ariaLabel="Flow atlas: dominant heading per cell">
            <FlowArrows atlas={data.atlas} />
          </MiniMap>
        </Figure>

        <Figure title="Trajectories" finding={`${data.atlas.movements.length} vehicle movements, start to end of each track.`}>
          <MiniMap camera={cam} ariaLabel="Vehicle trajectories from start to end">
            <FlowArrows atlas={data.atlas} trajectories />
          </MiniMap>
        </Figure>

        <Figure
          title="Where vehicles stand still"
          finding={`${stops.length} stops recorded; ${longStops.length} last 10 s or more, the stopped-vehicle threshold. The rule has to tell these apart from queues at the signal.`}
        >
          <MiniMap camera={cam} ariaLabel="Vehicle stops, sized by duration" legend="dot area ∝ stop duration">
            {stops.map((s, i) => (
              <circle key={i} cx={s.gx * 1920} cy={s.gy * 1080} r={4 + Math.sqrt(s.duration_s) * 3} fill="var(--data)" opacity="0.3">
                <title>{`track #${s.track_id}: ${s.duration_s.toFixed(1)} s`}</title>
              </circle>
            ))}
          </MiniMap>
        </Figure>

        <Figure
          title="Median signal and the near stop line"
          finding={`${data.crossings.length} vehicles crossed the near stop line: ${Object.entries(byState).map(([s, n]) => `${n} on ${s}`).join(', ')}. camera.yaml records this match as the evidence that this head governs the near approach.`}
        >
          <BarList items={Object.entries(byState).map(([s, n]) => ({ label: `on ${s}`, value: n }))} ariaLabel="Stop-line crossings by signal state" />
          <p className="note">
            Complete phases in this clip: {full.map((p) => `${p.s} ${p.d.toFixed(1)} s`).join(', ') || 'none'}. The first and last phases are cut by the clip edges.
          </p>
        </Figure>

        <Figure
          title="Speeds"
          finding="Image speed divided by each object's own box height, so the far road is not penalised; there is no metric calibration for this camera."
          wide
        >
          <div className="multiples">
            {speedClasses.map(([c, s]) => (
              <div key={c} className="multiple">
                <p className="multiple-h">
                  {c} <span className="num muted">n={s.n}</span>
                </p>
                <Histogram edges={e.speed_rel.edges} counts={s.hist} cap={4} marker={s.median} markerLabel={`median ${s.median}`} unit="box h/s" ariaLabel={`${c} speed distribution, median ${s.median} box heights per second`} />
              </div>
            ))}
          </div>
        </Figure>

        <Figure
          title="How close normal traffic gets"
          finding={`Across ${ib.n_pairs} neighbouring pairs, half never predict a pass closer than ${ib.dmin.p50.toFixed(2)} box heights, and 5% come within ${ib.dmin.p5.toFixed(2)}. Part B risk is measured against this baseline, not against zero.`}
        >
          <div className="table-scroll">
            <table className="ledger num-table">
              <thead>
                <tr><th scope="col">percentile</th><th scope="col">time to closest approach (s)</th><th scope="col">predicted miss distance (box h)</th></tr>
              </thead>
              <tbody>
                {['p1', 'p5', 'p10', 'p50'].map((p) => (
                  <tr key={p}><th scope="row">{p}</th><td className="num">{ib.tca[p].toFixed(2)}</td><td className="num">{ib.dmin[p].toFixed(2)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Figure>
      </div>
    </Section>
  );
}
