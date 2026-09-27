import { useWidth } from '../charts/charts.jsx';
import { Section, Pending, pretty } from './common.jsx';

function Box({ x, y, w, h, title, sub, tone = 'ink' }) {
  return (
    <g transform={`translate(${x},${y})`}>
      <rect width={w} height={h} rx={6} className={`pd-box pd-${tone}`} filter="url(#pd-rough)" />
      <text x={w / 2} y={sub ? h / 2 - 4 : h / 2 + 5} textAnchor="middle" className="pd-title">{title}</text>
      {sub && <text x={w / 2} y={h / 2 + 14} textAnchor="middle" className="pd-sub">{sub}</text>}
    </g>
  );
}

function Arrow({ d }) {
  return <path d={d} className="pd-arrow" markerEnd="url(#pd-head)" filter="url(#pd-rough)" />;
}

function Diagram({ v }) {
  const [ref, width] = useWidth(900);
  const det = v.pipeline.detector, trk = v.pipeline.tracker;
  const a = [
    ['video', '.mp4, decoded once'],
    ['YOLO11m', `${det.imgsz} px, conf ${det.conf}, 1 in ${v.cache_stride} frames`],
    ['ByteTrack', `buffer ${trk.buffer_s} s`],
    ['stitching', 'tracklets + rider fusion'],
    ['world model', 'kinematics, tca, dmin'],
    ['event engines', 'rules × flow atlas'],
    ['segments', '[start, end, label]'],
  ];
  const b = [
    ['frame t', 'from the harness'],
    ['own detector + tracker', 'no Part A output'],
    ['causal risk', 'tca, dmin, closing, signal'],
    ['smoother', 'fast rise, slow decay'],
    ['calibrator', 'monotonic'],
    ['P(accident ≤ 5 s)', 'per frame'],
  ];
  const narrow = width < 760;
  if (narrow) {
    const bw = Math.min(width - 24, 300), bh = 48, gap = 22;
    const col = (items, x0, tone) => items.map(([t, s], i) => <Box key={t} x={x0} y={40 + i * (bh + gap)} w={bw} h={bh} title={t} sub={s} tone={tone} />);
    const arrows = (n, x0) => Array.from({ length: n - 1 }, (_, i) => <Arrow key={i} d={`M${x0 + bw / 2},${40 + i * (bh + gap) + bh} v${gap - 4}`} />);
    const hA = 40 + a.length * (bh + gap);
    return (
      <div ref={ref} className="pipeline">
        <svg width={width} height={hA + 40 + b.length * (bh + gap)} role="img" aria-label="Pipeline diagram: Part A events and Part B causal risk">
          <Defs />
          <text x={12} y={24} className="pd-lane">Part A · events (may look ahead)</text>
          {col(a, 12, 'ink')}
          {arrows(a.length, 12)}
          <g transform={`translate(0,${hA})`}>
            <text x={12} y={24} className="pd-lane pd-lane-b">Part B · risk (causal, frame by frame)</text>
            {col(b, 12, 'risk')}
            {arrows(b.length, 12)}
          </g>
        </svg>
      </div>
    );
  }
  const n = a.length, gap = 18;
  const bw = (width - 24 - gap * (n - 1)) / n, bh = 58;
  const xa = (i) => 12 + i * (bw + gap);
  const bwB = (width - 24 - gap * (b.length - 1)) / b.length;
  const xb = (i) => 12 + i * (bwB + gap);
  return (
    <div ref={ref} className="pipeline">
      <svg width={width} height={330} role="img" aria-label="Pipeline diagram: Part A events and Part B causal risk">
        <Defs />
        <text x={12} y={22} className="pd-lane">Part A · events (may look ahead)</text>
        <Box x={xa(5) - 10} y={32} w={bw * 0.9} h={30} title="camera.yaml · law" tone="soft" />
        <Box x={xa(5) + bw * 0.9} y={32} w={bw * 0.9} h={30} title="atlas · normal" tone="soft" />
        <Arrow d={`M${xa(5) + bw * 0.35},62 v26`} />
        <Arrow d={`M${xa(5) + bw * 1.2},62 C${xa(5) + bw * 1.2},80 ${xa(5) + bw * 0.8},76 ${xa(5) + bw * 0.7},90`} />
        {a.map(([t, s], i) => <Box key={t} x={xa(i)} y={94} w={bw} h={bh} title={t} sub={s} />)}
        {a.slice(1).map((_, i) => <Arrow key={i} d={`M${xa(i) + bw + 2},${94 + bh / 2} h${gap - 6}`} />)}
        <text x={12} y={206} className="pd-lane pd-lane-b">Part B · risk (causal, frame by frame; never reads the file or Part A)</text>
        {b.map(([t, s], i) => <Box key={t} x={xb(i)} y={220} w={bwB} h={bh} title={t} sub={s} tone="risk" />)}
        {b.slice(1).map((_, i) => <Arrow key={i} d={`M${xb(i) + bwB + 2},${220 + bh / 2} h${gap - 6}`} />)}
      </svg>
    </div>
  );
}

function Defs() {
  return (
    <defs>
      <filter id="pd-rough">
        <feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="2" seed="7" />
        <feDisplacementMap in="SourceGraphic" scale="2.5" />
      </filter>
      <marker id="pd-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M1,1 L9,5 L1,9" fill="none" stroke="var(--ink)" strokeWidth="1.6" strokeLinecap="round" />
      </marker>
    </defs>
  );
}

export default function Approach({ data }) {
  const v = data.eda.video;
  const meta = data.eventsMeta;
  const engines = meta.enabled_in_submission;
  const w = v.weights?.[0];
  const rows = [
    ['Object detector', 'learned', `Ultralytics YOLO11m (${w?.file}, sha256 ${w?.sha256.slice(0, 12)}…), ${v.pipeline.detector.imgsz} px on a ${v.pipeline.detector.prescale_width} px downscale`, 'shipped'],
    ['Tracker', 'algorithm', `ByteTrack, high/low thresholds ${v.pipeline.tracker.track_high_thresh}/${v.pipeline.tracker.track_low_thresh}, ${v.pipeline.tracker.buffer_s} s buffer`, 'shipped'],
    ['Tracklet stitching, rider fusion', 'rules', 'joins broken tracks by gap, distance and heading; folds riders into their bike (Part A only)', 'shipped'],
    ['Traffic-light state', 'rules', 'HSV lamp-cell classifier on the signal head located in camera.yaml', 'shipped'],
    ['Flow atlas', 'statistics, no labels', `flow, dwell and interaction percentiles from ${data.atlas.interaction_baseline.n_pairs} normal pairs on the sample tracks`, 'shipped'],
    ['Event engines', 'rules', `per-class state machines on tracks + camera.yaml, some cross-checked against the atlas (wrong-way); ${Object.keys(engines).length} of ${meta.classes.length} classes have an engine`, `${Object.values(engines).filter(Boolean).length} enabled`],
    ['Part B risk', 'analytic + calibration', 'time to closest approach, predicted miss distance, closing speed, braking, signal and wrong-way cues; causal smoother; monotonic calibrator', 'shipped'],
    ['Near-miss / accident classifier', 'learned', 'gradient-boosted trees on window features', 'planned'],
    ['Fire / smoke detector', 'learned', 'small detector trained on licence-cleared data', 'planned'],
  ];
  return (
    <Section
      id="approach"
      n="2"
      title="Problem and approach"
      lede="A fixed camera, a handful of unlabelled clips and a hidden test set with events the samples do not contain. We keep an explicit world of tracks and pairs, write each traffic law as a rule on that world, and let a flow atlas learned from normal traffic say what is ordinary here."
    >
      <Diagram v={v} />
      <div className="two-col">
        <div>
          <h3 className="sub-h">What is learned and what is a rule</h3>
          <div className="table-scroll">
            <table className="ledger">
              <thead>
                <tr><th scope="col">Component</th><th scope="col">Kind</th><th scope="col">Detail</th><th scope="col">Status</th></tr>
              </thead>
              <tbody>
                {rows.map(([c, k, d, s]) => (
                  <tr key={c}>
                    <th scope="row">{c}</th>
                    <td><span className={`kind kind-${k.split(' ')[0].replace(/[^a-z]/g, '')}`}>{k}</span></td>
                    <td>{d}</td>
                    <td className={s === 'planned' ? 'muted' : ''}>{s}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h3 className="sub-h">The fourteen official classes</h3>
          <ul className="class-grid">
            {meta.classes.map((c) => {
              const has = c in engines;
              return (
                <li key={c} className={has ? 'has' : ''}>
                  <span>{pretty(c)}</span>
                  <span className="num">{has ? (engines[c] ? 'engine · on' : 'engine · off') : 'no engine yet'}</span>
                </li>
              );
            })}
          </ul>
          <p className="note">
            A class stays off in the submission until it shows near-zero false fires on all sample footage: the scorer averages F1 over
            every class we predict, so a false class costs as much as a missed one.
          </p>
        </div>
      </div>
      <Pending what="Datasets used for training, with licences." source="DATASETS.md from the dataset licence task (handoff H6)." />
      <Pending what="Ablations: detector, frame stride, tracker, atlas on/off, rules versus learned." source="experiments/EXP-*.json once dev labels are locked." />
    </Section>
  );
}
