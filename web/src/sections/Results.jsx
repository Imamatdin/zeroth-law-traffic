import { eventVar } from '../design/tokens.js';
import { Stat, TableView } from '../charts/charts.jsx';
import { Section, Pending, pretty, showInScene } from './common.jsx';

export default function Results({ data }) {
  const meta = data.eventsMeta;
  const r = data.risk;
  const events = [...data.events].sort((a, b) => a.start - b.start);
  const perClass = data.labels.map((l) => [l, events.filter((e) => e.label === l).length]);
  return (
    <Section
      id="results"
      n="4"
      title="Results on the sample videos"
      lede="What our pipeline reports for each sample clip. Everything here is in-sample and provisional until the team's own labels are locked; nothing is scored yet."
    >
      <article className="video-result">
        <header className="vr-head">
          <h3 className="num">{data.eda.video.file}</h3>
          <p className="muted">{meta.status}</p>
        </header>
        <div className="stats">
          <Stat value={`${events.length}`} label="events in the development replay" note={`${meta.segments.length} after merging`} />
          <Stat value={`${meta.submitted.length}`} label="events in the submission" note="every class gated off for now" />
          <Stat value={r.percentiles.risk['100'].toFixed(3)} label="peak risk P(accident ≤ 5 s)" note={`threshold ${r.threshold}`} />
          <Stat value={`${r.alarm_starts.length}`} label="alarms raised" note="no annotated accident in this clip" />
        </div>

        <p className="note">
          The development replay runs every engine, including disabled ones, so we can inspect them: {perClass.map(([l, n]) => `${n} ${pretty(l)}`).join(', ')}.
          Use the buttons to watch each one in the scene above; the timeline and risk curve for this clip sit under the scene.
        </p>

        <div className="table-scroll">
          <table className="ledger events-ledger">
            <thead>
              <tr>
                <th scope="col">class</th>
                <th scope="col">start (s)</th>
                <th scope="col">end (s)</th>
                <th scope="col">length (s)</th>
                <th scope="col">tracks</th>
                <th scope="col">confidence</th>
                <th scope="col"><span className="visually-hidden">action</span></th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.key}>
                  <th scope="row">
                    <span className="ev-dot" style={{ background: eventVar(e.label) }} aria-hidden="true" />
                    {pretty(e.label)}
                  </th>
                  <td className="num">{e.start.toFixed(2)}</td>
                  <td className="num">{e.end.toFixed(2)}</td>
                  <td className="num">{(e.end - e.start).toFixed(1)}</td>
                  <td className="num">#{e.track_ids.join(', #')}</td>
                  <td className="num">{e.confidence.toFixed(2)}</td>
                  <td>
                    <button type="button" className="ink-btn" onClick={() => showInScene(e)}>
                      show in scene
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <TableView
          caption="Risk percentiles"
          summary="Risk percentiles for this clip"
          columns={['percentile', 'pair score (raw)', 'calibrated P']}
          rows={Object.keys(r.percentiles.risk).map((q) => [q, r.percentiles.raw[q], r.percentiles.risk[q]])}
        />

        <Pending what="Annotated render of this clip (boxes, track ids, event labels) to play beside the drawing." source="scripts/render_review.py output, hosted with the site." />
        <Pending what="Scores against our own labels: F1 per class at tIoU 0.3 / 0.5 / 0.7, Score B." source="scripts/dev_eval.py once dev labels are locked (gate G1)." />
        <Pending what="Honest failure cases, with frames." source="error analysis against the locked dev labels." />
      </article>

      <Pending what="The remaining sample clips, each with the same timeline, risk curve and event list." source="their perception caches, exported with scripts/export_web.py." />
    </Section>
  );
}
