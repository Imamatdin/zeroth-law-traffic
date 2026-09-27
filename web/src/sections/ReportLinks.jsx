import { Section, Pending } from './common.jsx';

const REPO = 'https://github.com/Imamatdin/zeroth-law-traffic';

export function Report() {
  return (
    <Section id="report" n="6" title="Report" lede="One page: what worked, what did not, and what we would do next. Written at the architecture freeze, from the experiment records.">
      <div className="report-cols">
        {['What worked', 'What did not', 'What we would do next'].map((h) => (
          <div key={h}>
            <h3 className="sub-h">{h}</h3>
            <Pending what="Not written yet." source="experiments/ records and the dev-label scores, at the architecture freeze (G9)." />
          </div>
        ))}
      </div>
    </Section>
  );
}

export function Links({ data }) {
  const base = `${import.meta.env.BASE_URL}data`;
  return (
    <Section id="links" n="7" title="Links">
      <ul className="links">
        <li>
          <span className="links-k">Repository</span>
          <a href={REPO} rel="noopener noreferrer" target="_blank">{REPO.replace('https://', '')}</a>
        </li>
        <li>
          <span className="links-k">Data behind this page</span>
          <span>
            <a href={`${base}/index.json`}>index.json</a>
            {' · '}
            {['events', 'risk', 'replay', 'eda'].map((f, i) => (
              <span key={f}>
                {i ? ' · ' : ''}
                <a href={`${base}/${data.id}/${f}.json`}>{`${data.id}/${f}.json`}</a>
              </span>
            ))}
            <span className="muted"> (pipeline commit <span className="num">{data.pipelineCommit?.slice(0, 7) ?? 'unknown'}</span>)</span>
          </span>
        </li>
        <li>
          <span className="links-k">Model weights</span>
          <Pending what="GitHub Release with the weights and weights/download.sh (SHA256-checked)." source="packaging step 11.4." />
        </li>
        <li>
          <span className="links-k">predictions_samples.json</span>
          <Pending what="Our output on the sample videos from the frozen commit." source="run_submission.py at the submission tag (step 11.7)." />
        </li>
      </ul>
    </Section>
  );
}
