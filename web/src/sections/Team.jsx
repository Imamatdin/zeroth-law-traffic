import { Section, Pending } from './common.jsx';

// Lanes as assigned in docs/TEAM_BRIEF.md. What each person actually delivered is confirmed from their
// contrib/<name>/bio.md at hand-in; until then this lists assignments, not credits.
const PEOPLE = [
  {
    name: 'Iko',
    role: 'Lead',
    lane: 'Owns the single codebase: perception pipeline, event rules, flow atlas, Part B risk, website and the final submission.',
    links: [['GitHub', 'https://github.com/Imamatdin']],
  },
  {
    name: 'Jalol',
    role: 'Perception and hazards',
    lane: 'Event annotation and cross-review, detector benchmark, tracker stress renders, fire/smoke detector, conditional detector fine-tune, blind install QA.',
    links: [],
  },
  {
    name: 'Javohir',
    role: 'Scene and learned signals',
    lane: 'Camera map draft, event annotation and cross-review, traffic-light state classifier, dataset licence table, near-miss/accident classifier, blind install QA.',
    links: [],
  },
];

export default function Team() {
  return (
    <Section id="team" n="1" title="Team" lede="Three people, one codebase. Teammates deliver into their own folders; the lead pulls their work into the pipeline through fixed interfaces.">
      <ol className="cast">
        {PEOPLE.map((p) => (
          <li key={p.name} className="cast-row">
            <div className="cast-name">
              <h3>{p.name}</h3>
              <p className="hand cast-role">{p.role}</p>
            </div>
            <p className="cast-lane">
              <span className="cast-k">assigned lane</span> {p.lane}
            </p>
            <div className="cast-links">
              {p.links.map(([k, href]) => (
                <a key={k} href={href} rel="noopener noreferrer" target="_blank">{k}</a>
              ))}
            </div>
          </li>
        ))}
      </ol>
      <Pending
        what="Full names, photos, LinkedIn and portfolio links, previous projects, and what each person actually delivered."
        source="each member's contrib/<name>/bio.md (handoff H10)."
      />
    </Section>
  );
}
