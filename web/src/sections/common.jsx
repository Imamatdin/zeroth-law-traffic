// Shared pieces for the rubric sections.

// The hero registers its controls here so other sections can fly the scene to an event.
export const sceneApi = { flyTo: null };

export function showInScene(e) {
  sceneApi.flyTo?.(e);
  document.getElementById('scene')?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

export function Section({ id, n, title, lede, children }) {
  return (
    <section id={id} className="section" aria-labelledby={`${id}-title`}>
      <header className="section-head">
        <span className="section-n hand" aria-hidden="true">{n}</span>
        <h2 id={`${id}-title`}>{title}</h2>
        {lede && <p className="section-lede">{lede}</p>}
      </header>
      {children}
    </section>
  );
}

// A clearly marked gap: what is missing and where it will come from. Never filled with invented content.
export function Pending({ what, source }) {
  return (
    <div className="pending" role="note">
      <span className="pending-tag hand">to fill</span>
      <p>
        <strong>{what}</strong>
        {source && <span className="pending-src"> Source: {source}</span>}
      </p>
    </div>
  );
}

export function Figure({ title, finding, children, wide = false }) {
  return (
    <figure className={`fig${wide ? ' fig-wide' : ''}`}>
      <figcaption>
        <h3>{title}</h3>
        {finding && <p className="finding">{finding}</p>}
      </figcaption>
      {children}
    </figure>
  );
}

export const pretty = (s) => s.replace(/_/g, ' ');
export const CLASS_NAMES = ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck'];
