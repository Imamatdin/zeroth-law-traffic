import { useEffect, useState } from 'react';
import { API_BASE, apiAvailable, health, submitVideo } from '../api/client.js';
import { Section, Pending, pretty } from './common.jsx';

export default function Demo() {
  const [status, setStatus] = useState(apiAvailable() ? 'checking' : 'offline');
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!apiAvailable()) return;
    health().then(() => setStatus('online')).catch(() => setStatus('offline'));
  }, []);

  const run = async (ev) => {
    ev.preventDefault();
    if (!file) return;
    setError(null);
    setResult(null);
    try {
      setResult(await submitVideo(file, setJob));
    } catch (e) {
      setError(String(e.message ?? e));
    }
  };

  const online = status === 'online';
  return (
    <Section
      id="demo"
      n="5"
      title="Try it on your own clip"
      lede="Once the demo server is up, you will be able to upload a clip from this camera and get its events and risk curve back, drawn like the scene above. It is planned to run on a CPU, so expect it to take a while."
    >
      <form className="demo" onSubmit={run}>
        <p className={`demo-status ${online ? 'is-on' : ''}`} role="status">
          <span className="dot" aria-hidden="true" />
          {online ? 'Backend online.' : status === 'checking' ? 'Checking the backend…' : 'Backend offline: the demo server is not deployed yet.'}
        </p>
        <label className="drop">
          <input type="file" accept="video/mp4" disabled={!online} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <span className="hand drop-h">{file ? file.name : 'choose an .mp4'}</span>
          <span className="muted">{file ? `${(file.size / 1e6).toFixed(1)} MB` : 'from the same fixed camera'}</span>
        </label>
        <button type="submit" className="ink-btn" disabled={!online || !file}>Run the pipeline</button>
        {job && !result && (
          <p className="num" role="status">
            {job.state}{job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : ''}
          </p>
        )}
        {error && <p className="demo-error" role="alert">{error}</p>}
        {result && (
          <div className="demo-result">
            <h3>{result.events.length} events</h3>
            <ul>
              {result.events.map(([s, e, l], i) => (
                <li key={i} className="num">{`${pretty(l)} ${s.toFixed(1)}–${e.toFixed(1)} s`}</li>
              ))}
            </ul>
          </div>
        )}
      </form>
      <Pending
        what="Accepted size and length, the backend itself, and the drawn result view (timeline, annotated playback, risk curve) for an uploaded clip."
        source={`FastAPI on a Hugging Face CPU Space; the site calls ${API_BASE || 'VITE_API_BASE'} (see src/api/client.js).`}
      />
    </Section>
  );
}
