import { useEffect, useRef, useState } from 'react';
import { apiAvailable, health, submitVideo } from '../api/client.js';
import { LineChart } from '../charts/charts.jsx';
import { Section, pretty } from './common.jsx';

function Playback({ result, url }) {
  const video = useRef(null);
  const [time, setTime] = useState(0);
  const { replay, risk, events } = result;
  let index = 0;
  while (index + 1 < replay.t.length && replay.t[index + 1] <= time) index++;
  const visible = replay.tracks.flatMap((tr) => {
    const j = tr.k.indexOf(index);
    if (j < 0) return [];
    const q = replay.q;
    return [{ id: tr.id, label: replay.classes[tr.cls], x: (tr.x[j] - tr.w[j] / 2) / q * 100,
      y: (tr.y[j] - tr.h[j]) / q * 100, w: tr.w[j] / q * 100, h: tr.h[j] / q * 100 }];
  });
  const seek = (t) => { if (video.current) { video.current.currentTime = t; setTime(t); } };
  return <div className="demo-result">
    <h3>{events.submitted.length} events · {replay.tracks.length} tracks</h3>
    <p className="num">Processed {replay.duration.toFixed(1)} s in {result.metadata.processing_seconds.toFixed(1)} s ({result.metadata.wall_time_ratio.toFixed(2)}× video duration).</p>
    {result.metadata.warnings.map((warning) => <p className="note" key={warning}>{warning}</p>)}
    <div className="demo-playback">
      <video ref={video} src={url} controls playsInline preload="metadata" onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)} />
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {visible.map((b) => <g key={b.id}><rect x={b.x} y={b.y} width={b.w} height={b.h} /><text x={b.x} y={Math.max(2, b.y - .5)}>#{b.id} {b.label}</text></g>)}
      </svg>
    </div>
    <label className="demo-seek">Playback time: {time.toFixed(1)} s
      <input aria-label="Demo playback time" type="range" min="0" max={replay.duration} step="0.1" value={Math.min(time, replay.duration)} onChange={(e) => seek(Number(e.target.value))} />
    </label>
    <LineChart x={risk.t} y={risk.risk} yMax={1} yLabel="risk" ariaLabel="Demo risk over time" />
    <p className="muted">Risk is an uncalibrated demo estimate. This faster model may miss small or distant objects; it is separate from the frozen submission.</p>
    {events.submitted.length ? <ul>{events.submitted.map(([s, e, label], i) => <li key={i}><button className="ink-btn" onClick={() => seek(s)}>{pretty(label)} · {s.toFixed(1)}–{e.toFixed(1)} s</button></li>)}</ul>
      : <p>{result.metadata.scene_events_skipped ? 'Camera-specific event rules were skipped for this scene.' : 'No enabled event was detected in this excerpt.'}</p>}
  </div>;
}

export default function Demo() {
  const [status, setStatus] = useState(apiAvailable() ? 'checking' : 'offline');
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState(null);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const check = () => { setStatus('checking'); health().then(() => setStatus('online')).catch(() => setStatus('offline')); };
  useEffect(() => { if (apiAvailable()) check(); }, []);
  useEffect(() => {
    if (!file) { setUrl(null); return; }
    const next = URL.createObjectURL(file); setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  const run = async (ev) => {
    ev.preventDefault();
    if (!file || running) return;
    setError(null); setResult(null); setJob(null); setRunning(true);
    try { setResult(await submitVideo(file, setJob)); }
    catch (e) { setError(e.message ?? String(e)); }
    finally { setRunning(false); }
  };
  const online = status === 'online';
  return <Section id="demo" n="5" title="Try it on your own clip" lede="Upload an MP4 of up to 60 seconds. Follow detections on your video and inspect its risk curve. Camera-specific events are reported only when the view matches our reference camera.">
    <form className="demo" onSubmit={run}>
      <p className={`demo-status ${online ? 'is-on' : ''}`} role="status"><span className="dot" aria-hidden="true" />
        {online ? 'Backend online.' : status === 'checking' ? 'Checking the backend…' : 'Backend unavailable or waking up.'}
      </p>
      {!online && apiAvailable() && <button className="ink-btn" type="button" onClick={check}>Check again</button>}
      <label className="drop"><input aria-label="Upload MP4 video" type="file" accept="video/mp4" disabled={!online || running} onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); setError(null); setJob(null); }} />
        <span className="hand drop-h">{file ? file.name : 'choose an .mp4'}</span>
        <span className="muted">{file ? `${(file.size / 1e6).toFixed(1)} MB` : '60 seconds maximum · MP4 · up to 3 GiB'}</span>
      </label>
      <button type="submit" className="ink-btn" disabled={!online || !file || running}>{running ? 'Processing…' : 'Run the pipeline'}</button>
      {running && <p className="num" role="status">{job ? `${job.stage} · ${Math.round(job.progress * 100)}%` : 'Uploading…'}</p>}
      {error && <p className="demo-error" role="alert">{error}</p>}
    </form>
    {result && url && <Playback result={result} url={url} />}
    <p className="note">A CPU demo using YOLO11n, one shared pass and roughly eight analysis frames per second. One upload runs at a time. Uploads are deleted after processing; results expire after one hour or a server restart. The selected video plays locally in your browser.</p>
  </Section>;
}
