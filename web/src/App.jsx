import { useEffect, useState } from 'react';
import Hero from './hero/Hero.jsx';
import { loadIndex, loadVideo } from './data/load.js';
import './app.css';

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadIndex()
      .then((idx) => loadVideo(idx.videos[0].id))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <header className="masthead">
        <a className="wordmark" href="#top">
          Zeroth Law Traffic
        </a>
        <span className="hand masthead-note">WIUT Hackathon 2026 · computer vision track</span>
      </header>
      <main id="top">
        {error ? (
          <p className="load-state">
            Could not load the exported data ({error}). Run <code>scripts/export_web.py</code> first.
          </p>
        ) : data ? (
          <Hero data={data} />
        ) : (
          <p className="load-state hand">unpacking the sketchbook…</p>
        )}
      </main>
    </>
  );
}
