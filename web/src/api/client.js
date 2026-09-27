// Slot for the live demo backend (FastAPI on a Hugging Face CPU Space), not built yet.
// Set VITE_API_BASE at build time; with it unset the demo section says the backend is offline.
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';

export const apiAvailable = () => API_BASE !== '';

export async function health() {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

// Expected contract: POST /jobs (multipart "video") -> {job_id}; GET /jobs/{id} -> {state, progress,
// result?: {events: [[start, end, label]], risk: [[t, score]], tracks_url?}}. Mirrors predictions.json.
export async function submitVideo(file, onProgress) {
  const body = new FormData();
  body.append('video', file);
  const r = await fetch(`${API_BASE}/jobs`, { method: 'POST', body });
  if (!r.ok) throw new Error(`upload ${r.status}`);
  const { job_id } = await r.json();
  for (;;) {
    const s = await (await fetch(`${API_BASE}/jobs/${job_id}`)).json();
    onProgress?.(s);
    if (s.state === 'done') return s.result;
    if (s.state === 'error') throw new Error(s.error ?? 'job failed');
    await new Promise((res) => setTimeout(res, 1500));
  }
}
