export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export const apiAvailable = () => API_BASE !== '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { signal: AbortSignal.timeout(120000), ...options });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail?.message ?? `Server returned HTTP ${response.status}. Please try again.`);
  return body;
}

export async function health() {
  const result = await request('/health');
  if (!result?.ready) throw new Error('The demo is still starting.');
  return result;
}

export async function submitVideo(file, onProgress) {
  const body = new FormData();
  body.append('video', file);
  const { job_id } = await request('/jobs', { method: 'POST', body, signal: AbortSignal.timeout(30 * 60 * 1000) });
  const deadline = Date.now() + 30 * 60 * 1000;
  while (Date.now() < deadline) {
    const s = await request(`/jobs/${job_id}`);
    onProgress?.(s);
    if (s.state === 'done') return request(`/jobs/${job_id}/result`);
    if (s.state === 'error') throw new Error(s.error?.message ?? 'Processing failed.');
    await new Promise((res) => setTimeout(res, 1500));
  }
  throw new Error('Processing is taking more than 30 minutes. The server may still be working.');
}
