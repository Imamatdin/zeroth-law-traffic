// Browser contract tests with explicitly mocked API responses; not a live benchmark.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://127.0.0.1:4174';
const out = process.argv[3] ?? '../private/browser-demo';
mkdirSync(out, { recursive: true });
const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let mode = 'generic';
    let polls = 0;
    const bundle = () => ({
      metadata: { processing_seconds: 1.25, wall_time_ratio: 1.25, scene_events_skipped: mode === 'generic',
        warnings: mode === 'generic' ? ['Camera mismatch: scene-dependent events disabled.'] : [] },
      replay: { duration: 1, q: 10000, t: [0, .5], classes: ['person'],
        tracks: [{ id: 1, cls: 0, k: [0, 1], x: [3000, 4000], y: [7000, 7000], w: [1000, 1000], h: [2000, 2000] }] },
      risk: { t: [0, .5], risk: [.1, .2] },
      events: { submitted: mode === 'camera' ? [[0, .5, 'jaywalking']] : [] },
    });
    await page.route('http://127.0.0.1:7860/**', async route => {
      const path = new URL(route.request().url()).pathname;
      if (mode === 'failure' && path === '/jobs') return route.fulfill({ status: 409, json: { detail: { code: 'busy', message: 'One job is already running.' } } });
      if (path === '/health') return route.fulfill({ json: { ready: true, max_seconds: 60 } });
      if (path === '/jobs') { polls = 0; return route.fulfill({ status: 202, json: { job_id: 'test' } }); }
      if (path.endsWith('/result')) return route.fulfill({ json: bundle() });
      return route.fulfill({ json: ++polls === 1 ? { state: 'running', stage: 'detecting', progress: .5 } : { state: 'done', result_url: '/jobs/test/result' } });
    });
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.getByText('Backend online.', { exact: true }).waitFor();
    const upload = async () => {
      await page.getByLabel('Upload MP4 video').setInputFiles({ name: 'browser-contract-fixture.mp4', mimeType: 'video/mp4', buffer: Buffer.from('mocked API test; not a real video') });
      await page.getByRole('button', { name: 'Run the pipeline', exact: true }).click();
    };
    await upload();
    await page.getByText('Camera-specific event rules were skipped for this scene.', { exact: true }).waitFor();
    assert.equal(await page.locator('.demo-playback rect').count(), 1);
    assert.equal(await page.locator('.demo-result svg[role="img"]').count(), 1);
    await page.locator('#demo').screenshot({ path: `${out}/demo-generic-${width}.png` });
    mode = 'camera'; await upload();
    await page.getByRole('button', { name: /jaywalking ·/ }).waitFor();
    await page.getByRole('button', { name: /jaywalking ·/ }).click();
    mode = 'failure'; await upload();
    await page.getByRole('alert').filter({ hasText: 'One job is already running.' }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Run the pipeline', exact: true }).isEnabled(), true);
    const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth,
      sections: [...document.querySelectorAll('section')].map(e => [e.id, e.clientWidth, e.scrollWidth]) }));
    assert.equal(overflow.scroll > overflow.width + 1, false, JSON.stringify(overflow));
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: generic result, camera events, result fetch, busy error, retry, no overflow or JS errors`);
    await page.close();
  }
} finally { await browser.close(); }
