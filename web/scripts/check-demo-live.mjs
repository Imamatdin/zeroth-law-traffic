// Real uploads through the rendered website. No API mocking.
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const base = process.argv[2] ?? 'https://zeroth-law-traffic.vercel.app';
const longRun = process.env.DEMO_TEST_LONG === '1';
const out = resolve(longRun ? '../private/live-demo-120s' : '../private/live-demo');
mkdirSync(out, { recursive: true });
const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${base}/?paused#demo`, { waitUntil: 'domcontentloaded' });
  await page.getByText('Backend online.', { exact: true }).waitFor({ timeout: 90000 });
  for (const [file, camera] of (longRun ? [['C3905-120s-upload.mp4', true]] : [['foreign-5s.mp4', false], ['C3905-5s-upload.mp4', true]])) {
    await page.getByLabel('Upload MP4 video').setInputFiles(resolve('../private/test-clips', file));
    const responsePromise = page.waitForResponse(r => r.url().includes('/jobs/') && r.url().endsWith('/result') && r.status() === 200, { timeout: 900000 });
    await page.getByRole('button', { name: 'Run the pipeline', exact: true }).click();
    const response = await Promise.race([responsePromise, page.locator('.demo-error').waitFor({ timeout: 900000 }).then(async () => { throw new Error(await page.locator('.demo-error').innerText()); })]);
    const result = await response.json();
    if (longRun) {
      assert.ok(result.replay.duration >= 119.9 && result.replay.duration <= 120);
      assert.ok(result.risk.t.at(-1) > 119, 'Processing must reach the end');
    }
    assert.equal(result.metadata.scene_check.matches, camera);
    assert.equal(result.metadata.scene_events_skipped, !camera);
    await page.locator('.demo-result').waitFor();
    assert.equal(await page.locator('.demo-playback video').count(), 1);
    assert.equal(await page.locator('.demo-result svg[role="img"]').count(), 1);
    assert.ok(await page.locator('.demo-playback rect').count() > 0);
    await page.getByLabel('Demo playback time').fill(longRun ? '119' : '1');
    await page.locator('#demo').screenshot({ path: `${out}/${file}.png`, timeout: 20000 });
    writeFileSync(`${out}/${file}.json`, JSON.stringify(result));
    console.log(JSON.stringify({ file, camera, seconds: result.metadata.processing_seconds,
      duration: result.replay.duration, tracks: result.replay.tracks.length,
      events: result.events.submitted.length, riskSamples: result.risk.t.length }));
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#demo').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#demo').evaluate(e => e.scrollWidth <= e.clientWidth + 1), true);
  await page.locator('#demo').screenshot({ path: `${out}/mobile.png`, timeout: 20000 });
  assert.deepEqual(errors, []);
  console.log('PASS: live website, real upload(s), correct scene gates, boxes, risk curve, seeking, mobile layout, no JS errors');
} finally { await browser.close(); }
