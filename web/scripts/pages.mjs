// Section screenshots and checks for the full page. Usage: node scripts/pages.mjs [baseUrl] [outDir]
// Kept lean (one context at a time, DPR 1) so it runs on a low-memory laptop.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://localhost:4173/';
const out = process.argv[3] ?? 'shots/pages';
mkdirSync(out, { recursive: true });
const SECTIONS = ['team', 'approach', 'eda', 'results', 'demo', 'report', 'links'];
const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
const results = [];
const errors = [];
const check = (name, ok, detail = '') => results.push(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);

for (const [tag, viewport, mobile] of [['desktop', { width: 1440, height: 900 }, false], ['phone', { width: 390, height: 844 }, true]]) {
  const ctx = await browser.newContext({ viewport, isMobile: mobile, hasTouch: mobile });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${tag}: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && errors.push(`${tag}: ${m.text()}`));
  await page.goto(`${base}?t=12.6&paused`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__zlt?.renderer?.current?.boxes?.length > 0);
  await page.evaluate(() => document.fonts.ready);
  const staticHeader = await page.addStyleTag({ content: '.masthead{position:static!important}' });
  for (const id of SECTIONS) {
    const el = page.locator(`#${id}`);
    check(`${tag}: section #${id} rendered`, (await el.count()) === 1);
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
    await el.screenshot({ path: `${out}/${tag}-${id}.png` });
  }
  await staticHeader.evaluate((el) => el.remove());
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(`${tag}: no horizontal page scroll`, overflow <= 0, `${overflow}px`);
  const pending = await page.locator('.pending').count();
  check(`${tag}: placeholders are marked`, pending > 0, `${pending} "to fill" notes`);

  await page.locator('nav a[href="#eda"]').click();
  await page.waitForTimeout(900);
  const edaTop = await page.evaluate(() => document.getElementById('eda').getBoundingClientRect().top);
  check(`${tag}: nav jumps to EDA below the sticky header`, edaTop > 0 && edaTop < 140, `top ${Math.round(edaTop)}px`);

  const btn = page.locator('#results tr', { hasText: 'red light' }).locator('.ink-btn');
  await btn.scrollIntoViewIfNeeded();
  await btn.click();
  await page.waitForTimeout(1500);
  const state = await page.evaluate(() => ({ t: window.__zlt.clock.t, top: document.getElementById('scene').getBoundingClientRect().top }));
  check(`${tag}: "show in scene" seeks 1.5 s before the red light and scrolls up`, Math.abs(state.t - (33.537 - 1.5)) < 0.05 && Math.abs(state.top) < 160, `t=${state.t.toFixed(2)}, scene top ${Math.round(state.top)}px`);
  await ctx.close();
}
await browser.close();
console.log(results.join('\n'));
console.log(errors.length ? `page errors:\n${errors.join('\n')}` : 'no page errors');
