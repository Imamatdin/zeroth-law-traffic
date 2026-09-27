// Interaction check for the hero: selection by click and keyboard, event fly-to, timeline keyboard scrub,
// layer toggle. Prints what the page reports and saves screenshots. Usage: node scripts/interact.mjs [baseUrl] [outDir]
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://localhost:4173/';
const out = process.argv[3] ?? 'shots/interact';
mkdirSync(out, { recursive: true });
const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
const errors = [];
const results = [];
const check = (name, ok, detail) => results.push(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);

async function open(viewport, t, mobile = false) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${base}?t=${t}&paused`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__zlt?.renderer?.current?.boxes?.length > 0);
  await page.waitForTimeout(2200);
  return { ctx, page };
}

const notebook = (page) => page.locator('.notebook').innerText();
const clockT = (page) => page.evaluate(() => window.__zlt.clock.t);

{
  const { ctx, page } = await open({ width: 1440, height: 900 }, 40);
  const canvas = page.locator('.diorama canvas');
  const cb = await canvas.boundingBox();
  const target = await page.evaluate(() => {
    const r = window.__zlt.renderer.current;
    const d = window.__zlt.data;
    const cars = r.boxes.filter((b) => d.byId.get(b.id).cls === 3 && b.x1 - b.x0 > 30 && b.x0 > 0 && b.x1 < r.vw && b.y0 > 0 && b.y1 < r.vh);
    const b = cars[0];
    return b && { id: b.id, x: (b.x0 + b.x1) / 2, y: (b.y0 * 0.3 + b.y1 * 0.7) };
  });
  await page.mouse.click(cb.x + target.x, cb.y + target.y);
  await page.waitForTimeout(600);
  let nb = await notebook(page);
  check('click selects the car under the pointer', nb.includes(`#${target.id}`), `clicked #${target.id}; notebook: ${nb.split('\n').slice(0, 3).join(' | ')}`);
  await page.screenshot({ path: `${out}/select-car.png` });

  await page.locator('.notebook').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press(']');
  await page.waitForTimeout(400);
  const nb2 = await notebook(page);
  check('] cycles to another road user', nb2 !== nb && /#\d+/.test(nb2), nb2.split('\n').slice(0, 2).join(' | '));
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);
  check('Esc clears the selection', (await notebook(page)).includes('field notes'));

  await page.locator('.tl-event[aria-label^="red light"]').first().click();
  await page.waitForTimeout(2500);
  const t = await clockT(page);
  nb = await notebook(page);
  check('clicking the red-light bar seeks 1.5 s before it', Math.abs(t - (33.537 - 1.5)) < 0.05, `t=${t.toFixed(2)}`);
  check('notebook shows the red-light evidence', /red light/.test(nb) && /signal state/.test(nb), nb.split('\n').slice(0, 4).join(' | '));
  await page.evaluate(() => window.__zlt.clock.seek(34.2));
  await page.waitForTimeout(2200);
  await page.screenshot({ path: `${out}/fly-red-light.png` });

  const slider = page.locator('.tl-scrub');
  await slider.focus();
  const t0 = await clockT(page);
  await page.keyboard.press('ArrowRight');
  const t1 = await clockT(page);
  check('timeline slider: ArrowRight advances one second', Math.abs(t1 - t0 - 1) < 1e-6, `${t0.toFixed(2)} -> ${t1.toFixed(2)}`);

  await page.getByLabel('Flow atlas').check();
  await page.evaluate(() => window.__zlt.renderer.current.resetView());
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/atlas.png` });
  check('flow atlas toggles on', await page.getByLabel('Flow atlas').isChecked());
  await ctx.close();
}

{
  const { ctx, page } = await open({ width: 390, height: 844 }, 12.6, true);
  const canvas = page.locator('.diorama canvas');
  await canvas.scrollIntoViewIfNeeded();
  const cb = await canvas.boundingBox();
  const target = await page.evaluate(() => {
    const r = window.__zlt.renderer.current;
    const b = r.boxes.find((q) => q.id === 18);
    return b && { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2 };
  });
  if (target) {
    await page.touchscreen.tap(cb.x + target.x, cb.y + target.y);
    await page.waitForTimeout(700);
    const nb = await notebook(page);
    check('phone: tapping walker #18 selects it', nb.includes('#18'), nb.split('\n').slice(0, 3).join(' | '));
    await page.screenshot({ path: `${out}/phone-tap.png`, fullPage: true });
  } else check('phone: walker #18 visible at 12.6 s', false);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check('phone: no horizontal page scroll', overflow <= 0, `overflow ${overflow}px`);
  await ctx.close();
}

await browser.close();
console.log(results.join('\n'));
console.log(errors.length ? `page errors:\n${errors.join('\n')}` : 'no page errors');
