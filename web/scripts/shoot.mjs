// Screenshots of the hero for design review. Usage: node scripts/shoot.mjs [baseUrl] [outDir]
// Needs a running server (npm run preview). Times are chosen from the exported C3905 events.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://localhost:4173/';
const out = process.argv[3] ?? 'shots';
mkdirSync(out, { recursive: true });

const shots = [
  { name: 'desktop-12s', w: 1440, h: 900, t: 12.6 },
  { name: 'desktop-17s', w: 1440, h: 900, t: 17.2 },
  { name: 'desktop-59s', w: 1440, h: 900, t: 59.2 },
  { name: 'desktop-78s', w: 1440, h: 900, t: 78 },
  { name: 'phone-12s', w: 390, h: 844, t: 12.6, mobile: true },
  { name: 'phone-17s', w: 390, h: 844, t: 17.2, mobile: true },
];

const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
const errors = [];
for (const s of shots) {
  const ctx = await browser.newContext({ viewport: { width: s.w, height: s.h }, deviceScaleFactor: s.mobile ? 2 : 1, isMobile: !!s.mobile, hasTouch: !!s.mobile });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${s.name}: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && errors.push(`${s.name}: ${m.text()}`));
  await page.goto(`${base}?t=${s.t}&paused`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__zlt?.renderer?.current);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/${s.name}.png` });
  if (s.name.startsWith('desktop-12')) await page.screenshot({ path: `${out}/${s.name}-full.png`, fullPage: true });
  await ctx.close();
}
await browser.close();
console.log(errors.length ? errors.join('\n') : 'no page errors');
