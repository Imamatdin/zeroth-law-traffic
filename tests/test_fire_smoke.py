import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import fire_smoke
from event_fixtures import context, track
from visual_fixtures import SIZE, samples

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["fire_smoke"]
ORANGE = np.array([0, 140, 255], float)       # BGR


def traffic():
    """A car passing along y=300 (source px) every 10 s, box 80 px high: the depth scale reference."""
    rows = []
    for k in range(6):
        t = np.round(np.arange(10.0 * k, 10.0 * k + 8.0, 0.1), 3)
        rows.append(track(k + 1, 3, t, 100.0 + 100.0 * (t - 10.0 * k), np.full(len(t), 300.0), w=80.0, h=80.0))
    return rows


def disc(frame, x, y, r, colour):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    frame[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = colour


def plume(frame, x, y, r, amp=70.0):
    """Soft grey haze: brighter, smoothing the texture underneath."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    wgt = np.clip(1.0 - np.hypot(xx - x, yy - y) / r, 0.0, 1.0)[..., None] ** 0.5
    frame[:] = frame * (1 - 0.8 * wgt) + (np.mean(frame) + amp) * 0.8 * wgt


def run(vis, extra=()):
    ctx = context(traffic() + list(extra), [(0, "green")], duration=60.0)
    ctx.visual = vis
    return fire_smoke.detect(ctx, CFG)


class FireSmokeTests(unittest.TestCase):
    def test_flickering_flames_on_the_road_fire(self):
        radius = lambda t: 5 + 3 * ((int(t) * 7) % 3)                     # 5, 8 or 11 px, changing
        vis = samples(paint=lambda f, t: disc(f, 140, 150, radius(t), ORANGE) if 20 <= t < 40 else None)
        (ev,) = run(vis)
        self.assertEqual(ev.evidence["kind"], "fire")
        self.assertAlmostEqual(ev.start, 20.0, delta=1.01)
        self.assertAlmostEqual(ev.end, 40.0, delta=2.01)

    def test_steady_orange_lamp_does_not_fire(self):
        vis = samples(paint=lambda f, t: disc(f, 140, 150, 8, ORANGE))
        self.assertEqual(run(vis), [])

    def test_lights_on_a_moving_tracked_vehicle_do_not_fire(self):
        # The light rides on car 1 (x = 100 + 100 t px source at y = 300), inside its box.
        def paint(f, t):
            if t < 8:
                disc(f, int((100 + 100 * t) * 0.2), 56, 3 + int(t) % 3, ORANGE)
        self.assertEqual(run(samples(paint=paint)), [])

    def test_scrolling_orange_sign_on_a_standing_bus_does_not_fire(self):
        t = np.round(np.arange(0.0, 60.0, 0.1), 3)
        bus = track(50, 4, t, np.full(len(t), 750.0), np.full(len(t), 800.0), w=300.0, h=200.0)

        def sign(f, t):
            width = 12 + 6 * ((int(t) * 7) % 3)                              # scrolling text: 12-24 px
            f[125:129, 130:130 + width] = ORANGE
        self.assertEqual(run(samples(paint=sign), [bus]), [])

    def test_growing_smoke_plume_fires(self):
        vis = samples(paint=lambda f, t: plume(f, 140, 150, 10 + (t - 20)) if 20 <= t < 45 else None)
        (ev,) = run(vis)
        self.assertEqual(ev.evidence["kind"], "smoke")
        self.assertAlmostEqual(ev.start, 20.0, delta=2.01)

    def test_brightening_video_does_not_fire(self):
        self.assertEqual(run(samples(gain_of_t=lambda t: 0.7 + 0.6 * t / 60.0)), [])

    def test_without_visual_samples_nothing_fires(self):
        self.assertEqual(run(None), [])


if __name__ == "__main__":
    unittest.main()
