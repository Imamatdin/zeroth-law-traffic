import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import road_obstacle
from event_fixtures import context, track
from visual_fixtures import checker, glow, samples

CFG = dict(yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["road_obstacle"])
CFG["illum_sigma_px"] = 8          # the fixture samples are 200 px wide, not 640


def traffic():
    """A car passing along y=300 (source px) every 10 s, box 80 px high: the depth scale reference."""
    rows = []
    for k in range(6):
        t = np.round(np.arange(10.0 * k, 10.0 * k + 8.0, 0.1), 3)
        rows.append(track(k + 1, 3, t, 100.0 + 100.0 * (t - 10.0 * k), np.full(len(t), 300.0), w=80.0, h=80.0))
    return rows


def run(vis, extra_tracks=()):
    ctx = context(traffic() + list(extra_tracks), [(0, "green")], duration=60.0)
    ctx.visual = vis
    return road_obstacle.detect(ctx, CFG)


class RoadObstacleTests(unittest.TestCase):
    def test_object_that_appears_on_the_road_and_is_removed_fires_for_its_stay(self):
        vis = samples(paint=lambda f, t: checker(f, 140, 140) if 20 <= t < 45 else None)
        (ev,) = run(vis)
        self.assertAlmostEqual(ev.start, 20.0, delta=1.01)
        self.assertAlmostEqual(ev.end, 45.0, delta=1.01)

    def test_short_stay_does_not_fire(self):
        vis = samples(paint=lambda f, t: checker(f, 140, 140) if 20 <= t < 30 else None)
        self.assertEqual(run(vis), [])

    def test_parked_vehicle_explained_by_its_track_does_not_fire(self):
        vis = samples(paint=lambda f, t: checker(f, 140, 140) if 10 <= t < 50 else None)
        t = np.round(np.arange(10.0, 50.0, 0.1), 3)
        parked = track(99, 3, t, np.full(len(t), 725.0), np.full(len(t), 750.0), w=50.0, h=50.0)
        self.assertEqual(run(vis, [parked]), [])

    def test_smooth_light_pool_is_not_an_object(self):
        vis = samples(paint=lambda f, t: glow(f, 145, 145) if t >= 20 else None)
        self.assertEqual(run(vis), [])

    def test_video_getting_darker_does_not_fire(self):
        vis = samples(gain_of_t=lambda t: 1.0 - 0.6 * t / 60.0)
        self.assertEqual(run(vis), [])

    def test_object_on_the_sidewalk_is_not_on_the_road(self):
        vis = samples(paint=lambda f, t: checker(f, 1, 100) if 20 <= t < 45 else None)
        self.assertEqual(run(vis), [])

    def test_without_visual_samples_nothing_fires(self):
        self.assertEqual(run(None), [])


if __name__ == "__main__":
    unittest.main()
