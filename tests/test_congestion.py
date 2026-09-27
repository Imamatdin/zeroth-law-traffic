import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.atlas.flow import Atlas, build_atlas
from src.events import congestion
from event_fixtures import context, track

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["congestion"]
LANES = (250.0, 350.0, 450.0)


def drive(track_id, t0, y, speed=100.0):
    t = np.round(np.arange(t0, t0 + 8.4, 0.1), 3)
    return track(track_id, 3, t, 80.0 + speed * (t - t0), np.full(len(t), y))


def three_lane_atlas():
    """A three-lane road flowing right (+x) at y = 250, 350, 450."""
    rows = [drive(100 + 10 * k + j, 0.0, y) for k in range(8) for j, y in enumerate(LANES)]
    return Atlas(build_atlas(pd.concat(rows, ignore_index=True), grid=(10, 10), min_samples=10))


def standing(track_id, x, y, t0, t1):
    t = np.round(np.arange(t0, t1, 0.1), 3)
    return track(track_id, 3, t, np.full(len(t), x), np.full(len(t), y))


def jam(t0, t1, lanes=LANES, per_lane=2):
    return [standing(1 + 10 * j + k, 300.0 + 150.0 * k, y, t0, t1) for j, y in enumerate(lanes) for k in range(per_lane)]


def run(tracks, phases=((0, "green"),), signalled=False, duration=120.0):
    ctx = context(tracks, list(phases), duration=duration)
    ctx.scene.approaches = {"right": {"direction": [1.0, 0.0]}}
    ctx.scene.stop_line_meta = {"stop": {"approach": "right" if signalled else "none", "signal": "sig"}}
    ctx.atlas = three_lane_atlas()
    return congestion.detect(ctx, CFG)


class CongestionTests(unittest.TestCase):
    def test_long_standstill_across_all_lanes_without_a_known_signal_fires(self):
        (ev,) = run(jam(5.0, 80.0))
        self.assertAlmostEqual(ev.start, 5.0, delta=0.11)
        self.assertAlmostEqual(ev.end, 80.0, delta=0.21)
        self.assertEqual(ev.evidence["approach"], "right")
        self.assertFalse(ev.evidence["signal_known"])

    def test_standstill_shorter_than_a_cycle_without_a_known_signal_does_not_fire(self):
        self.assertEqual(run(jam(5.0, 45.0)), [])

    def test_a_lane_that_keeps_flowing_is_not_congestion(self):
        flowing = [drive(200 + k, 3.0 * k, LANES[1]) for k in range(30)]
        self.assertEqual(run(jam(5.0, 80.0, lanes=(LANES[0], LANES[2]), per_lane=3) + flowing), [])

    def test_red_light_queue_that_clears_on_green_does_not_fire(self):
        self.assertEqual(run(jam(5.0, 45.0), phases=[(0, "red"), (40.0, "green")], signalled=True), [])

    def test_queue_still_standing_through_green_fires_from_when_it_stopped(self):
        (ev,) = run(jam(5.0, 70.0), phases=[(0, "red"), (40.0, "green")], signalled=True)
        self.assertAlmostEqual(ev.start, 5.0, delta=0.11)
        self.assertTrue(ev.evidence["signal_known"])
        self.assertGreaterEqual(ev.evidence["standstill_green_s"], 29.0)

    def test_too_few_vehicles_or_no_atlas_do_not_fire(self):
        self.assertEqual(run(jam(5.0, 80.0, lanes=LANES[:1])), [])
        ctx = context(jam(5.0, 80.0), [(0, "green")], duration=120.0)
        ctx.scene.approaches = {"right": {"direction": [1.0, 0.0]}}
        self.assertEqual(congestion.detect(ctx, CFG), [])


if __name__ == "__main__":
    unittest.main()
