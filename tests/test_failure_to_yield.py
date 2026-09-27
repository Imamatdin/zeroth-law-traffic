import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.events import failure_to_yield
from event_fixtures import context, track

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["failure_to_yield"]


def line(track_id, cls, t0, t1, p0, p1, w=40.0, h=40.0):
    """Constant velocity from p0 at t0 to p1 at t1, sampled at 10 Hz."""
    t = np.round(np.arange(t0, t1 + 1e-9, 0.1), 3)
    s = (t - t0) / (t1 - t0)
    x = p0[0] + s * (p1[0] - p0[0])
    y = p0[1] + s * (p1[1] - p0[1])
    return track(track_id, cls, t, x, y, w=w, h=h)


def car_down(x=500.0, t0=0.0):
    """Drives down through the crossing (y 510-560) at 100 px/s: footprint on it from 4.1 s until 4.8 s after t0."""
    return line(1, 3, t0, t0 + 9.0, (x, 100.0), (x, 1000.0 - 10.0))


def run(tracks):
    return failure_to_yield.detect(context(tracks, [(0, "green")]), CFG)


class FailureToYieldTests(unittest.TestCase):
    def test_driving_through_while_a_pedestrian_crosses_in_front_fires_for_the_crossing_time(self):
        ped = line(9, 0, 0.0, 10.0, (400.0, 540.0), (600.0, 540.0), w=15, h=40)
        (ev,) = run([car_down(), ped])
        self.assertAlmostEqual(ev.start, 4.1, delta=0.11)       # footprint (lower half, 20 px) reaches y=510
        self.assertAlmostEqual(ev.end, 4.9, delta=0.11)         # first sample with the footprint past y=560
        self.assertEqual(ev.evidence["pedestrians"], [9])
        self.assertEqual(ev.evidence["crosswalk"], "xing")

    def test_pedestrian_at_the_far_end_of_the_crossing_does_not_fire(self):
        ped = line(9, 0, 0.0, 10.0, (800.0, 540.0), (880.0, 540.0), w=15, h=40)
        self.assertEqual(run([car_down(), ped]), [])

    def test_vehicle_that_waits_for_the_pedestrian_does_not_fire(self):
        # Stops with its footprint on the crossing edge until the pedestrian has passed its path.
        t = np.round(np.arange(0.0, 20.0, 0.1), 3)
        y = np.where(t < 4.0, 100 + 100 * t, np.where(t < 12.0, 500.0, 500 + 100 * (t - 12.0)))
        keep = y < 990
        car = track(1, 3, t[keep], np.full(keep.sum(), 500.0), y[keep])
        ped = line(9, 0, 3.0, 11.0, (300.0, 540.0), (700.0, 540.0), w=15, h=40)
        self.assertEqual(run([car, ped]), [])

    def test_pedestrian_on_the_sidewalk_is_not_on_the_crossing(self):
        ped = line(9, 0, 0.0, 10.0, (30.0, 400.0), (30.0, 600.0), w=15, h=40)
        self.assertEqual(run([car_down(), ped]), [])

    def test_pedestrian_stepping_onto_the_crossing_counts(self):
        # Steps up onto the crossing from below it, in front of the arriving car.
        ped = line(9, 0, 3.5, 6.5, (510.0, 590.0), (510.0, 500.0), w=15, h=40)
        self.assertEqual(len(run([car_down(), ped])), 1)

    def test_rider_on_a_motorcycle_is_not_a_pedestrian(self):
        bike = line(8, 2, 0.0, 10.0, (400.0, 540.0), (600.0, 540.0), w=40, h=40)
        rider = line(9, 0, 0.0, 10.0, (400.0, 530.0), (600.0, 530.0), w=15, h=40)
        self.assertEqual(run([car_down(), bike, rider]), [])

    def test_no_crossing_in_the_map_means_no_events(self):
        ctx = context([car_down(), line(9, 0, 0.0, 10.0, (400.0, 540.0), (600.0, 540.0), w=15, h=40)],
                      [(0, "green")])
        ctx.scene.regions = [r for r in ctx.scene.regions if r.kind != "crosswalks"]
        self.assertEqual(failure_to_yield.detect(ctx, CFG), [])


if __name__ == "__main__":
    unittest.main()
