import unittest

import numpy as np

from src.events import red_light
from event_fixtures import context, track

CFG = {"stop_line": "stop", "approach": "down", "vehicle_classes": [2, 3, 4, 5], "min_alignment": 0.5,
       "min_past_px": 25, "max_lateral_margin": 0.05}


def drive(track_id, t0, y0=300.0, speed=100.0, dur=8.0, x=500.0):
    t = np.round(np.arange(t0, t0 + dur, 0.1), 3)
    return track(track_id, 3, t, np.full(len(t), x), y0 + speed * (t - t0))


class RedLightTests(unittest.TestCase):
    def test_crossing_on_red_fires_with_exact_start_and_junction_exit_end(self):
        # Foot reaches y=500 at t0 + 2.0 s; front point = bottom edge, so crossing is at 12.0 s.
        ctx = context([drive(1, 10.0)], [(0, "red")])
        (ev,) = red_light.detect(ctx, CFG)
        self.assertAlmostEqual(ev.start, 12.0, places=6)
        # Enters the intersection (y>570) and leaves it (y>895): first sample outside is 16.0 s.
        self.assertAlmostEqual(ev.end, 16.0, places=6)
        self.assertEqual(ev.track_ids, (1,))
        self.assertEqual(ev.evidence["signal_state"], "red")

    def test_crossing_on_green_or_yellow_does_not_fire(self):
        for state in ("green", "yellow"):
            ctx = context([drive(1, 10.0)], [(0, state)])
            self.assertEqual(red_light.detect(ctx, CFG), [])

    def test_vehicle_stopping_just_before_the_line_does_not_fire(self):
        t = np.round(np.arange(0, 10, 0.1), 3)
        y = np.minimum(300 + 100 * t, 490.0)
        ctx = context([track(1, 3, t, np.full(len(t), 500.0), y)], [(0, "red")])
        self.assertEqual(red_light.detect(ctx, CFG), [])

    def test_boundary_red_starts_just_before_or_after_crossing(self):
        ctx = context([drive(1, 10.0)], [(0, "green"), (11.9, "red")])
        self.assertEqual(len(red_light.detect(ctx, CFG)), 1)
        ctx = context([drive(1, 10.0)], [(0, "green"), (12.1, "red")])
        self.assertEqual(red_light.detect(ctx, CFG), [])

    def test_opposite_direction_and_off_segment_do_not_fire(self):
        t = np.round(np.arange(10, 18, 0.1), 3)
        up = track(1, 3, t, np.full(len(t), 500.0), 800 - 100 * (t - 10))
        beside = drive(2, 10.0, x=980.0)
        self.assertEqual(red_light.detect(context([up, beside], [(0, "red")]), CFG), [])

    def test_track_ending_inside_junction_ends_at_last_sample(self):
        ev = red_light.detect(context([drive(1, 10.0, dur=4.0)], [(0, "red")]), CFG)[0]
        self.assertAlmostEqual(ev.end, 13.9 + 0.1, places=6)

    def test_pedestrians_are_ignored(self):
        t = np.round(np.arange(10, 18, 0.1), 3)
        ped = track(1, 0, t, np.full(len(t), 500.0), 300 + 100 * (t - 10))
        self.assertEqual(red_light.detect(context([ped], [(0, "red")]), CFG), [])


if __name__ == "__main__":
    unittest.main()
