import unittest

import numpy as np

from src.events import stop_line
from event_fixtures import context, track

CFG = {"stop_line": "stop", "approach": "down", "vehicle_classes": [2, 3, 4, 5], "min_past_px": 25,
       "max_lateral_margin": 0.05, "min_stop_s": 1.0, "still_rel_speed": 0.15, "max_gap_s": 0.2,
       "min_past_box_h": 0.5}


def approach_and_stop(track_id, stop_y, stop_at=5.0, until=20.0, x=500.0):
    """Drives down at 100 px/s from y=300 and halts at stop_y."""
    t = np.round(np.arange(0, until, 0.1), 3)
    y = np.minimum(300 + 100 * t, stop_y)
    return track(track_id, 3, t, np.full(len(t), x), y)


class StopLineTests(unittest.TestCase):
    def test_stops_past_line_on_red_until_green(self):
        # Foot and front reach y=540 (on the crossing, past the line) at t=2.4 s; red until 15 s.
        ctx = context([approach_and_stop(1, 540)], [(0, "red"), (15.0, "green")])
        (ev,) = stop_line.detect(ctx, CFG)
        self.assertAlmostEqual(ev.end, 15.0, places=6)
        self.assertGreaterEqual(ev.start, 2.4)
        self.assertLessEqual(ev.start, 2.6)
        self.assertEqual(ev.evidence["on_crosswalk_fraction"], 1.0)

    def test_stopping_before_the_line_does_not_fire(self):
        ctx = context([approach_and_stop(1, 495)], [(0, "red"), (15.0, "green")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])

    def test_stopping_inside_the_intersection_does_not_fire(self):
        ctx = context([approach_and_stop(1, 700)], [(0, "red"), (15.0, "green")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])

    def test_stopping_past_line_on_green_does_not_fire(self):
        ctx = context([approach_and_stop(1, 540)], [(0, "green")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])

    def test_red_onset_after_stopping_starts_the_event(self):
        ctx = context([approach_and_stop(1, 540)], [(0, "yellow"), (6.0, "red"), (15.0, "green")])
        (ev,) = stop_line.detect(ctx, CFG)
        self.assertAlmostEqual(ev.start, 6.0, places=6)
        self.assertAlmostEqual(ev.end, 15.0, places=6)

    def test_brief_stop_shorter_than_persistence_does_not_fire(self):
        t = np.round(np.arange(0, 12, 0.1), 3)
        y = np.where(t < 2.4, 300 + 100 * t, np.where(t < 3.0, 540, 540 + 100 * (t - 3.0)))
        ctx = context([track(1, 3, t, np.full(len(t), 500.0), y)], [(0, "red")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])

    def test_track_first_seen_past_the_line_does_not_fire(self):
        t = np.round(np.arange(0, 20, 0.1), 3)
        ctx = context([track(1, 3, t, np.full(len(t), 500.0), np.full(len(t), 540.0))], [(0, "red")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])

    def test_no_green_until_video_end_clamps_to_duration(self):
        ctx = context([approach_and_stop(1, 540, until=30)], [(0, "red")], duration=30.0)
        (ev,) = stop_line.detect(ctx, CFG)
        self.assertEqual(ev.end, 30.0)

    def test_crawl_then_stop_starts_at_the_sustained_stop(self):
        # Past the line by 3 s, crawling at 6 px/s (0.15 body heights/s, not still) until 6 s, then halted.
        t = np.round(np.arange(0, 20, 0.1), 3)
        y = np.where(t < 2.4, 300 + 100 * t, np.where(t < 6.0, 540 + 6 * (t - 2.4), 540 + 6 * 3.6))
        ctx = context([track(1, 3, t, np.full(len(t), 500.0), y)], [(0, "red"), (15.0, "green")])
        (ev,) = stop_line.detect(ctx, CFG)
        self.assertGreaterEqual(ev.start, 5.9)
        self.assertLessEqual(ev.start, 6.1)

    def test_front_just_over_the_line_within_box_margin_does_not_fire(self):
        # 40 px box: must be >= 25 px past; stopping 15 px past the line is within the margin.
        ctx = context([approach_and_stop(1, 515)], [(0, "red"), (15.0, "green")])
        self.assertEqual(stop_line.detect(ctx, CFG), [])


if __name__ == "__main__":
    unittest.main()
