import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import solid_line_crossing
from event_fixtures import context, track

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["solid_line_crossing"]


def drive(track_id, xs_of_t, t_end=8.0, cls=3, y0=150.0, speed=100.0):
    """Drives down (+y) at `speed`; x follows xs_of_t(t). Box 40x40, foot at the bottom centre."""
    t = np.round(np.arange(0.0, t_end, 0.1), 3)
    return track(track_id, cls, t, xs_of_t(t), y0 + speed * t)


def lane_change(x_from=450.0, x_to=550.0, t0=3.0, dur=1.0):
    return lambda t: np.interp(t, [t0, t0 + dur], [x_from, x_to])


def run(tracks, lines=None):
    ctx = context(tracks, [(0, "green")])
    ctx.scene.solid_lines = {"solid": np.array([[500.0, 100.0], [500.0, 900.0]])} if lines is None else lines
    return solid_line_crossing.detect(ctx, CFG)


class SolidLineCrossingTests(unittest.TestCase):
    def test_lane_change_across_the_line_starts_at_the_wheel_and_ends_fully_across(self):
        (ev,) = run([drive(1, lane_change())])
        # Box half-width 20: the right corner is past x=500 once the foot is beyond 480 (t=3.4), the left
        # corner passes it when the foot is beyond 520 (first sample t=3.8).
        self.assertAlmostEqual(ev.start, 3.4, delta=0.11)
        self.assertAlmostEqual(ev.end, 3.8, delta=0.11)
        self.assertEqual(ev.evidence["line"], "solid")
        self.assertTrue(ev.evidence["fully_across"])

    def test_staying_in_lane_with_jitter_does_not_fire(self):
        wobble = lambda t: 470.0 + 12.0 * np.sin(7 * t)
        self.assertEqual(run([drive(1, wobble)]), [])

    def test_brief_drift_across_and_back_does_not_fire(self):
        drift = lambda t: np.interp(t, [3.0, 3.3, 3.6, 3.9], [450.0, 540.0, 540.0, 450.0])
        self.assertEqual(run([drive(1, drift)]), [])

    def test_crossing_beyond_the_end_of_the_marking_does_not_fire(self):
        line = {"short": np.array([[500.0, 100.0], [500.0, 300.0]])}
        # At t=3-4 s the car is at y=450-550, past the end of the solid part.
        self.assertEqual(run([drive(1, lane_change())], lines=line), [])

    def test_id_switch_between_neighbouring_lanes_does_not_fire(self):
        jump = lambda t: np.where(t < 3.0, 450.0, 550.0)
        self.assertEqual(run([drive(1, jump)]), [])

    def test_pedestrians_and_maps_without_solid_lines_do_not_fire(self):
        self.assertEqual(run([drive(1, lane_change(), cls=0, speed=30.0)]), [])
        self.assertEqual(run([drive(1, lane_change())], lines={}), [])


if __name__ == "__main__":
    unittest.main()
