import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.events import stopped_vehicle
from event_fixtures import context, track

CFG = {k: v for k, v in yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["stopped_vehicle"].items()}
CFG.update({"stop_line": "stop", "approach": "down"})


def arrive_stop_leave(track_id, x, y_stop, t_stop, t_go, t_end=40.0, speed=100.0, jitter=None):
    """Drives down at `speed`, stands at (x, y_stop) from t_stop to t_go, then drives on."""
    t = np.round(np.arange(max(0.0, t_stop - 2.0), t_end, 0.1), 3)
    y = np.where(t < t_stop, y_stop - speed * (t_stop - t), np.where(t < t_go, y_stop, y_stop + speed * (t - t_go)))
    xs = np.full(len(t), float(x))
    if jitter is not None:
        for tj in jitter:                                  # box jumps 15 px for one sample
            k = int(np.argmin(np.abs(t - tj)))
            xs[k] += 15.0
    keep = y < 990
    return track(track_id, 3, t[keep], xs[keep], y[keep])


def run(tracks, phases=((0, "green"),), duration=60.0):
    return stopped_vehicle.detect(context(tracks, list(phases), duration=duration), CFG)


class StoppedVehicleTests(unittest.TestCase):
    def test_isolated_stop_fires_from_stop_to_move_off(self):
        (ev,) = run([arrive_stop_leave(1, 300, 300, 5.0, 20.0)])
        self.assertAlmostEqual(ev.start, 5.0, delta=0.31)
        self.assertAlmostEqual(ev.end, 20.0, delta=0.51)

    def test_stop_shorter_than_ten_seconds_does_not_fire(self):
        self.assertEqual(run([arrive_stop_leave(1, 300, 300, 5.0, 13.0)]), [])

    def test_box_jitter_does_not_split_the_stop(self):
        evs = run([arrive_stop_leave(1, 300, 300, 5.0, 25.0, jitter=(9.0, 14.0, 19.0))])
        self.assertEqual(len(evs), 1)
        self.assertAlmostEqual(evs[0].end, 25.0, delta=0.51)

    def test_signal_queue_head_and_the_cars_behind_are_explained(self):
        # Head just before the stop line (y=500) on red; the tail at y=320 is 180 px (4.5 box heights)
        # behind the line, beyond the signal rule's reach, so only the queue chain explains it.
        queue = [arrive_stop_leave(1 + k, 500, 480 - 55 * k, 3.0 + k, 25.0 + k) for k in range(4)]
        self.assertEqual(run(queue, phases=[(0, "red"), (25.0, "green")]), [])

    def test_queue_chain_is_what_explains_the_tail(self):
        queue = [arrive_stop_leave(1 + k, 500, 480 - 55 * k, 3.0 + k, 25.0 + k) for k in range(4)]
        cfg = {**CFG, "queue_gap_h": 0.0}
        evs = stopped_vehicle.detect(context(queue, [(0, "red"), (25.0, "green")], duration=60.0), cfg)
        self.assertEqual([e.track_ids for e in evs], [(4,)])

    def test_same_queue_head_on_green_is_a_stopped_vehicle(self):
        head = arrive_stop_leave(1, 500, 480, 3.0, 25.0)
        self.assertEqual(len(run([head], phases=[(0, "green")])), 1)

    def test_normal_stopping_place_is_explained(self):
        others = [arrive_stop_leave(10 + k, 300, 300, 2.0 + 6 * k, 6.0 + 6 * k, t_end=6.0 + 6 * k + 1.0)
                  for k in range(2)]
        me = arrive_stop_leave(1, 300, 300, 20.0, 35.0, t_end=50.0)
        self.assertEqual(run(others + [me]), [])

    def test_standstill_among_slow_traffic_is_explained(self):
        me = arrive_stop_leave(1, 600, 700, 3.0, 20.0)
        jam = [arrive_stop_leave(10 + k, 600 + dx, 700 + dy, 3.0, 20.0)
               for k, (dx, dy) in enumerate(((60, 0), (-60, 0), (0, -70)))]
        self.assertEqual([e.track_ids for e in run([me] + jam)], [])

    def test_vehicle_on_the_sidewalk_is_ignored(self):
        self.assertEqual(run([arrive_stop_leave(1, 30, 300, 5.0, 25.0)]), [])

    def test_stop_running_past_the_end_ends_at_duration(self):
        (ev,) = run([arrive_stop_leave(1, 300, 300, 5.0, 99.0, t_end=30.0)], duration=30.0)
        self.assertAlmostEqual(ev.end, 30.0, delta=0.11)


if __name__ == "__main__":
    unittest.main()
