import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import accident
from event_fixtures import context, track
from interaction_fixtures import brake, cruise, moving

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["accident"]


def run(tracks):
    return accident.detect(context(tracks, [(0, "green")]), CFG)


def standing(track_id, x, y, t1=8.0, cls=3):
    t = np.round(np.arange(0.0, t1 + 1e-9, 0.1), 3)
    return track(track_id, cls, t, np.full(len(t), float(x)), np.full(len(t), float(y)))


class AccidentTests(unittest.TestCase):
    def test_side_impact_fires_from_first_contact_until_both_stop(self):
        # Both reach (500, 700) at ~2.45 s and stop dead there (2000 px/s^2).
        a = moving(1, 3, (0, 700), (1, 0), brake(200.0, 2.45, 2000.0))
        b = moving(2, 3, (500, 200), (0, 1), brake(200.0, 2.45, 2000.0))
        (ev,) = run([a, b])
        self.assertEqual(ev.evidence["kind"], "road_users")
        self.assertEqual(sorted(ev.track_ids), [1, 2])
        self.assertAlmostEqual(ev.start, 2.4, delta=0.21)
        self.assertGreater(ev.end, ev.start)
        self.assertLess(ev.end, 4.0)

    def test_queue_creeping_into_contact_does_not_fire(self):
        lead = standing(1, 500, 700)
        creep = moving(2, 3, (500, 600), (0, 1), brake(20.0, 3.0, 10.0))
        self.assertEqual(run([lead, creep]), [])

    def test_near_miss_without_contact_does_not_fire(self):
        a = moving(1, 3, (0, 700), (1, 0), cruise(200.0))
        b = moving(2, 3, (500, 200), (0, 1), brake(200.0, 2.02, 330.0))
        self.assertEqual(run([a, b]), [])

    def test_boxes_overlapping_while_passing_do_not_fire(self):
        a = moving(1, 3, (0, 700), (1, 0), cruise(200.0))
        b = moving(2, 3, (1000, 695), (-1, 0), cruise(200.0), t1=4.9)
        self.assertEqual(run([a, b]), [])

    def test_vehicle_running_into_a_refuge_and_stopping_is_a_fixed_object_collision(self):
        # Drives up towards the refuge (y 200-260) and stops dead with its footprint on it.
        car = moving(1, 3, (500, 600), (0, -1), brake(200.0, 1.625, 2000.0))
        (ev,) = run([car])
        self.assertEqual(ev.evidence["kind"], "fixed_object")

    def test_parking_gently_at_the_kerb_does_not_fire(self):
        car = moving(1, 3, (300, 500), (-1, 0), brake(100.0, 0.5, 45.0))     # stops at x ~ 69, beside the sidewalk
        self.assertEqual(run([car]), [])


if __name__ == "__main__":
    unittest.main()
