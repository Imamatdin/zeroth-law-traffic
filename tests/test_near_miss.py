import unittest
from pathlib import Path

import yaml

from src.events import near_miss
from event_fixtures import context
from interaction_fixtures import brake, cruise, moving

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["near_miss"]


def crossing_car():
    """Drives right along y=700 at 200 px/s: through x=500 at t=2.5 s."""
    return moving(1, 3, (0, 700), (1, 0), cruise(200.0))


def run(tracks):
    return near_miss.detect(context(tracks, [(0, "green")]), CFG)


class NearMissTests(unittest.TestCase):
    def test_hard_brake_that_avoids_a_crossing_car_fires_from_the_brake_onset(self):
        # Would meet the crossing car at (500, 700) at t=2.5; brakes at 2.02 s (8 box heights/s^2) and
        # stops 35 px (under one box height) short of its path, without touching it.
        braker = moving(2, 3, (500, 200), (0, 1), brake(200.0, 2.02, 330.0))
        (ev,) = run([crossing_car(), braker])
        self.assertEqual(ev.evidence["actor"], 2)
        self.assertEqual(ev.evidence["action"], "brake")
        self.assertAlmostEqual(ev.start, 2.02, delta=0.35)
        self.assertGreater(ev.end, 2.4)
        self.assertLess(ev.end, 4.0)

    def test_gentle_stop_well_before_the_path_does_not_fire(self):
        stopper = moving(2, 3, (500, 200), (0, 1), brake(200.0, 0.5, 60.0))
        self.assertEqual(run([crossing_car(), stopper]), [])

    def test_crossing_behind_without_evasive_action_does_not_fire(self):
        late = moving(2, 3, (500, 200), (0, 1), cruise(200.0), t0=1.0, t1=8.0)
        self.assertEqual(run([crossing_car(), late]), [])

    def test_braking_too_late_to_avoid_contact_is_not_a_near_miss(self):
        crash = moving(2, 3, (500, 200), (0, 1), brake(200.0, 2.2, 330.0))
        self.assertEqual(run([crossing_car(), crash]), [])

    def test_stopping_two_box_heights_short_is_not_near(self):
        early = moving(2, 3, (500, 200), (0, 1), brake(200.0, 1.8, 330.0))
        self.assertEqual(run([crossing_car(), early]), [])

    def test_hard_brake_with_nobody_in_the_way_does_not_fire(self):
        alone = moving(2, 3, (500, 200), (0, 1), brake(200.0, 1.8, 330.0))
        far = moving(1, 3, (0, 950), (1, 0), cruise(200.0))
        self.assertEqual(run([far, alone]), [])

    def test_pedestrian_on_the_road_counts_as_the_other_road_user(self):
        walker = moving(1, 0, (300, 700), (1, 0), cruise(80.0), w=15)           # at (500, 700) at 2.5 s
        braker = moving(2, 3, (500, 200), (0, 1), brake(200.0, 2.02, 330.0))
        (ev,) = run([walker, braker])
        self.assertEqual(ev.evidence["cls"], [0, 3])


if __name__ == "__main__":
    unittest.main()
