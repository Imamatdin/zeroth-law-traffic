import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import illegal_u_turn
from event_fixtures import context
from turn_fixtures import arc_path

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "events.yaml").read_text(encoding="utf-8"))["illegal_u_turn"]
ZONE = np.array([[350, 250], [600, 250], [600, 550], [350, 550]], float)   # holds the U-turn around (400-500, 300-500)


def run(tracks, zones=True):
    ctx = context(tracks, [(0, "green")])
    ctx.scene.turn_restrictions = ([{"id": "no_u", "kind": "u_turn", "zone": ZONE, "source": "sign",
                                     "evidence": "test"}] if zones else [])
    return illegal_u_turn.detect(ctx, CFG)


class IllegalUTurnTests(unittest.TestCase):
    def test_u_turn_under_a_no_u_turn_sign_fires_from_start_to_completion(self):
        rows, a0, a1 = arc_path((100, 300), 0, 180, 100)
        (ev,) = run([rows])
        self.assertAlmostEqual(ev.start, a0, delta=0.5)
        self.assertAlmostEqual(ev.end, a1, delta=0.5)
        self.assertEqual(ev.evidence["restrictions"], ["no_u"])

    def test_u_turn_elsewhere_or_without_restrictions_does_not_fire(self):
        rows, _, _ = arc_path((100, 700), 0, -180, 60, before=100)
        self.assertEqual(run([rows]), [])
        rows, _, _ = arc_path((100, 300), 0, 180, 100)
        self.assertEqual(run([rows], zones=False), [])

    def test_ordinary_turn_in_the_zone_is_not_a_u_turn(self):
        rows, _, _ = arc_path((100, 300), 0, 90, 150)
        self.assertEqual(run([rows]), [])


if __name__ == "__main__":
    unittest.main()
