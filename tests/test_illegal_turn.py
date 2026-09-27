import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.events import illegal_turn
from src.scene.geometry import Scene
from event_fixtures import context
from turn_fixtures import arc_path

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "events.yaml").read_text(encoding="utf-8"))["illegal_turn"]
ZONE = np.array([[300, 200], [500, 200], [500, 400], [300, 400]], float)   # around the turn at (400, 300)


def run(tracks, restrictions=(), lanes=()):
    ctx = context(tracks, [(0, "green")])
    ctx.scene.turn_restrictions = [dict(r) for r in restrictions]
    ctx.scene.lane_arrows = [dict(r) for r in lanes]
    return illegal_turn.detect(ctx, CFG)


def no_right():
    return {"id": "no_right", "kind": "right", "zone": ZONE, "source": "sign", "evidence": "test"}


class IllegalTurnTests(unittest.TestCase):
    def test_right_turn_where_a_sign_prohibits_it_fires_from_start_to_completion(self):
        rows, a0, a1 = arc_path((100, 300), 0, 90, 150)
        (ev,) = run([rows], [no_right()])
        self.assertAlmostEqual(ev.start, a0, delta=0.5)
        self.assertAlmostEqual(ev.end, a1, delta=0.5)
        self.assertEqual(ev.evidence["direction"], "right")
        self.assertEqual(ev.evidence["restrictions"], ["no_right"])

    def test_left_turn_and_straight_drive_are_not_right_turns(self):
        left, _, _ = arc_path((100, 300), 0, -90, 150)
        straight, _, _ = arc_path((100, 300), 0, 0.001, 150, track_id=2)
        self.assertEqual(run([left, straight], [no_right()]), [])

    def test_without_restrictions_nothing_is_illegal(self):
        rows, _, _ = arc_path((100, 300), 0, 90, 150)
        self.assertEqual(run([rows]), [])

    def test_turn_from_a_lane_whose_arrows_do_not_allow_it(self):
        lane = {"id": "straight_only", "zone": np.array([[50, 250], [390, 250], [390, 350], [50, 350]], float),
                "allowed": ["straight"], "source": "marking", "evidence": "test"}
        rows, _, _ = arc_path((100, 300), 0, 90, 150)
        (ev,) = run([rows], lanes=[lane])
        self.assertEqual(ev.evidence["restrictions"], ["straight_only"])
        self.assertEqual(run([rows], lanes=[{**lane, "allowed": ["straight", "right"]}]), [])

    def test_u_turn_is_left_to_illegal_u_turn(self):
        rows, _, _ = arc_path((100, 300), 0, 180, 100)
        self.assertEqual(run([rows], [no_right()]), [])

    def test_map_rejects_restrictions_without_a_sign_marking_or_organizer_source(self):
        raw = {"coords": "normalized", "turn_restrictions": [
            {"id": "x", "kind": "left", "zone": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], "source": "atlas",
             "evidence": "frequent movement"}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                Scene.load(path, 100, 100)

    def test_real_map_has_no_turn_restrictions(self):
        scene = Scene.load(ROOT / "configs" / "camera.yaml", 3840, 2160)
        self.assertEqual((scene.turn_restrictions, scene.lane_arrows), ([], []))


if __name__ == "__main__":
    unittest.main()
