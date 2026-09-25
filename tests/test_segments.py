import importlib.util
import itertools
import random
import unittest

from src.contracts import Event
from src.postprocess.segments import SegmentRules, postprocess, to_official

ROOT_EVAL = importlib.util.spec_from_file_location("official_evaluator", "evaluate.py")
official = importlib.util.module_from_spec(ROOT_EVAL)
ROOT_EVAL.loader.exec_module(official)


def ev(label, s, e, conf=0.9, ids=()):
    return Event(label, s, e, conf, ids)


class SegmentTests(unittest.TestCase):
    def test_same_class_overlaps_become_one_covering_segment(self):
        out = postprocess([ev("jaywalking", 10, 14, ids=(1,)), ev("jaywalking", 12, 20, ids=(2,))], 60, 30)
        self.assertEqual([(e.start, e.end, e.track_ids) for e in out], [(10, 20, (1, 2))])

    def test_different_classes_may_overlap(self):
        out = postprocess([ev("wrong_way", 5, 15), ev("accident", 10, 12)], 60, 30)
        self.assertEqual(len(out), 2)

    def test_gap_merge_and_min_duration(self):
        rules = {"stopped_vehicle": SegmentRules(merge_gap_s=1.0, min_dur_s=2.0)}
        out = postprocess([ev("stopped_vehicle", 0, 5), ev("stopped_vehicle", 5.5, 9),
                           ev("stopped_vehicle", 30, 31)], 60, 30, rules)
        self.assertEqual([(e.start, e.end) for e in out], [(0, 9)])

    def test_clamped_to_duration(self):
        out = postprocess([ev("fire_smoke", 50, 80), ev("congestion", 59.99, 70)], 60, 30)
        self.assertEqual([(e.label, e.start, e.end) for e in out], [("fire_smoke", 50, 60)])

    def test_sub_frame_segment_is_widened_to_one_frame(self):
        rows = to_official(postprocess([ev("near_miss", 3.0001, 3.02)], 60, 30))
        self.assertEqual(rows, [[3.0, 3.033, "near_miss"]])

    def test_sub_half_frame_blip_is_dropped(self):
        self.assertEqual(postprocess([ev("near_miss", 3.0, 3.001)], 60, 30), [])

    def test_random_events_always_pass_official_validator(self):
        rng = random.Random(0)
        labels = ["accident", "jaywalking", "wrong_way"]
        for _ in range(200):
            duration = rng.uniform(5, 200)
            events = []
            for _ in range(rng.randint(0, 12)):
                s = rng.uniform(-5, duration + 5)
                e = s + rng.uniform(0.0005, 30)
                if e > 0:
                    events.append(ev(rng.choice(labels), max(s, 0), e))
            rows = to_official(postprocess(events, duration, 29.97))
            pred = {"team": "t", "videos": {"v.mp4": {"events": rows, "risk": []}}}
            errors, _ = official.validate(pred, {"v.mp4": {"duration": duration, "events": []}})
            self.assertEqual(errors, [])
            for label in labels:
                segs = sorted((s, e) for s, e, lab in rows if lab == label)
                for (s1, e1), (s2, e2) in itertools.pairwise(segs):
                    self.assertLessEqual(e1, s2)

    def test_deterministic_regardless_of_input_order(self):
        events = [ev("jaywalking", 1, 3), ev("accident", 2, 4), ev("jaywalking", 2.5, 6)]
        self.assertEqual(to_official(postprocess(events, 60, 30)),
                         to_official(postprocess(list(reversed(events)), 60, 30)))


if __name__ == "__main__":
    unittest.main()
