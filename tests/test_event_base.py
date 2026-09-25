import unittest

import numpy as np

from src.events.base import condition_segments


class ConditionSegmentTests(unittest.TestCase):
    t = np.round(np.arange(0, 5, 0.1), 3)

    def cond(self, *ranges):
        c = np.zeros(len(self.t), bool)
        for lo, hi in ranges:
            c |= (self.t >= lo - 1e-9) & (self.t < hi - 1e-9)
        return c

    def test_start_backdated_and_end_is_first_false_sample(self):
        (seg,) = condition_segments(self.t, self.cond((1.0, 2.5)), 1.0, 0.0, 0.1)
        self.assertEqual(seg[:2], (1.0, 2.5))

    def test_persistence_rejects_short_runs(self):
        self.assertEqual(condition_segments(self.t, self.cond((1.0, 1.5)), 1.0, 0.0, 0.1), [])

    def test_hysteresis_bridges_short_gaps_only(self):
        c = self.cond((1.0, 2.0), (2.3, 3.0))
        self.assertEqual([s[:2] for s in condition_segments(self.t, c, 0.5, 0.4, 0.1)], [(1.0, 3.0)])
        self.assertEqual(len(condition_segments(self.t, c, 0.5, 0.2, 0.1)), 2)

    def test_run_to_track_end_extends_by_one_step(self):
        (seg,) = condition_segments(self.t, self.cond((4.0, 9.0)), 0.5, 0.0, 0.1)
        self.assertAlmostEqual(seg[1], 5.0)


if __name__ == "__main__":
    unittest.main()
