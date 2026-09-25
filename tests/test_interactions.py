import unittest

import pandas as pd

from src.features.interactions import pair_features


def row(tid, x, y, vx, vy, h=0.1):
    return dict(frame=0, t=0.0, track_id=tid, cls=3, cls_major=3, gx=x, gy=y, vx=vx, vy=vy, box_h=h)


class InteractionTests(unittest.TestCase):
    def test_head_on_pair(self):
        # 1000 px apart on x, closing at 200 px/s each way, boxes 100 px tall.
        w = pd.DataFrame([row(1, 0.0, 0.5, 0.1, 0.0), row(2, 0.5, 0.5, -0.1, 0.0)])
        p = pair_features(w, 2000, 1000, max_dist_rel=20).iloc[0]
        self.assertAlmostEqual(p.dist, 10.0)
        self.assertAlmostEqual(p.closing_speed, 4.0, places=4)
        self.assertAlmostEqual(p.tca, 2.5, places=4)
        self.assertAlmostEqual(p.dmin, 0.0, places=3)
        self.assertAlmostEqual(p.heading_diff, 180.0)

    def test_diverging_pair_has_zero_tca(self):
        w = pd.DataFrame([row(1, 0.4, 0.5, -0.1, 0), row(2, 0.5, 0.5, 0.1, 0)])
        p = pair_features(w, 2000, 1000).iloc[0]
        self.assertEqual(p.tca, 0.0)
        self.assertLess(p.closing_speed, 0)

    def test_far_pairs_are_pruned(self):
        w = pd.DataFrame([row(1, 0.0, 0.5, 0, 0), row(2, 0.99, 0.5, 0, 0)])
        self.assertEqual(len(pair_features(w, 2000, 1000, max_dist_rel=5)), 0)


if __name__ == "__main__":
    unittest.main()
