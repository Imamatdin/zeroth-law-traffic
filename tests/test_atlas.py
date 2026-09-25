import unittest

import numpy as np
import pandas as pd

from src.atlas.flow import Atlas, build_atlas


def lane(track_id, y, heading, n=40):
    return [dict(track_id=track_id, frame=i, t=i / 10, cls_major=3, gx=0.1 + 0.02 * i, gy=y,
                 heading=heading, speed_rel=2.0, stationary_s=0.0, age_s=1.0 + i / 10)
            for i in range(n)]


class AtlasTests(unittest.TestCase):
    def test_dominant_heading_and_deviation(self):
        rows = []
        for k in range(10):
            rows += lane(k, 0.3, 0.0) + lane(100 + k, 0.7, 180.0)
        atlas = Atlas(build_atlas(pd.DataFrame(rows), grid=(10, 10), min_samples=5))
        dev = atlas.heading_deviation(np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.7, 0.3]),
                                      np.array([0.0, 0.0, 175.0]))
        np.testing.assert_allclose(dev, [0.0, 180.0, 175.0], atol=1e-6)

    def test_uncovered_cells_return_nan(self):
        atlas = Atlas(build_atlas(pd.DataFrame(lane(1, 0.3, 0.0)), grid=(10, 10), min_samples=50))
        self.assertTrue(np.isnan(atlas.heading_deviation([0.5], [0.3], [0.0])).all())


if __name__ == "__main__":
    unittest.main()
