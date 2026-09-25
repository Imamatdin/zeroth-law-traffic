import unittest

import numpy as np
import pandas as pd

from src.features.tracks import compute_world


def track_rows(track_id, frames, fps, x_of_t, y=500.0, h=100.0, cls=3):
    rows = []
    for f in frames:
        t = f / fps
        x = x_of_t(t)
        rows.append(dict(frame=f, t=t, track_id=track_id, cls=cls, score=0.9,
                         x1=x - 50, y1=y - h, x2=x + 50, y2=y, fx=x, fy=y))
    return rows


class WorldModelTests(unittest.TestCase):
    fps = 30.0

    def test_constant_velocity_recovered_and_stationary_counted(self):
        frames = list(range(0, 300, 3))
        moving = track_rows(1, frames, self.fps, lambda t: 100 + 200 * t)
        parked = track_rows(2, frames, self.fps, lambda t: 1000.0)
        w = compute_world(pd.DataFrame(moving + parked), 1920, 1080)
        m = w[w.track_id == 1]
        np.testing.assert_allclose(m["vx"] * 1920, 200, rtol=1e-6)
        np.testing.assert_allclose(m["speed_rel"], 2.0, rtol=1e-6)
        p = w[w.track_id == 2]
        self.assertAlmostEqual(p["stationary_s"].iloc[-1], frames[-1] / self.fps)
        self.assertTrue((m["stationary_s"] == 0).all())

    def test_causal_mode_ignores_future_samples(self):
        frames = list(range(0, 240, 3))
        swerve = lambda t: 100 + 50 * t + (400 * (t - 4) ** 2 if t > 4 else 0)
        full = pd.DataFrame(track_rows(7, frames, self.fps, swerve))
        prefix = full[full.t <= 4.0]
        a = compute_world(full, 1920, 1080, mode="causal")
        b = compute_world(prefix, 1920, 1080, mode="causal")
        cols = ["vx", "vy", "ax", "ay", "speed_rel", "stationary_s", "cls_major"]
        pd.testing.assert_frame_equal(a.iloc[:len(b)][cols].reset_index(drop=True),
                                      b[cols].reset_index(drop=True))

    def test_centered_mode_does_look_ahead(self):
        frames = list(range(0, 240, 3))
        swerve = lambda t: 100 + 50 * t + (400 * (t - 4) ** 2 if t > 4 else 0)
        full = pd.DataFrame(track_rows(7, frames, self.fps, swerve))
        a = compute_world(full, 1920, 1080)
        b = compute_world(full[full.t <= 4.0], 1920, 1080)
        self.assertNotAlmostEqual(a["vx"].iloc[len(b) - 1], b["vx"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
