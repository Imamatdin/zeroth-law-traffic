import unittest

import numpy as np
import pandas as pd

from src.atlas.flow import Atlas, build_atlas
from src.events import wrong_way
from event_fixtures import context, track

CFG = {"vehicle_classes": [2, 3, 4, 5], "min_resultant": 0.8, "min_speed_rel": 0.5, "against_deg": 135,
       "entry_deg": 90, "return_deg": 60, "max_unjudged_backdate_s": 1.0, "min_persist_s": 1.5,
       "max_gap_s": 1.0, "min_travel_box_h": 2.0, "frame_edge_px": 8}


def drive(track_id, t0, points, speed=100.0, cls=3):
    """Constant-speed polyline through `points` (foot positions), sampled at 10 Hz."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_total = seg.sum()
    t = np.round(np.arange(t0, t0 + s_total / speed + 1e-9, 0.1), 3)
    s = (t - t0) * speed
    cum = np.concatenate([[0], np.cumsum(seg)])
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return track(track_id, cls, t, x, y)


def two_lane_atlas():
    """Lane A at y=300 flows right (+x); lane B at y=700 flows left; nothing between them."""
    rows = []
    for k in range(12):
        rows.append(drive(100 + k, 0.0, [(80, 300), (920, 300)]))
        rows.append(drive(200 + k, 0.0, [(920, 700), (80, 700)]))
    return Atlas(build_atlas(pd.concat(rows, ignore_index=True), grid=(10, 10), min_samples=10))


def ctx_with(tracks):
    ctx = context(tracks, [(0, "green")], duration=60.0)
    ctx.atlas = two_lane_atlas()
    return ctx


class WrongWayTests(unittest.TestCase):
    def test_driving_against_the_lane_fires_until_leaving_the_frame(self):
        (ev,) = wrong_way.detect(ctx_with([drive(1, 5.0, [(900, 300), (100, 300)])]), CFG)
        self.assertAlmostEqual(ev.start, 5.0, delta=0.21)
        self.assertAlmostEqual(ev.end, 13.1, delta=0.11)      # last sample 13.0 + one step
        self.assertTrue(ev.evidence["left_frame"])

    def test_normal_traffic_does_not_fire(self):
        tracks = [drive(1, 0.0, [(100, 300), (900, 300)]), drive(2, 0.0, [(900, 700), (100, 700)])]
        self.assertEqual(wrong_way.detect(ctx_with(tracks), CFG), [])

    def test_entering_the_opposing_lane_starts_the_event_and_returning_ends_it(self):
        # Correct in lane B, cut across into lane A (still heading left), drive against it, return to B.
        path = [(900, 700), (800, 700), (700, 300), (400, 300), (300, 700), (100, 700)]
        (ev,) = wrong_way.detect(ctx_with([drive(1, 0.0, path)]), CFG)
        t_enter_a = (100 + np.hypot(100, 400)) / 100            # arrives in lane A at ~5.1 s
        t_leave_a = t_enter_a + 3.0                            # leaves lane A at ~8.1 s
        self.assertGreaterEqual(ev.start, t_enter_a - 1.0 - 0.21)
        self.assertLessEqual(ev.start, t_enter_a + 0.21)
        self.assertGreater(ev.end, t_leave_a)
        self.assertLess(ev.end, t_leave_a + np.hypot(100, 400) / 100 + 0.5)
        self.assertFalse(ev.evidence["left_frame"])

    def test_brief_reversal_does_not_fire(self):
        path = [(200, 300), (500, 300), (420, 300)]            # 0.8 s backwards: jitter or reversing
        self.assertEqual(wrong_way.detect(ctx_with([drive(1, 0.0, path)]), CFG), [])

    def test_mixed_or_empty_cells_give_no_verdict(self):
        # Between the lanes (y=500) no atlas cell has samples: any direction is unjudged.
        self.assertEqual(wrong_way.detect(ctx_with([drive(1, 0.0, [(900, 500), (100, 500)])]), CFG), [])

    def test_pedestrians_and_boxes_at_the_frame_border_are_ignored(self):
        ped = drive(1, 0.0, [(900, 300), (100, 300)], speed=40.0, cls=0)
        cut = drive(2, 0.0, [(900, 300), (100, 300)])
        cut["y2"] = 1000.0                                     # box touches the frame bottom
        self.assertEqual(wrong_way.detect(ctx_with([ped, cut]), CFG), [])

    def test_without_an_atlas_nothing_fires(self):
        ctx = context([drive(1, 5.0, [(900, 300), (100, 300)])], [(0, "green")])
        self.assertEqual(wrong_way.detect(ctx, CFG), [])


if __name__ == "__main__":
    unittest.main()
