import unittest

import numpy as np
import pandas as pd

from src.events import jaywalking
from event_fixtures import context, track

CFG = {"min_on_road_s": 1.0, "max_gap_s": 0.5, "crosswalk_margin_px": 20, "rider_overlap": 0.5,
       "margin_box_h": 0.0, "max_walk_speed_rel": 2.5}


def walk(track_id, t0, t1, p0, p1, cls=0, w=20.0, h=60.0):
    t = np.round(np.arange(t0, t1, 0.1), 3)
    f = (t - t0) / (t1 - t0)
    return track(track_id, cls, t, p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]), w, h)


class JaywalkingTests(unittest.TestCase):
    def test_crossing_mid_block_fires_from_curb_to_curb(self):
        # Sidewalk is x < 60; walk from x=30 to x=330 at 50 px/s along y=300 (no crossing there).
        (ev,) = jaywalking.detect(context([walk(1, 0.0, 6.0, (30, 300), (330, 300))], [(0, "red")]), CFG)
        # Confirmed beyond the 20 px margin, but the start is the step off the sidewalk (x > 60, t = 0.6 s).
        self.assertAlmostEqual(ev.start, 0.7, delta=0.11)
        self.assertAlmostEqual(ev.end, 6.0, delta=0.11)
        self.assertEqual(ev.track_ids, (1,))

    def test_leaving_the_road_ends_the_event(self):
        # Road -> refuge (450..550 x): the event ends when the foot reaches the refuge (x=450, t = 4.4 s).
        (ev,) = jaywalking.detect(context([walk(1, 0.0, 6.0, (230, 230), (530, 230))], [(0, "red")]), CFG)
        self.assertAlmostEqual(ev.start, 0.0, places=6)
        self.assertAlmostEqual(ev.end, 4.4, delta=0.11)

    def test_pedestrian_on_crossing_or_refuge_does_not_fire(self):
        on_xing = walk(1, 0.0, 8.0, (150, 535), (850, 535))
        on_refuge = walk(2, 0.0, 8.0, (470, 230), (530, 230))
        self.assertEqual(jaywalking.detect(context([on_xing, on_refuge], [(0, "red")]), CFG), [])

    def test_walking_along_crossing_edge_within_margin_does_not_fire(self):
        edge = walk(1, 0.0, 8.0, (150, 495), (850, 495))  # 15 px above the crossing's top edge
        self.assertEqual(jaywalking.detect(context([edge], [(0, "red")]), CFG), [])

    def test_short_step_off_does_not_fire(self):
        blip = walk(1, 0.0, 0.8, (150, 450), (160, 450))
        self.assertEqual(jaywalking.detect(context([blip], [(0, "red")]), CFG), [])

    def test_cyclist_and_vehicle_occupant_are_not_pedestrians(self):
        rider = walk(1, 0.0, 6.0, (200, 300), (400, 300))
        bike = walk(2, 0.0, 6.0, (200, 305), (400, 305), cls=1, w=30, h=40)
        driver = walk(3, 0.0, 6.0, (700, 350), (700, 350), w=10, h=20)
        car = walk(4, 0.0, 6.0, (700, 360), (700, 360), cls=3, w=120, h=80)
        self.assertEqual(jaywalking.detect(context([rider, bike, driver, car], [(0, "red")]), CFG), [])

    def test_fast_mover_is_not_a_pedestrian(self):
        fast = walk(1, 0.0, 2.0, (200, 300), (700, 300))  # 250 px/s, 60 px tall: ~4 body heights/s
        self.assertEqual(jaywalking.detect(context([fast], [(0, "red")]), CFG), [])

    def test_rider_whose_foot_is_on_an_undersized_bike_box(self):
        rider = walk(1, 0.0, 6.0, (300, 300), (300, 300), w=30, h=90)
        bike = walk(2, 0.0, 6.0, (300, 290), (300, 290), cls=2, w=30, h=40)  # foot 10 px below the box
        self.assertEqual(jaywalking.detect(context([rider, bike], [(0, "red")]), CFG), [])

    def test_margin_scales_with_person_height(self):
        near_edge = walk(1, 0.0, 8.0, (150, 480), (850, 480), h=80)  # 30 px above the crossing
        self.assertEqual(len(jaywalking.detect(context([near_edge], [(0, "red")]), CFG)), 1)
        cfg = {**CFG, "margin_box_h": 0.5}  # 40 px margin for an 80 px person
        self.assertEqual(jaywalking.detect(context([near_edge], [(0, "red")]), cfg), [])

    def test_box_cut_by_frame_bottom_has_no_foot_point(self):
        cut = walk(1, 0.0, 6.0, (300, 1000), (500, 1000))
        self.assertEqual(jaywalking.detect(context([cut], [(0, "red")]), CFG), [])

    def test_standing_at_the_curb_edge_within_margin_does_not_fire(self):
        from event_fixtures import scene
        ctx = context([walk(1, 0.0, 6.0, (300, 990), (500, 990), h=40)], [(0, "red")])
        ctx.scene.road = scene().road.copy()
        ctx.scene.road[2:, 1] = 1000 - 20  # road ends 20 px above the pedestrians' feet
        self.assertEqual(jaywalking.detect(ctx, CFG), [])

    def test_boundaries_use_exact_geometry_after_margin_confirmation(self):
        # Leaves the crossing's top edge (y=510) walking up at 10 px/s: the 20 px margin confirms it
        # only after 2 s, but the event starts when the foot leaves the crossing (t = 0.1 s).
        (ev,) = jaywalking.detect(context([walk(1, 0.0, 8.0, (500, 511), (500, 431))], [(0, "red")]), CFG)
        self.assertAlmostEqual(ev.start, 0.1, delta=0.11)
        self.assertAlmostEqual(ev.end, 8.0, delta=0.11)

    def test_duplicate_segments_after_extension_are_merged_by_postprocess(self):
        from src.postprocess.segments import postprocess
        evs = jaywalking.detect(context([walk(1, 0.0, 6.0, (30, 300), (330, 300))], [(0, "red")]), CFG)
        self.assertEqual(len(postprocess(evs, 60, 10)), 1)

    def test_walking_through_a_refuge_tip_keeps_one_event(self):
        # Road -> across the refuge (450..550 x, 200..260 y) in 2 s without stopping -> road again.
        cfg = {**CFG, "walk_through_gap_s": 2.5, "walk_through_max_still_s": 1.0}
        path = walk(1, 0.0, 12.0, (130, 230), (850, 230))
        (ev,) = jaywalking.detect(context([path], [(0, "red")]), cfg)
        self.assertAlmostEqual(ev.start, 0.0, places=6)
        self.assertAlmostEqual(ev.end, 12.0, delta=0.11)

    def test_stopping_on_a_refuge_ends_the_event(self):
        cfg = {**CFG, "walk_through_gap_s": 2.5, "walk_through_max_still_s": 1.0}
        t1 = walk(1, 0.0, 4.0, (130, 230), (490, 230))
        wait = walk(1, 4.1, 5.9, (490, 230), (490, 230))          # stands on the refuge ~2 s
        t2 = walk(1, 6.0, 10.0, (490, 230), (850, 230))
        evs = jaywalking.detect(context([pd.concat([t1, wait, t2])], [(0, "red")]), cfg)
        self.assertEqual(len(evs), 2)

    def test_edge_jitter_does_not_drag_the_start_back(self):
        # Walks along the crossing's top edge (y = 510) with feet jittering 5 px outside it, then
        # leaves it at t = 6 s. With a 10 px boundary margin the start is the real departure.
        cfg = {**CFG, "boundary_px": 10.0}
        along = walk(1, 0.0, 6.0, (150, 505), (450, 505))
        away = walk(1, 6.1, 12.0, (455, 500), (455, 250))
        (ev,) = jaywalking.detect(context([pd.concat([along, away])], [(0, "red")]), cfg)
        self.assertGreater(ev.start, 5.9)


if __name__ == "__main__":
    unittest.main()
