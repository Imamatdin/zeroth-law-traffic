import unittest

import numpy as np
import pandas as pd

from src.features.stitching import StitchConfig, fuse_riders, stitch


def frag(track_id, cls, t0, t1, x0, vx, y=500.0, h=60.0, step=0.1):
    t = np.round(np.arange(t0, t1 + 1e-9, step), 3)
    x = x0 + vx * (t - t0)
    return pd.DataFrame({"frame": np.round(t * 10).astype(int), "t": t, "track_id": track_id, "cls": cls,
                         "score": 0.9, "x1": x - 20, "y1": y - h, "x2": x + 20, "y2": y, "fx": x, "fy": y})


def ids(out):
    return out.groupby("raw_track_id")["track_id"].first().to_dict()


class StitchingTests(unittest.TestCase):
    def test_occlusion_gap_on_constant_velocity_is_joined(self):
        a = frag(1, 3, 0, 4, 100, 50)
        b = frag(2, 3, 5.5, 9, 100 + 50 * 5.5, 50)   # 1.5 s hidden, continues on the same line
        out, links = stitch(pd.concat([a, b]))
        self.assertEqual(ids(out), {1: 1, 2: 1})
        self.assertEqual(len(links), 1)

    def test_rider_seen_as_motorcycle_then_person_is_joined(self):
        a = frag(24, 2, 0, 4, 100, 30)
        b = frag(1088, 0, 4.6, 8, 100 + 30 * 4.6, 30)
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {24: 24, 1088: 24})

    def test_car_and_pedestrian_never_join(self):
        a = frag(1, 3, 0, 4, 100, 30)
        b = frag(2, 0, 4.3, 8, 100 + 30 * 4.3, 30)
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {1: 1, 2: 2})

    def test_far_or_late_or_overlapping_fragments_are_not_joined(self):
        a = frag(1, 3, 0, 4, 100, 50)
        far = frag(2, 3, 4.5, 8, 900, 50)
        late = frag(3, 3, 8.0, 10, 100 + 50 * 8.0, 50)          # 4 s gap > max_gap_s
        overlap = frag(4, 3, 3.5, 4.5, 100 + 50 * 3.5, 50)      # starts before a ends
        self.assertEqual(ids(stitch(pd.concat([a, far, late, overlap]))[0]), {1: 1, 2: 2, 3: 3, 4: 4})

    def test_size_mismatch_is_not_joined(self):
        a = frag(1, 3, 0, 4, 100, 50, h=60)
        b = frag(2, 3, 4.3, 8, 100 + 50 * 4.3, 50, h=150)
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {1: 1, 2: 2})

    def test_closest_candidate_wins_and_chains_form(self):
        a = frag(1, 3, 0, 2, 100, 50)
        good = frag(2, 3, 2.3, 4, 100 + 50 * 2.3, 50)
        worse = frag(3, 3, 2.3, 4, 100 + 50 * 2.3 + 25, 50)
        nxt = frag(4, 3, 4.4, 6, 100 + 50 * 4.4, 50)
        self.assertEqual(ids(stitch(pd.concat([a, good, worse, nxt]))[0]), {1: 1, 2: 1, 3: 3, 4: 1})

    def test_stationary_object_hidden_long_is_joined_without_drift(self):
        a = frag(1, 3, 0, 4, 400, 0.5)
        b = frag(2, 3, 6.5, 9, 401.5, 0.0)
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {1: 1, 2: 1})

    def test_output_has_unique_frame_track_rows_and_is_deterministic(self):
        parts = [frag(i, 3, 2 * i, 2 * i + 1.7, 100 + 100 * i, 50) for i in range(1, 6)]
        df = pd.concat(parts)
        out1, _ = stitch(df)
        out2, _ = stitch(df.sample(frac=1, random_state=0))
        self.assertFalse(out1.duplicated(["frame", "track_id"]).any())
        pd.testing.assert_frame_equal(out1.sort_values(["raw_track_id", "t"]).reset_index(drop=True),
                                      out2.sort_values(["raw_track_id", "t"]).reset_index(drop=True))

    def test_rider_track_is_fused_into_its_bike_and_then_continues_it(self):
        bike = frag(24, 2, 0, 4, 100, 30, h=60)
        rider = frag(21, 0, 0, 4, 100, 30, y=470, h=90)       # foot 30 px up, inside the bike box
        after = frag(1088, 0, 4.4, 8, 100 + 30 * 4.4, 30, y=470, h=90)
        walker = frag(7, 0, 0, 4, 600, 0)                     # unrelated pedestrian stays separate
        fused, info = fuse_riders(pd.concat([bike, rider, after, walker]))
        self.assertEqual([(f["rider"], f["bike"]) for f in info], [(21, 24)])
        self.assertFalse((fused["track_id"] == 21).any())
        out, _ = stitch(fused)
        self.assertEqual(ids(out), {24: 24, 1088: 24, 7: 7})

    def test_pedestrian_passing_a_bike_briefly_is_not_fused(self):
        bike = frag(2, 1, 0, 6, 300, 0, h=60)
        ped = frag(3, 0, 0, 6, 200, 40, h=90)                 # walks past the parked bike
        self.assertEqual(fuse_riders(pd.concat([bike, ped]))[1], [])

    def test_person_waiting_beside_a_bike_then_walking_away_is_not_fused(self):
        bike = frag(33, 1, 0, 10, 300, 0, h=60)
        waiting = frag(16, 0, 0, 3, 300, 0, y=480, h=90)
        walking = frag(16, 0, 3.1, 12, 300, 60, y=480, h=90)   # the same person, 9 s of walking
        self.assertEqual(fuse_riders(pd.concat([bike, waiting, walking]))[1], [])

    def test_person_with_a_few_bicycle_frames_is_not_a_bike(self):
        ped = frag(20, 0, 0, 5, 300, 20, h=90)
        ped.loc[ped.index[:5], "cls"] = 1
        other = frag(368, 0, 0, 5, 300, 20, y=470, h=90)
        self.assertEqual(fuse_riders(pd.concat([ped, other]))[1], [])

    def test_moving_fragments_with_opposite_headings_are_not_joined(self):
        a = frag(1, 3, 0, 4, 100, 50)
        b = frag(2, 3, 4.3, 8, 100 + 50 * 4.3, -50)   # starts at the prediction but drives back
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {1: 1, 2: 2})

    def test_moving_fragments_with_very_different_speeds_are_not_joined(self):
        a = frag(1, 3, 0, 4, 100, 30)
        b = frag(2, 3, 4.3, 8, 100 + 30 * 4.3, 150)
        self.assertEqual(ids(stitch(pd.concat([a, b]))[0]), {1: 1, 2: 2})

    def test_fragment_leaving_through_the_frame_edge_is_not_continued(self):
        a = frag(1, 3, 0, 4, 100, 50)
        a.loc[a.index[-1], "x2"] = 999.0                # last box touches the right edge of a 1000 px frame
        b = frag(2, 3, 4.3, 8, 100 + 50 * 4.3, 50)
        self.assertEqual(ids(stitch(pd.concat([a, b]), width=1000, height=1000)[0]), {1: 1, 2: 2})


if __name__ == "__main__":
    unittest.main()
