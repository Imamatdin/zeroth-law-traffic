"""Regression: a late object must confirm on the next observed frame."""
import unittest
import numpy as np
from src.contracts import Detections
from src.runtime.tracker import PrivateTracker


def detection(cls=3):
    return Detections(np.array([[100., 100., 180., 220.]]),
                      np.array([cls]), np.array([.9]))


class ObservationTrackerTests(unittest.TestCase):
    def test_late_car_and_person_at_every_supported_stride(self):
        for stride in range(3, 16):
            for cls in (0, 3):
                with self.subTest(stride=stride, cls=cls):
                    tracker = PrivateTracker(30)
                    tracker.update(Detections.empty(), 0)
                    tracker.update(Detections.empty(), stride)
                    self.assertEqual(tracker.update(detection(cls), 2 * stride), [])
                    rows = tracker.update(detection(cls), 3 * stride)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0].cls, cls)
                    self.assertEqual(rows[0].frame_idx, 3 * stride)
                    self.assertAlmostEqual(rows[0].t, stride / 10)
                    self.assertEqual(tracker.tracker._tracker.frame_id, 4)
                    self.assertAlmostEqual(tracker.frame_rate, 30 / stride)
                    self.assertEqual(tracker.tracker.args.track_buffer, round(60 / stride))

    def test_widening_preserves_confirmation_and_private_ids(self):
        a, b = PrivateTracker(30), PrivateTracker(30)
        for tracker in (a, b):
            for idx in (0, 3, 6, 9, 12):
                tracker.update(Detections.empty(), idx)
            self.assertEqual(tracker.update(detection(), 15), [])
        # This was Claude's repro: synthetic empty frames used to delete ID 1.
        self.assertEqual(a.update(detection(), 30)[0].track_id, 1)
        self.assertEqual(b.update(detection(), 21)[0].track_id, 1)
        self.assertEqual(a.update(detection(), 45)[0].track_id, 1)
        self.assertEqual(b.update(detection(), 27)[0].track_id, 1)

    def test_lost_buffer_is_seconds_including_stride_change(self):
        for stride in (3, 6, 15):
            with self.subTest(stride=stride):
                tracker = PrivateTracker(30)
                tracker.update(detection(), 0)
                tracker.update(detection(), 3)
                for idx in range(3 + stride, 64, stride):
                    tracker.update(Detections.empty(), idx)
                self.assertEqual(len(tracker.tracker._tracker.lost_stracks), 1)
                tracker.update(Detections.empty(), 63 + stride)
                from ultralytics.trackers.basetrack import TrackState
                self.assertTrue(all(t.state == TrackState.Removed
                                    for t in tracker.tracker._tracker.lost_stracks))
                self.assertEqual(len(tracker.tracker._tracker.removed_stracks), 1)

    def test_chronology_rejected_without_mutation(self):
        tracker = PrivateTracker(30)
        tracker.update(detection(), 3)
        for idx in (3, 0):
            with self.assertRaises(ValueError):
                tracker.update(detection(), idx)
        self.assertEqual(tracker.tracker._tracker.frame_id, 1)

    def test_widening_during_loss_preserves_elapsed_time(self):
        from ultralytics.trackers.basetrack import TrackState
        tracker = PrivateTracker(30)
        for idx in (0, 3, 6):
            tracker.update(detection(), idx)
        for idx in (9, 12, 27, 42, 57):
            tracker.update(Detections.empty(), idx)
            self.assertTrue(all(t.state == TrackState.Lost
                                for t in tracker.tracker._tracker.lost_stracks))
            self.assertEqual(len(tracker.tracker._tracker.lost_stracks), 1)
        tracker.update(Detections.empty(), 72)
        self.assertEqual(len(tracker.tracker._tracker.removed_stracks), 1)


if __name__ == '__main__':
    unittest.main()

