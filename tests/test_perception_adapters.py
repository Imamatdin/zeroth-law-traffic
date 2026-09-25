import importlib.util
import unittest

import numpy as np

from src.contracts import Detections

HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


class ClassMappingTests(unittest.TestCase):
    def test_coco_to_internal_ids(self):
        from src.perception.detector import COCO_TO_INTERNAL

        # person, bicycle, motorcycle, car, bus, truck
        self.assertEqual([COCO_TO_INTERNAL[c] for c in (0, 1, 3, 2, 5, 7)], [0, 1, 2, 3, 4, 5])


@unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics not installed")
class ByteTrackAdapterTests(unittest.TestCase):
    def test_ids_persist_for_a_moving_box_and_boxes_are_detections(self):
        from src.perception.tracker import ByteTrackTracker

        tracker = ByteTrackTracker(fps=30.0, stride=3)
        ids = set()
        for step in range(12):
            frame = step * 3
            box = np.array([[100 + 4 * step, 200, 180 + 4 * step, 260]], dtype=float)
            tracks = tracker.update(Detections(box, np.array([3]), np.array([0.9])), frame)
            if step:
                self.assertEqual(len(tracks), 1)
                self.assertEqual(tracks[0].xyxy, tuple(box[0]))
                self.assertAlmostEqual(tracks[0].t, frame / 30.0)
                ids.add(tracks[0].track_id)
        self.assertEqual(len(ids), 1)

    def test_reset_starts_fresh(self):
        from src.perception.tracker import ByteTrackTracker

        tracker = ByteTrackTracker(fps=30.0, stride=1)
        box = Detections(np.array([[0, 0, 50, 50]], float), np.array([0]), np.array([0.9]))
        for i in range(3):
            tracker.update(box, i)
        tracker.reset()
        restarted = tracker.update(box, 0)
        self.assertEqual([t.track_id for t in restarted], [1])

    def test_buffer_is_defined_in_seconds(self):
        from src.perception.tracker import ByteTrackTracker

        self.assertEqual(ByteTrackTracker(29.97, 3, buffer_s=2.0).args.track_buffer, 20)
        self.assertEqual(ByteTrackTracker(29.97, 1, buffer_s=2.0).args.track_buffer, 60)


if __name__ == "__main__":
    unittest.main()
