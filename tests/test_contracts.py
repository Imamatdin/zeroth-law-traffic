import unittest

import numpy as np

from evaluate import OFFICIAL_CLASSES
from src.contracts import Detections, EVENT_LABELS, Event, Interaction, RiskSample, TrackState


class ContractTests(unittest.TestCase):
    def test_labels_match_unchanged_official_evaluator(self):
        self.assertEqual(EVENT_LABELS, set(OFFICIAL_CLASSES))

    def test_empty_frame_keeps_shapes(self):
        det = Detections.empty()
        self.assertEqual(det.xyxy.shape, (0, 4))
        self.assertEqual(det.cls.dtype, np.int64)

    def test_rejects_unmapped_classes_invalid_boxes_and_nonfinite_scores(self):
        for box, cls, score in [([1, 2, 0, 4], 0, .9), ([0, 0, 4, 4], 7, .9),
                                ([0, 0, 4, 4], 0, float("nan"))]:
            with self.subTest(box=box, cls=cls, score=score), self.assertRaises(ValueError):
                Detections(np.array([box]), np.array([cls]), np.array([score]))

    def test_arrays_cannot_be_mutated_by_callers_input_buffer(self):
        boxes = np.array([[0, 0, 4, 4]], dtype=np.float32)
        det = Detections(boxes, np.array([3]), np.array([.5]))
        boxes[0, 2] = 100
        self.assertEqual(det.xyxy[0, 2], 4)

    def test_event_boundaries_and_risk_values(self):
        for start, end in [(2, 2), (3, 2), (-1, 2), (0, float("inf"))]:
            with self.assertRaises(ValueError):
                Event("accident", start, end, .5)
        for value in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                RiskSample(0, value)

    def test_footpoint_and_pair_identity(self):
        track = TrackState(1, 0, 0, 3, .5, (10, 20, 30, 40))
        self.assertEqual(track.foot_point, (20, 40))
        with self.assertRaises(ValueError):
            Interaction(0, 1, 1, 0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
