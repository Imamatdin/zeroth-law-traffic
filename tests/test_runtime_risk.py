"""Part B through the submission's RiskEstimator: detections -> its own ByteTrack -> causal risk model."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.contracts import Detections
from src.scene.geometry import Scene

W, H, FPS = 3840, 2160, 29.97
H_CAR, W_CAR = 150.0, 250.0
HAS_MODEL = (Path(__file__).resolve().parents[1] / "weights/yolo11m.pt").is_file()


def car_box(x, y):
    return [x - W_CAR / 2, y - H_CAR, x + W_CAR / 2, y]


def course(t, v=150.0, meet=(2000.0, 400.0), t_meet=10.0, swerve_after=None):
    """Two cars whose foot points meet at `meet` at t_meet; car B turns away after `swerve_after`."""
    boxes = [car_box(meet[0] - v * (t_meet - t), meet[1])]
    if swerve_after is None or t <= swerve_after:
        bx, by = meet[0], meet[1] + v * (t_meet - t)
    else:
        by = meet[1] + v * (t_meet - swerve_after)
        bx = meet[0] + 3 * v * (t - swerve_after)
    boxes.append(car_box(bx, by))
    inside = [b for b in boxes if b[0] >= 0 and b[1] >= 0 and b[2] <= W and b[3] <= H]
    if not inside:
        return Detections.empty()
    return Detections(np.array(inside, float), np.full(len(inside), 3), np.full(len(inside), 0.9))


class FrameDetector:
    """Stands in for YOLO: the 'frame' carries (scenario, frame index); boxes come from `course`."""

    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.calls = 0

    def predict(self, frames):
        self.calls += 1
        scenario, idx = frames[0].ravel()[:2]
        return [self.scenarios[int(scenario)](idx / FPS)]


@unittest.skipUnless(HAS_MODEL, "local model absent")
class RiskEstimatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.runtime import pipeline
        cls.p = pipeline

    def run_estimator(self, scenarios, scenario_of, n_frames, scene=None):
        p = self.p
        det = FrameDetector(scenarios)
        meta = dict(fps=FPS, n_frames=n_frames, width=W, height=H, video_id="synthetic")
        forbidden = Mock(side_effect=AssertionError("Part B must not use video IO, stitching or Part A"))
        # The wall-clock guard is Codex's (tested with a mocked clock); here it must not thin out updates
        # on a slow machine, so these tests stay about the risk wiring.
        with patch.object(p.BudgetGuard, "SHARES", {"A": 1e6, "B": 1e6}), \
             patch.object(p.BudgetGuard, "DECODE", {"A": 0.0, "B": 0.0}), \
             patch.object(p, "new_detector", return_value=det), \
             patch.object(p.Scene, "load", return_value=scene or Scene(W, H)), \
             patch.object(p, "ATLAS_DATA", None), \
             patch("cv2.VideoCapture", forbidden), patch.object(p, "stitch_if_present", forbidden), \
             patch.object(p, "detect_events", forbidden):
            est = p.RiskEstimator()
            est.reset(meta)
            risks = []
            for idx in range(n_frames):
                frame = np.array([scenario_of(idx), idx], dtype=np.int64)
                risks.append(est.step(frame, idx / FPS))
        return np.array(risks), det, est

    def test_collision_course_raises_risk_before_contact(self):
        n = int(10.0 * FPS)
        risks, det, _ = self.run_estimator({0: course}, lambda i: 0, n)
        t = np.arange(n) / FPS
        contact = 10.0 - min(W_CAR, H_CAR) / 150.0          # first box overlap
        self.assertTrue(np.all((risks >= 0) & (risks <= 1)))
        self.assertGreater(risks.max(), 0.5)
        first = t[np.argmax(risks >= 0.5)]
        self.assertGreaterEqual(contact - first, 2.5)
        self.assertLessEqual(contact - first, 5.0)
        self.assertLessEqual(det.calls, n // 3 + 1)          # stride 3: skipped frames return the last score

    def test_causality_identical_frame_prefix_gives_identical_risk(self):
        cut = int(6.0 * FPS)
        n = int(10.0 * FPS)
        scenarios = {0: course, 1: lambda t: course(t, swerve_after=6.0)}
        r_a, _, _ = self.run_estimator(scenarios, lambda i: 0, n)
        r_b, _, _ = self.run_estimator(scenarios, lambda i: 0 if i <= cut else 1, n)
        np.testing.assert_array_equal(r_a[:cut + 1], r_b[:cut + 1])
        self.assertFalse(np.array_equal(r_a[cut + 1:], r_b[cut + 1:]))

    def test_signal_is_read_from_part_b_own_frame(self):
        p = self.p
        scene = Scene(W, H)
        scene.signals["sig"] = {"lamps_px": {"red": (0, 0, 4, 4), "yellow": (0, 4, 4, 8), "green": (0, 8, 4, 12)}}
        seen = []

        class Spy:
            def reset(self, meta):
                pass

            def update(self, tracks, t, signals):
                seen.append(signals)
                return 0.0

        det = FrameDetector({0: lambda t: Detections.empty()})
        frame = np.zeros((12, 4, 3), np.uint8)
        frame[0:4] = (40, 40, 240)                            # red lamp lit in this frame
        with patch.object(p, "new_detector", return_value=det), patch.object(p.Scene, "load", return_value=scene), \
             patch.object(p, "RiskModel") as model:
            model.return_value = Spy()
            model.return_value.scene = scene
            det.predict = lambda frames: [Detections.empty()]
            est = p.RiskEstimator()
            est.reset(dict(fps=FPS, n_frames=30, width=W, height=H, video_id="synthetic"))
            est.step(frame, 0.0)
        self.assertEqual(seen, [{"sig": "red"}])


if __name__ == "__main__":
    unittest.main()
