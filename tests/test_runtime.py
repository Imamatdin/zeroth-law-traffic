import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import cv2
import numpy as np
from src.contracts import Detections
from src.runtime.budget import BudgetGuard
from src.perception.detector import UltralyticsDetector
from src.runtime import decode
from src.runtime.decode import _read_frame, frames, metadata, prescale, restore


class RuntimeTests(unittest.TestCase):
    def test_budget_widens_causally(self):
        now = [0.0]
        guard = BudgetGuard("B", dict(n_frames=3000, fps=30), clock=lambda: now[0])
        self.assertTrue(guard.due(0))
        now[0] = 1.0
        guard.observed(0, 1.0)
        self.assertEqual(guard.stride, 3)  # One slow call never changes stride.
        for idx in (3, 6, 9, 12):
            now[0] += 1
            guard.observed(idx, 1.0)
        self.assertGreater(guard.stride, 3)
        self.assertLessEqual(guard.stride, 15)
        self.assertEqual(guard.stride % 3, 0)
        self.assertFalse(guard.due(3))
        self.assertLess(sum(BudgetGuard.SHARES.values()), 3)

    def test_restore_nonuniform_rounding(self):
        d = Detections(np.array([[1, 2, 100, 200.]]), np.array([3]), np.array([.9]))
        native = restore(d, (1080, 1920, 3), dict(width=3840, height=2161))
        np.testing.assert_allclose(native.xyxy, [[2, 4.00185185, 200, 400.185185]], rtol=1e-6)
        self.assertEqual(d.xyxy[0, 0], 1)

    def test_partial_raw_read_and_truncation(self):
        class ShortPipe(io.BytesIO):
            def read(self, n):
                return super().read(min(2, n))
        self.assertEqual(_read_frame(ShortPipe(b"abcdef"), 6), b"abcdef")
        with self.assertRaises(OSError):
            _read_frame(ShortPipe(b"abc"), 6)

    def test_ffmpeg_and_fallback_original_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "test.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(11):
                writer.write(np.full((48, 64, 3), i * 10, np.uint8))
            writer.release()
            meta = metadata(path)
            a = list(frames(path, meta, backend="ffmpeg"))
            with patch("imageio_ffmpeg.get_ffmpeg_exe", side_effect=RuntimeError("missing binary")):
                b = list(frames(path, meta, backend="ffmpeg"))
            with patch.dict("os.environ", {}, clear=True):
                c = list(frames(path, meta))
            for stream in (a, b, c):
                self.assertEqual([i for i, _ in stream], [0, 3, 6, 9])
            self.assertEqual(a[0][1].shape, (48, 64, 3))
            self.assertEqual(c[0][1].shape, (48, 64, 3))

    def test_default_backend_is_opencv_and_env_selects_ffmpeg(self):
        with patch.object(decode, "opencv_frames", return_value=iter([("cv", None)])),                 patch.object(decode, "ffmpeg_frames", return_value=iter([("ff", None)])):
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(next(frames("v", {}))[0], "cv")
            with patch.dict("os.environ", {"ZLT_DECODE": "ffmpeg"}):
                self.assertEqual(next(frames("v", {}))[0], "ff")
            with patch.dict("os.environ", {"ZLT_DECODE": "gstreamer"}), self.assertRaises(ValueError):
                next(frames("v", {}))

    def test_opencv_prescale_matches_cache_builder(self):
        builder = SimpleNamespace(prescale_width=1920)
        rng = np.random.default_rng(0)
        for h, w in ((2160, 3840), (1441, 2560), (1080, 1920), (48, 64)):
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            expected, _ = UltralyticsDetector._prescale(builder, frame)
            np.testing.assert_array_equal(prescale(frame), expected)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "weights/yolo11m.pt").is_file(), "local model absent")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.runtime import pipeline
        cls.p = pipeline

    def test_risk_skips_without_inference_or_io_and_owns_state(self):
        p = self.p
        detectors = [Mock(), Mock()]
        for d in detectors:
            d.predict.return_value = [Detections.empty()]
        meta = dict(fps=30, n_frames=3000, width=64, height=48, video_id="in-memory")
        with patch.object(p, "new_detector", side_effect=detectors), patch("cv2.VideoCapture", side_effect=AssertionError("risk video IO")):
            a, b = p.RiskEstimator(), p.RiskEstimator()
            a.reset(meta)
            b.reset(meta)
            self.assertIsNot(a.tracker, b.tracker)
            self.assertEqual(a.step(np.zeros((48, 64, 3), np.uint8), 0), 0)
            a.last_score = .4
            self.assertEqual(a.step(None, 1 / 30), .4)
            self.assertEqual(detectors[0].predict.call_count, 1)
            self.assertEqual(detectors[1].predict.call_count, 0)
            with self.assertRaises(ValueError):
                a.step(None, 0)

    def test_tracker_ids_are_independent_when_interleaved(self):
        p = self.p
        a, b = p.PrivateTracker(30), p.PrivateTracker(30)
        det = Detections(np.array([[10, 10, 30, 30.]]), np.array([3]), np.array([.9]))
        self.assertEqual(a.update(det, 0)[0].track_id, 1)
        self.assertEqual(b.update(det, 0)[0].track_id, 1)
        self.assertEqual(a.update(det, 3)[0].track_id, 1)

    def test_pipeline_native_geometry_and_enabled_registry(self):
        from src.contracts import Event
        from src.events import registry
        from src.scene.geometry import Scene
        p = self.p
        meta = dict(fps=30, n_frames=300, width=3840, height=2160, video_id="synthetic")
        det = Detections(np.array([[10, 20, 30, 40.]]), np.array([3]), np.array([.9]))
        detector = Mock()
        detector.predict.return_value = [det]
        image = np.zeros((1080, 1920, 3), np.uint8)
        seen = []
        def hook(tracks, width, height):
            self.assertEqual((width, height), (3840, 2160))
            np.testing.assert_allclose(tracks.x1, 20)
            seen.append("stitch")
            return tracks
        def event(ctx, cfg):
            self.assertEqual(seen, ["stitch"])
            self.assertEqual(ctx.samples.frame.tolist(), [0, 3])
            np.testing.assert_allclose(ctx.samples.x2, 60)
            return [Event("accident", 0.0, 0.2, 0.9, (), {})]
        disabled = Mock(side_effect=AssertionError("disabled engine called"))
        with patch.object(p, "metadata", return_value=meta), patch.object(p, "frames", return_value=((i, image) for i in (0, 3))), \
             patch.object(p, "new_detector", return_value=detector), patch.object(p.Scene, "load", return_value=Scene(3840, 2160)), \
             patch.object(p, "stitch_if_present", side_effect=hook), patch.object(p, "load_config", return_value={"accident": {"enabled": True}}), \
             patch.dict(registry.ENGINES, {"accident": event, "near_miss": disabled}, clear=True):
            self.assertEqual(p.detect_events("in-memory"), [[0.0, 0.2, "accident"]])
        disabled.assert_not_called()

    def test_empty_perception_runs_without_world_sort_failure(self):
        from src.scene.geometry import Scene
        p = self.p
        meta = dict(fps=30, n_frames=300, width=64, height=48, video_id="empty")
        detector = Mock()
        detector.predict.return_value = [Detections.empty()]
        with patch.object(p, "metadata", return_value=meta), patch.object(p, "frames", return_value=((0, np.zeros((48, 64, 3), np.uint8)) for _ in range(1))), \
             patch.object(p, "new_detector", return_value=detector), patch.object(p.Scene, "load", return_value=Scene(64, 48)), \
             patch.object(p, "stitch_if_present", side_effect=lambda tracks, w, h: tracks):
            self.assertEqual(p.detect_events("in-memory"), [])

    def test_stitch_hook_absent_present_and_errors_propagate(self):
        p = self.p
        tracks = object()
        with patch.object(p.importlib.util, "find_spec", return_value=None):
            self.assertIs(p.stitch_if_present(tracks, 3840, 2160), tracks)
        module = Mock()
        module.prepare_tracks.return_value = (tracks, {"links": []})
        with patch.object(p.importlib.util, "find_spec", return_value=object()), patch.object(p.importlib, "import_module", return_value=module):
            self.assertIs(p.stitch_if_present(tracks, 3840, 2160), tracks)
            module.prepare_tracks.assert_called_once_with(tracks, width=3840, height=2160)
            module.prepare_tracks.side_effect = ValueError("broken hook")
            with self.assertRaises(ValueError):
                p.stitch_if_present(tracks, 3840, 2160)


if __name__ == "__main__":
    unittest.main()
