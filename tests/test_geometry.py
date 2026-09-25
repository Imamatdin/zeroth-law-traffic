import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.scene.geometry import Scene, crosses_segment, point_in_polygon, side_of_line

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)

MAP = """
coords: normalized
road: [[0, 0], [1, 0], [1, 1], [0, 1]]
islands:
  - {id: isl, polygon: [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]}
crosswalks:
  - {id: cw, polygon: [[0.0, 0.8], [1.0, 0.8], [1.0, 0.9], [0.0, 0.9]]}
stop_lines:
  - {id: s1, points: [[0.1, 0.7], [0.9, 0.7]]}
signals:
  - {id: sig, roi: [0.1, 0.1, 0.2, 0.3]}
"""


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self):
        pts = np.array([[5, 5], [15, 5], [-1, 5], [9.9, 9.9]])
        self.assertEqual(point_in_polygon(pts, SQUARE).tolist(), [True, False, False, True])

    def test_concave_polygon(self):
        u_shape = np.array([[0, 0], [9, 0], [9, 9], [6, 9], [6, 3], [3, 3], [3, 9], [0, 9]], float)
        self.assertEqual(point_in_polygon(np.array([[4.5, 6], [1.5, 6]]), u_shape).tolist(), [False, True])

    def test_side_and_crossing(self):
        self.assertEqual(side_of_line(np.array([[0, 1], [0, -1]]), [-1, 0], [1, 0]).tolist(), [1, -1])
        self.assertTrue(crosses_segment([0, -1], [0, 1], [-1, 0], [1, 0]))
        self.assertFalse(crosses_segment([5, -1], [5, 1], [-1, 0], [1, 0]))
        self.assertFalse(crosses_segment([0, -2], [0, -1], [-1, 0], [1, 0]))

    def test_scene_scales_to_video_and_excludes_islands(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.yaml"
            path.write_text(MAP, encoding="utf-8")
            scene = Scene.load(path, 3840, 2160)
        pts = np.array([[0.5 * 3840, 0.5 * 2160], [0.2 * 3840, 0.2 * 2160]])
        self.assertEqual(scene.on_carriageway(pts).tolist(), [False, True])
        self.assertEqual(scene.in_kind(np.array([[1920, 0.85 * 2160]]), "crosswalks").tolist(), [True])
        self.assertEqual(scene.signals["sig"]["roi_px"], (384, 216, 768, 648))
        np.testing.assert_allclose(scene.stop_lines["s1"][0], [384, 0.7 * 2160])

    def test_rejects_out_of_range_points(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.yaml"
            path.write_text("coords: normalized\nroad: [[0, 0], [1.2, 0], [1, 1]]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Scene.load(path, 100, 100)


if __name__ == "__main__":
    unittest.main()
