import unittest

import numpy as np

from src.scene.signal import classify_head, smooth_states

CELLS = {"red": (0, 0, 10, 10), "yellow": (0, 10, 10, 20), "green": (0, 20, 10, 30)}
BGR = {"red": (40, 40, 240), "yellow": (30, 200, 250), "green": (180, 230, 40)}


def head(*on):
    frame = np.full((30, 10, 3), 25, np.uint8)
    for name in on:
        x1, y1, x2, y2 = CELLS[name]
        frame[y1:y2, x1:x2] = BGR[name]
    return frame


class SignalTests(unittest.TestCase):
    def test_states_by_lamp_position(self):
        self.assertEqual(classify_head(head("red"), CELLS), "red")
        self.assertEqual(classify_head(head("red", "yellow"), CELLS), "red")
        self.assertEqual(classify_head(head("yellow"), CELLS), "yellow")
        self.assertEqual(classify_head(head("green"), CELLS), "green")

    def test_ambiguous_or_dark_is_unknown(self):
        self.assertEqual(classify_head(head(), CELLS), "unknown")
        self.assertEqual(classify_head(head("red", "green"), CELLS), "unknown")

    def test_smoothing_removes_short_flicker(self):
        s = ["red"] * 5 + ["unknown"] * 2 + ["red"] * 5 + ["green"] * 6
        self.assertEqual(smooth_states(s, 3), ["red"] * 12 + ["green"] * 6)

    def test_flashing_green_stays_green(self):
        s = ["green"] * 6 + (["unknown"] * 5 + ["green"] * 5) * 3 + ["yellow"] * 30
        self.assertEqual(smooth_states(s, 3), ["green"] * 36 + ["yellow"] * 30)


if __name__ == "__main__":
    unittest.main()
