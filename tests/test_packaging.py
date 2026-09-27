import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.check_harness_output import check


class PackagingOutputTests(unittest.TestCase):
    def check_output(self, errors=None, risk=None, name='clip.mp4'):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / 'clip.mp4'
            video.touch()
            output = root / 'predictions.json'
            output.write_text(json.dumps({
                'videos': {name: {'events': [], 'risk': [[0, 0], [.1, 0]] if risk is None else risk}},
                'log': {name: {'errors': errors or [], 'duration': .2, 'total_sec': .3}}}))
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.return_value = 2
            with patch('scripts.check_harness_output.cv2.VideoCapture', return_value=capture):
                check(output, video)

    def test_complete_risk_with_no_events_is_valid(self):
        self.check_output()

    def test_timeout_or_exception_must_fail_even_if_format_is_valid(self):
        with self.assertRaisesRegex(ValueError, 'over time budget'):
            self.check_output(errors=['over time budget'], risk=[])

    def test_truncated_risk_must_fail(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.check_output(risk=[[0, 0]])

    def test_missing_video_must_fail(self):
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.check_output(name='other.mp4')
