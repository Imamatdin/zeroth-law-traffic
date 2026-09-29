import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from demo.app import create_app
from demo.video import InvalidVideo, inspect_video


class FakeProcessor:
    def __init__(self, block=False, fail=False):
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.fail, self.path = fail, None

    def process(self, path, progress):
        self.path = path
        self.release.wait(5)
        if self.fail:
            raise RuntimeError('private implementation detail')
        for stage in ('decoding', 'detecting', 'events', 'risk'):
            progress(stage, .5)
        return {'metadata': {'profile': {'name': 'test'}}, 'replay': {'video': path.name}}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.tmp.name) / 'sample.mp4'
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (96, 64))
        for _ in range(30):
            writer.write(np.zeros((64, 96, 3), np.uint8))
        writer.release()
        cls.body = cls.video.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def upload(self, client, name='x.mp4', body=None):
        return client.post('/jobs', files={'video': (name, self.body if body is None else body, 'video/mp4')})

    def wait(self, client, jid):
        for _ in range(200):
            result = client.get('/jobs/' + jid).json()
            if result['state'] != 'running':
                return result
            time.sleep(.01)
        self.fail('Job did not finish')

    def test_success_progress_busy_and_cleanup(self):
        processor = FakeProcessor(block=True)
        with TestClient(create_app(processor)) as c:
            try:
                self.assertEqual(c.get('/health').json()['max_seconds'], 60)
                r = self.upload(c)
                self.assertEqual(r.status_code, 202, r.text)
                jid = r.json()['job_id']
                self.assertEqual(self.upload(c).status_code, 409)
                self.assertEqual(c.get(f'/jobs/{jid}/result').status_code, 409)
                processor.release.set()
                state = self.wait(c, jid)
                self.assertEqual(state['state'], 'done')
                self.assertEqual(state['progress'], 1)
                result = c.get(f'/jobs/{jid}/result')
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['replay']['video'], 'x.mp4')
                self.assertFalse(processor.path.exists())
            finally:
                processor.release.set()

    def test_failure_is_clear_and_releases_slot_and_files(self):
        processor = FakeProcessor(fail=True)
        with TestClient(create_app(processor)) as c:
            jid = self.upload(c).json()['job_id']
            state = self.wait(c, jid)
            self.assertEqual(state['error']['code'], 'processing_failed')
            self.assertNotIn('private implementation', str(state))
            self.assertEqual(c.get(f'/jobs/{jid}/result').status_code, 422)
            self.assertFalse(processor.path.exists())
            self.assertFalse(c.get('/health').json()['busy'])

    def test_reject_bad_format_length_size_and_unknown_id(self):
        with TestClient(create_app(FakeProcessor(), max_bytes=2048)) as c:
            self.assertEqual(self.upload(c, 'x.avi').status_code, 415)
            self.assertEqual(self.upload(c, body=b'not an mp4').status_code, 422)
            self.assertEqual(self.upload(c, body=b'0' * 4096).status_code, 413)
            with patch('demo.app.inspect_video', side_effect=InvalidVideo('too_long', 'Video must be at most 60 seconds.')):
                r = self.upload(c)
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()['detail']['code'], 'too_long')
            self.assertEqual(c.get('/jobs/no-such-job').status_code, 404)
            self.assertFalse(c.get('/health').json()['busy'])

    def test_actual_metadata_duration_limit(self):
        with patch('demo.video.cv2.VideoCapture') as ctor:
            cap = ctor.return_value
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_FRAME_WIDTH: 96, cv2.CAP_PROP_FRAME_HEIGHT: 64,
                cv2.CAP_PROP_FRAME_COUNT: 1830}[prop]
            with self.assertRaisesRegex(InvalidVideo, '60 seconds'):
                inspect_video(self.video)
            cap.release.assert_called_once()

    def test_localhost_cors_and_result_expiry(self):
        with TestClient(create_app(FakeProcessor(), ttl=.03)) as c:
            allowed = c.options('/jobs', headers={'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'})
            self.assertEqual(allowed.headers['access-control-allow-origin'], 'http://localhost:5173')
            denied = c.options('/jobs', headers={'Origin': 'https://untrusted.example', 'Access-Control-Request-Method': 'POST'})
            self.assertNotIn('access-control-allow-origin', denied.headers)
            jid = self.upload(c).json()['job_id']
            self.wait(c, jid)
            time.sleep(.06)
            self.assertEqual(c.get(f'/jobs/{jid}/result').status_code, 404)

    def test_capacity_retains_completed_result_until_next_submission(self):
        with TestClient(create_app(FakeProcessor(), max_results=1)) as c:
            first = self.upload(c).json()['job_id']
            self.assertEqual(self.wait(c, first)['state'], 'done')
            self.assertEqual(c.get(f'/jobs/{first}/result').status_code, 200)
            second = self.upload(c).json()['job_id']
            self.assertEqual(self.wait(c, second)['state'], 'done')
            self.assertEqual(c.get(f'/jobs/{first}').status_code, 404)
            self.assertEqual(c.get(f'/jobs/{second}/result').status_code, 200)


if __name__ == '__main__':
    unittest.main()
