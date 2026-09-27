from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from demo.engine import Engine, Profile
from demo.scene_check import check_scene, REFERENCE
from src.contracts import Detections


class EngineTests(unittest.TestCase):
    def test_scene_gate_resolution_and_unrelated_image(self):
        ref = cv2.imread(str(REFERENCE))
        self.assertTrue(check_scene(ref, 3840, 2160)['matches'])
        self.assertFalse(check_scene(ref, 1920, 1080)['matches'])
        self.assertFalse(check_scene(np.zeros_like(ref), 3840, 2160)['matches'])

    def test_generic_video_still_has_detection_tracking_and_risk_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (96, 64))
            for _ in range(30):
                writer.write(np.zeros((64, 96, 3), np.uint8))
            writer.release()
            engine = Engine.__new__(Engine)
            engine.profile, engine.tracker_config, engine.config, engine.atlas = Profile(), {}, {}, {}
            engine.detector = Mock()
            engine.detector.predict.return_value = [Detections(np.array([[20., 20., 45., 45.]]), np.array([3]), np.array([.9]))]
            stages = []
            with patch('demo.engine.run_engines', side_effect=AssertionError('Must not run camera rules')):
                result = engine.process(path, lambda stage, _: stages.append(stage))
            self.assertTrue(result['metadata']['scene_events_skipped'])
            self.assertEqual(result['events']['segments'], [])
            self.assertEqual(result['signal']['signals'], {})
            self.assertGreater(len(result['replay']['tracks']), 0)
            self.assertEqual(len(result['risk']['risk']), 4)
            self.assertEqual(len(result['detections']['rows']), 4)
            self.assertEqual(set(stages), {'decoding', 'detecting', 'events', 'risk'})
            # Same fields as web/public/data/C3905 documents.
            self.assertEqual(set(result['replay']), {'video','fps','duration','width','height','q','t','classes','units','tracks'})
            self.assertEqual(set(result['replay']['tracks'][0]), {'id','cls','raw','k','x','y','w','h','v','hd','st'})
            self.assertEqual(set(result['events']), {'status','mode','classes','enabled_in_submission','submitted','raw_events','segments'})
            self.assertEqual(set(result['risk']), {'status','threshold','horizon_s','alarm_starts','percentiles','t','risk','raw','smoothed','pairs','note'})
            self.assertEqual(set(result['signal']), {'signals','near_stop_crossings','note'})
            contract = json.loads((Path(__file__).parent / 'web_export_fields.json').read_text())
            for name in ('replay', 'events', 'risk', 'signal'):
                self.assertEqual(sorted(result[name]), contract[name])
            self.assertEqual(sorted(result['replay']['tracks'][0]), contract['track'])


if __name__ == '__main__':
    unittest.main()
