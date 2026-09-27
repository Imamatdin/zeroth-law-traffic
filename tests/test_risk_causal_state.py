import csv
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from src.anticipation.risk import RiskModel
from src.contracts import TrackState
from risk_fixtures import mover, run, scene

ROOT = Path(__file__).resolve().parents[1]


class CausalRiskStateTests(unittest.TestCase):
    def test_c3905_track20_never_becomes_wrong_way_motorcycle(self):
        # Raw, unstitched observations, including the 1.2 s gap before flicker.
        with (ROOT / 'tests/fixtures/c3905_track20.csv').open() as f:
            rows = {int(r['frame']): r for r in csv.DictReader(f)}
        model = RiskModel.from_files(ROOT / 'configs/camera.yaml', 3840, 2160,
                                     ROOT / 'configs/atlas.json')
        checked = 0
        for idx in range(0, max(rows) + 1, 3):
            t = idx / (30000 / 1001)
            row = rows.get(idx)
            tracks = [] if row is None else [TrackState(
                20, idx, t, int(row['cls']), float(row['score']),
                tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')))]
            model.update(tracks, t)
            state = model._state_arrays(t)
            if state is not None:
                self.assertEqual(state['cls'].tolist(), [0])
                self.assertFalse(model._wrong_way(state, t).any())
                checked += 1
        self.assertGreater(checked, 100)
        self.assertEqual(dict(model.class_votes[20]), {0: 120, 2: 14, 1: 4})

    def test_vote_uses_entire_past_survives_gap_and_resets(self):
        model = RiskModel(scene())
        person = mover(1, 0, (500, 900), (20, 0), 150, 50)
        for t in np.arange(0, 5, .1):
            model.update([replace(person(t), cls=0 if t < 3 else 2)], float(t))
        self.assertEqual(model._class_vote(1), 0)
        model.update([], 7)
        self.assertNotIn(1, model.tracks)
        model.update([replace(person(7.1), cls=2)], 7.1)
        self.assertEqual(model._class_vote(1), 0)
        model.reset({})
        model.update([replace(person(0), cls=2)], 0)
        self.assertEqual(model._class_vote(1), 2)

    def test_rider_is_not_a_second_threat_to_third_vehicle(self):
        bike = mover(1, 2, (500, 900), (200, 0), 90, 120)
        rider = mover(2, 0, (500, 880), (200, 0), 150, 50)
        car = mover(3, 3, (1500, 1900), (0, -200), 150, 250)
        _, solo, _ = run([bike, car], 6)
        _, paired, evidence = run([bike, rider, car], 6)
        np.testing.assert_array_equal(solo, paired)
        self.assertGreater(solo.max(), 0)
        self.assertTrue(all(2 not in e.get('pair', []) for e in evidence))

    def test_new_bike_explains_rider_and_dismount_is_immediate(self):
        model = RiskModel(scene())
        person = mover(1, 0, (500, 880), (0, 0), 150, 50)
        bike = mover(2, 2, (500, 900), (0, 0), 90, 120)
        for t in (0, .5, 1):
            model.update([person(t)], t)
        self.assertFalse(model._state_arrays(1)['rider'][0])
        model.update([person(1.1), bike(1.1)], 1.1)
        self.assertTrue(model._state_arrays(1.1)['rider'][0])
        away = replace(person(1.2), xyxy=(1000, 700, 1050, 880))
        model.update([away, bike(1.2)], 1.2)
        self.assertFalse(model._state_arrays(1.2)['rider'][0])

    def test_dark_samples_preserve_red_phase_and_threat(self):
        runner = mover(1, 3, (1700, 1400 - 200 * 8), (0, 200), 150, 250)
        cross = mover(2, 3, (1700 - 300 * 8, 1400), (300, 0), 150, 250)
        _, red, evidence = run([runner, cross], 8, signal=lambda t: 'red', with_signal=True)
        _, dark, dark_evidence = run([runner, cross], 8,
            signal=lambda t: 'red' if t == round(t) else 'unknown', with_signal=True)
        np.testing.assert_array_equal(red, dark)
        self.assertEqual(evidence, dark_evidence)
        self.assertTrue(any('red_threat' in e.get('flags', []) for e in evidence))

    def test_hold_expires_in_seconds_unknown_does_not_refresh_it(self):
        model = RiskModel(scene(True))
        model.update([], 0, {'sig': 'unknown'})
        self.assertEqual(model.signal_states['sig'], 'unknown')
        model.update([], .1, {'sig': 'red', 'other': 'green'})
        for t in (.2, .6, 1.1):
            model.update([], t, {'sig': 'unknown'})
            self.assertEqual(model.signal_states['sig'], 'red')
        model.update([], 1.1001, None)
        self.assertEqual(model.signal_states['sig'], 'unknown')
        model.update([], 2, {'sig': 'red'})
        model.update([], 2.1, {'sig': 'green'})
        self.assertEqual(model.signal_states['sig'], 'green')
        model.reset({})
        model.update([], 2.2, {'sig': 'unknown'})
        self.assertEqual(model.signal_states['sig'], 'unknown')

    def test_signal_prefix_causality_and_independent_models(self):
        a, b = RiskModel(scene(True)), RiskModel(scene(True))
        for t, state in ((0, 'unknown'), (.1, 'red'), (.6, 'unknown')):
            a.update([], t, {'sig': state})
            b.update([], t, {'sig': state})
            self.assertEqual(a.signal_states, b.signal_states)
        a.update([], .7, {'sig': 'green'})
        self.assertEqual(b.signal_states['sig'], 'red')


if __name__ == '__main__':
    unittest.main()
