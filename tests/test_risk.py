import unittest

import numpy as np

from src.anticipation.risk import Calibrator, RiskConfig, RiskModel
from risk_fixtures import braking_mover, mover, run, scene

H_CAR, W_CAR = 150.0, 250.0


def crossing_course(v=150.0, meet=(2000.0, 400.0), t_meet=10.0, id_a=1, id_b=2):
    """Car A moves right, car B moves up; their foot points meet at `meet` at t_meet. Both are fully
    in frame for at least 8 s before contact at every tested speed."""
    a = mover(id_a, 3, (meet[0] - v * t_meet, meet[1]), (v, 0), H_CAR, W_CAR)
    b = mover(id_b, 3, (meet[0], meet[1] + v * t_meet), (0, -v), H_CAR, W_CAR)
    return a, b


def contact_time(v=150.0, t_meet=10.0):
    # Boxes overlap once A's box spans B's column (t_meet - W/v) and B's foot is within one box
    # height of A's (t_meet - H/v); the later of the two is first contact.
    return t_meet - min(W_CAR, H_CAR) / v


class RiskModelTests(unittest.TestCase):
    def test_output_always_in_unit_interval(self):
        rng = np.random.default_rng(0)
        movers = [mover(k, int(rng.choice([0, 3, 4, 5])), rng.uniform(300, 3500, 2), rng.uniform(-300, 300, 2),
                        float(rng.uniform(60, 300)), float(rng.uniform(40, 400))) for k in range(25)]
        _, risks, _ = run(movers, 8.0)
        self.assertTrue(np.all((risks >= 0) & (risks <= 1)))

    def test_causality_same_prefix_same_risk(self):
        a, b = crossing_course()
        swerve = mover(2, 3, (2000.0, 400.0 + 150.0 * 10), (0, -150.0), H_CAR, W_CAR, t1=6.0)
        away = mover(22, 3, (2000.0, 400.0 + 150 * 4.0), (400.0, 0.0), H_CAR, W_CAR, t0=6.1)
        _, r1, _ = run([a, b], 9.0)
        _, r2, _ = run([a, swerve, away], 9.0)
        n = int(round(6.0 / 0.1)) + 1                       # samples t = 0.0 .. 6.0 inclusive
        np.testing.assert_array_equal(r1[:n], r2[:n])
        self.assertFalse(np.array_equal(r1[n:], r2[n:]))

    def test_collision_course_crosses_half_3_to_5_s_before_contact(self):
        for v in (100.0, 150.0, 220.0):
            a, b = crossing_course(v)
            t, risks, _ = run([a, b], 10.0)
            first = t[np.argmax(risks >= 0.5)]
            self.assertTrue((risks >= 0.5).any(), v)
            lead = contact_time(v) - first
            self.assertGreaterEqual(lead, 2.5, v)
            self.assertLessEqual(lead, 5.0, v)

    def test_collision_course_still_alerts_when_the_guard_widens_the_stride(self):
        # Stride 15 at 29.97 fps: one update every 0.5 s.
        from risk_fixtures import scene
        a, b = crossing_course(150.0)
        model = RiskModel(scene())
        model.reset({})
        times = np.round(np.arange(0.0, 10.0, 0.5), 3)
        risks = [model.update([s for s in (a(t), b(t)) if s is not None], float(t)) for t in times]
        first = times[int(np.argmax(np.array(risks) >= 0.5))]
        self.assertGreater(max(risks), 0.5)
        self.assertGreaterEqual(contact_time(150.0) - first, 2.0)

    def test_normal_signalised_crossing_traffic_stays_low(self):
        # Near approach (down) is red: cars brake to a stop before the line (y=1000); cross traffic
        # flows left-right through the junction below it on green.
        sig = lambda t: "red"
        stoppers = [braking_mover(10 + k, 3, (1300 + 300 * k, 300 - 400 * k), (0, 1), 200.0, 150.0,
                                  1.5 + 0.8 * k, H_CAR, W_CAR) for k in range(3)]
        cross = [mover(20 + k, 3, (100 - 900 * k, 1400 + 120 * (k % 2)), (300.0, 0), H_CAR, W_CAR)
                 for k in range(4)]
        _, risks, _ = run(stoppers + cross, 12.0, signal=sig, with_signal=True)
        self.assertLess(risks.max(), 0.2)

    def test_red_light_runner_raises_risk_against_cross_traffic(self):
        sig = lambda t: "red"
        runner = mover(1, 3, (1700, 1400 - 200 * 8), (0, 200.0), H_CAR, W_CAR)       # never brakes
        cross = mover(2, 3, (1700 - 300 * 8, 1400), (300.0, 0), H_CAR, W_CAR)
        _, r_red, _ = run([runner, cross], 8.0, signal=sig, with_signal=True)
        self.assertGreater(r_red.max(), 0.5)

    def test_expected_stop_suppresses_the_same_geometry(self):
        # Same paths, but the approaching car brakes comfortably for the red: it is expected to stop.
        sig = lambda t: "red"
        stopper = braking_mover(1, 3, (1700, 1400 - 200 * 8), (0, 1), 200.0, 120.0, 2.0, H_CAR, W_CAR)
        cross = mover(2, 3, (1700 - 300 * 8, 1400), (300.0, 0), H_CAR, W_CAR)
        _, risks, _ = run([stopper, cross], 8.0, signal=sig, with_signal=True)
        self.assertLess(risks.max(), 0.3)

    def test_same_direction_followers_are_not_conflicts(self):
        lead = mover(1, 3, (600, 900), (250.0, 0), H_CAR, W_CAR)
        follow = mover(2, 3, (250, 905), (250.0, 0), H_CAR, W_CAR)
        overtaker = mover(3, 3, (0, 880), (330.0, 0), H_CAR, W_CAR)
        _, risks, _ = run([lead, follow, overtaker], 8.0)
        self.assertLess(risks.max(), 0.1)

    def test_pedestrian_on_sidewalk_is_ignored_but_on_road_is_not(self):
        # Car drives down x=1500 at 250 px/s; its foot reaches y=1900 at t=8.
        car = mover(1, 3, (1500, 1900 - 250 * 8), (0, 250.0), H_CAR, W_CAR)
        # A pedestrian walking across the road at ~0.8 body heights/s reaches the car's path at t=8.
        crossing = mover(2, 0, (1500 - 100 * 8, 1900), (100.0, 0), 120.0, 40.0)
        _, r_road, _ = run([car, crossing], 8.0)
        # The same pedestrian standing on the sidewalk (x < 150) beside the car's path is no conflict.
        car_near_walk = mover(1, 3, (300, 1900 - 250 * 8), (0, 250.0), H_CAR, W_CAR)
        on_walk = mover(2, 0, (90, 1900), (0, 0), 120.0, 40.0)
        _, r_walk, _ = run([car_near_walk, on_walk], 8.0)
        self.assertGreater(r_road.max(), 0.5)
        self.assertLess(r_walk.max(), 0.1)

    def test_rider_and_bike_do_not_conflict_with_each_other(self):
        bike = mover(1, 2, (500, 900), (200.0, 0), 90.0, 120.0)
        rider = mover(2, 0, (500, 880), (200.0, 10.0), 150.0, 50.0)
        _, risks, _ = run([bike, rider], 6.0)
        self.assertEqual(risks.max(), risks.min())

    def test_time_must_not_go_backwards(self):
        model = RiskModel(scene())
        model.update([], 1.0)
        with self.assertRaises(ValueError):
            model.update([], 0.5)

    def test_reset_clears_history(self):
        a, b = crossing_course()
        model = RiskModel(scene())
        _, r1, _ = run([a, b], 9.0, model=model)
        _, r2, _ = run([a, b], 9.0, model=model)
        np.testing.assert_array_equal(r1, r2)


class CalibratorTests(unittest.TestCase):
    def test_logistic_default_is_monotonic_and_centered(self):
        c = Calibrator(0.55, 0.08)
        xs = np.linspace(0, 1.5, 50)
        ys = [c(x) for x in xs]
        self.assertTrue(np.all(np.diff(ys) >= 0))
        self.assertAlmostEqual(c(0.55), 0.5, delta=0.002)
        self.assertEqual(c(0.0), 0.0)

    def test_isotonic_fit_is_monotonic(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(0, 1, 500)
        labels = (rng.uniform(0, 1, 500) < raw ** 2).astype(float)
        c = Calibrator(0.5, 0.1)
        c.fit_isotonic(raw, labels)
        ys = [c(x) for x in np.linspace(0, 1, 100)]
        self.assertTrue(np.all(np.diff(ys) >= -1e-12))
        self.assertTrue(all(0 <= y <= 1 for y in ys))


if __name__ == "__main__":
    unittest.main()
