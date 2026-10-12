"""Acceptance harness tests: report integrity and pass/fail semantics, no Webots."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from tools.verify_webots import case_matrix, parse_seeds, read_fresh_report, report_failures


RUN_ID = 'abc123def456'


def good_report(scenario='demo', seed=42):
    return dict(run_id=RUN_ID, success=True, reason='goal_reached', scenario=scenario,
                seed=seed, distance_to_goal=0.61, final_speed=0.005, replans=2, telemetry_obstacles=12,
                telemetry_packets_received=350, telemetry_path_points=20,
                obstacle_contact_frames=0, contact_tracking_available=True, tracked_obstacles=1,
                sensors=dict(valid_frames=1020, invalid_frames=0))


class AcceptanceReportTests(unittest.TestCase):
    def test_valid_demo_meets_independent_acceptance_requirements(self):
        self.assertEqual(report_failures(good_report(), 'demo', 42, 0), [])

    def test_flat_does_not_require_obstacles_or_replanning(self):
        report = good_report('flat')
        report.update(replans=0, telemetry_obstacles=0, tracked_obstacles=0)
        self.assertEqual(report_failures(report, 'flat', 42, 0), [])

    def test_each_required_measurement_can_fail_a_successful_exit(self):
        changes = (
            {'success': False}, {'success': 'true'}, {'reason': 'simulation_timeout'},
            {'scenario': 'flat'}, {'seed': 7}, {'distance_to_goal': 0.8},
            {'distance_to_goal': float('nan')}, {'distance_to_goal': -1},
            {'final_speed': None}, {'final_speed': float('nan')}, {'final_speed': float('inf')},
            {'final_speed': -0.01}, {'final_speed': 0.02}, {'final_speed': True},
            {'contact_tracking_available': False}, {'contact_tracking_available': 'true'},
            {'tracked_obstacles': 0}, {'tracked_obstacles': True},
            {'sensors': {'valid_frames': 0, 'invalid_frames': 0}},
            {'sensors': {'valid_frames': 5, 'invalid_frames': 1}},
            {'sensors': {'valid_frames': True, 'invalid_frames': False}},
            {'telemetry_packets_received': 0}, {'telemetry_path_points': 0},
            {'replans': 0}, {'telemetry_obstacles': 0}, {'obstacle_contact_frames': 1},
        )
        for change in changes:
            with self.subTest(change=change):
                report = good_report()
                report.update(change)
                self.assertTrue(report_failures(report, 'demo', 42, 0))
        self.assertTrue(report_failures(good_report(), 'demo', 42, 1))

    def test_missing_contact_measurement_fails_demo_and_reported_collision_fails_flat(self):
        report = good_report()
        del report['obstacle_contact_frames']
        self.assertTrue(report_failures(report, 'demo', 42, 0))
        report['scenario'] = 'flat'
        self.assertEqual(report_failures(report, 'flat', 42, 0), [])
        report['sensors']['obstacle_contact_frames'] = 1
        self.assertTrue(report_failures(report, 'flat', 42, 0))

    def test_legacy_report_without_measured_stop_cannot_pass(self):
        report = good_report()
        del report['final_speed']
        self.assertTrue(report_failures(report, 'demo', 42, 0))

    def test_default_quick_and_extended_cases(self):
        self.assertEqual(case_matrix([42, 7], False), [('flat', 42), ('demo', 42), ('demo', 7)])
        self.assertEqual(case_matrix([42, 7], True), [('flat', 42), ('demo', 42)])
        self.assertEqual(parse_seeds('42, 7,123,42'), [42, 7, 123])


class FreshReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'logs').mkdir()
        self.path = self.root / 'logs' / f'validation-{RUN_ID}.json'
        self.started = time.time()
        self.path.write_text(json.dumps(good_report()), encoding='utf-8')
        self.stdout = f'{{"success": true}}\n[LORA] Validation: {self.path}\n'

    def test_read_exact_printed_path_and_track_run_identity(self):
        # A newer unrelated report must never be selected by a glob/latest rule.
        unrelated = self.root / 'logs' / 'validation-ffffffffffff.json'
        unrelated.write_text('{"success": false}', encoding='utf-8')
        seen = set()
        path, result = read_fresh_report(self.stdout, self.root, set(), self.started, seen)
        self.assertEqual(path, self.path.resolve())
        self.assertEqual(result['run_id'], RUN_ID)
        self.assertEqual(seen, {RUN_ID})

    def test_reject_report_that_existed_before_the_launch(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            read_fresh_report(self.stdout, self.root, {self.path.resolve()}, self.started, set())

    def test_reject_reused_run_id_even_at_new_path(self):
        with self.assertRaisesRegex(ValueError, 'reused'):
            read_fresh_report(self.stdout, self.root, set(), self.started, {RUN_ID})

    def test_reject_old_timestamp(self):
        os.utime(self.path, (self.started - 10, self.started - 10))
        with self.assertRaisesRegex(ValueError, 'predates'):
            read_fresh_report(self.stdout, self.root, set(), self.started, set())

    def test_reject_missing_ambiguous_or_outside_report_path(self):
        for output in ('', self.stdout + self.stdout, '[LORA] Validation: ' + str(self.root / self.path.name)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                read_fresh_report(output, self.root, set(), self.started, set())

    def test_reject_report_with_mismatched_identity_or_wrong_json_type(self):
        for report in (dict(good_report(), run_id='old-run'), ['not a report']):
            with self.subTest(report=report):
                self.path.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'run_id'):
                    read_fresh_report(self.stdout, self.root, set(), self.started, set())


if __name__ == '__main__':
    unittest.main()
