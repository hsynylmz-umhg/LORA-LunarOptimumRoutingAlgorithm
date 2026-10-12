"""Run independent, sequential Webots acceptance checks against fresh reports.

Examples:
    python tools/verify_webots.py --quick
    python tools/verify_webots.py --seeds 42,7,123

This launches real Webots processes. Do not run another LORA mission in the same
checkout concurrently, because mission files and the generated world are shared.
Unit tests in tests/test_acceptance_report.py never launch Webots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPORT_LINE = re.compile(r"^\[LORA\] Validation:\s*(.+?)\s*$", re.MULTILINE)
REPORT_NAME = re.compile(r"validation-([0-9a-f]{12})\.json\Z")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = list(dict.fromkeys(int(part.strip()) for part in value.split(',')))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Seeds must be comma-separated integers, e.g. 42,7,123') from error
    if not seeds or any(not 0 <= seed <= 2147483647 for seed in seeds):
        raise argparse.ArgumentTypeError('Seeds must be between 0 and 2147483647')
    return seeds


def case_matrix(seeds: list[int], quick: bool) -> list[tuple[str, int]]:
    return [('flat', seeds[0])] + [('demo', seed) for seed in (seeds[:1] if quick else seeds)]


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def report_failures(report: dict, scenario: str, seed: int, returncode: int) -> list[str]:
    """Acceptance requirements are checked independently of launcher exit logic."""
    failures = []
    if returncode != 0:
        failures.append(f'Launcher exited with status {returncode}')
    if report.get('scenario') != scenario or report.get('seed') != seed:
        failures.append('Report scenario/seed does not match the launched case')
    if report.get('success') is not True or report.get('reason') != 'goal_reached':
        failures.append(f'Mission did not reach its goal: {report.get("reason", "missing reason")}')
    distance = report.get('distance_to_goal')
    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
            or not math.isfinite(distance) or not 0 <= distance <= 0.650001):
        failures.append('Final goal distance must be finite and at most 0.65 m')
    speed = report.get('final_speed')
    if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
            or not math.isfinite(speed) or not 0 <= speed < 0.02):
        failures.append('Final measured speed must be finite, nonnegative and below 0.02 m/s')
    sensors = report.get('sensors')
    if not isinstance(sensors, dict):
        sensors = {}
    frames = sensors.get('valid_frames')
    if not _count(frames) or frames < 1:
        failures.append('No valid sensor frames were reported')
    invalid = sensors.get('invalid_frames')
    if not _count(invalid) or invalid != 0:
        failures.append('Invalid sensor frames must be exactly zero')
    packets = report.get('telemetry_packets_received')
    if not _count(packets) or packets < 1:
        failures.append('No telemetry packets reached the launcher')
    path_points = report.get('telemetry_path_points')
    if not _count(path_points) or path_points < 2:
        failures.append('The telemetry route is missing or incomplete')
    if scenario == 'demo':
        replans = report.get('replans')
        if not _count(replans) or replans < 1:
            failures.append('Obstacle demo must replan at least once')
        obstacles = report.get('telemetry_obstacles')
        if not _count(obstacles) or obstacles < 1:
            failures.append('Obstacle demo must report at least one sensed obstacle')
        if report.get('contact_tracking_available') is not True:
            failures.append('Obstacle demo requires active physical contact tracking')
        tracked = report.get('tracked_obstacles')
        if not _count(tracked) or tracked < 1:
            failures.append('Obstacle demo must track at least one physical obstacle')
        if 'obstacle_contact_frames' not in report:
            failures.append('Obstacle demo must report physical obstacle contact frames')
    # A flat world may not have obstacles. A reported collision always fails.
    for container in (report, sensors):
        if 'obstacle_contact_frames' in container:
            contacts = container['obstacle_contact_frames']
            if not _count(contacts) or contacts != 0:
                failures.append('A physical obstacle contact was measured')
    return failures


def read_fresh_report(stdout: str, root: Path, before: set[Path], started_at: float,
                      seen_run_ids: set[str]) -> tuple[Path, dict]:
    """Read only the report explicitly emitted by this subprocess, never latest."""
    matches = REPORT_LINE.findall(stdout)
    if len(matches) != 1:
        raise ValueError('Launcher must emit exactly one [LORA] Validation report path')
    path = Path(matches[0])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.parent != (root / 'logs').resolve():
        raise ValueError('Validation report must be inside this checkout logs directory')
    match = REPORT_NAME.fullmatch(path.name)
    if not match:
        raise ValueError('Validation report filename has no valid run_id')
    run_id = match.group(1)
    if path in before or run_id in seen_run_ids:
        raise ValueError('Refusing a stale or reused validation report')
    if path.stat().st_mtime < started_at - 1:
        raise ValueError('Validation report predates this launch')
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(report, dict) or report.get('run_id') != run_id:
        raise ValueError('Report run_id does not match its emitted filename')
    seen_run_ids.add(run_id)
    return path, report


def _stop_tree(proc: subprocess.Popen) -> None:
    """Clean up only the launcher tree created by this acceptance runner."""
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def run_case(root: Path, scenario: str, seed: int, timeout: float,
             webots: Path | None, suite_id: str, seen_run_ids: set[str]) -> dict:
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    # A path that existed before this launch is never accepted even if it is
    # accidentally overwritten with recent timestamps by another process.
    before = {path.resolve() for path in logs.glob('validation-*.json')}
    command = [sys.executable, str(root / 'lora_master.py'), '--headless',
               '--scenario', scenario, '--seed', str(seed), '--timeout', str(timeout)]
    if webots is not None:
        command += ['--webots', str(webots)]
    print(f'[ACCEPTANCE] Starting {scenario}, seed={seed}', flush=True)
    started_at = time.time()
    started_clock = time.monotonic()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    options = dict(cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, encoding='utf-8', errors='replace')
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options['start_new_session'] = True
    record = dict(scenario=scenario, seed=seed, command=command, passed=False, errors=[])
    try:
        proc = subprocess.Popen(command, **options)
    except OSError as error:
        record['errors'].append(f'Could not launch mission: {error}')
        return record
    try:
        try:
            # The launcher owns its 120s Webots timeout; allow import, shutdown
            # and report-writing overhead before enforcing this outer bound.
            stdout, _ = proc.communicate(timeout=timeout + 45)
        except subprocess.TimeoutExpired:
            _stop_tree(proc)
            stdout, _ = proc.communicate(timeout=5)
            record['errors'].append('Launcher exceeded the outer wall-clock limit')
    except BaseException:
        _stop_tree(proc)
        raise
    record['returncode'] = proc.returncode
    record['wall_seconds'] = round(time.monotonic() - started_clock, 3)
    launcher_log = logs / f'acceptance-{suite_id}-{scenario}-{seed}.log'
    launcher_log.write_text(stdout, encoding='utf-8')
    record['launcher_log'] = str(launcher_log)
    try:
        report_path, report = read_fresh_report(stdout, root, before, started_at, seen_run_ids)
        record['report_path'] = str(report_path)
        record['measurements'] = report
        record['errors'].extend(report_failures(report, scenario, seed, proc.returncode))
        record['physical_contact_measurement_available'] = (
            report.get('contact_tracking_available') is True
            and 'obstacle_contact_frames' in report)
    except (OSError, ValueError) as error:
        record['errors'].append(str(error))
    record['passed'] = not record['errors']
    print(f'[ACCEPTANCE] {"PASS" if record["passed"] else "FAIL"}: {scenario}, seed={seed}', flush=True)
    for error in record['errors']:
        print(f'  - {error}', flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Run flat and one demo (first selected seed)')
    parser.add_argument('--seeds', type=parse_seeds, default=parse_seeds('42,7'),
                        help='Demo seed list, default 42,7; first seed is also used for flat')
    parser.add_argument('--timeout', type=float, default=120, help='Wall-clock limit per Webots mission')
    parser.add_argument('--webots', type=Path, help='Optional Webots executable path')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    suite_id = uuid.uuid4().hex[:12]
    cases = case_matrix(args.seeds, args.quick)
    summary = dict(schema_version=1, suite_id=suite_id,
                   started_at_utc=datetime.now(timezone.utc).isoformat(),
                   python=sys.executable, expected_cases=len(cases), passed=False, results=[])
    seen_run_ids = set()
    interrupted = False
    try:
        for scenario, seed in cases:
            summary['results'].append(run_case(ROOT, scenario, seed, args.timeout,
                                              args.webots, suite_id, seen_run_ids))
    except KeyboardInterrupt:
        interrupted = True
        summary['interrupted'] = True
    finally:
        summary['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        summary['passed'] = (not interrupted and len(summary['results']) == len(cases)
                             and all(result['passed'] for result in summary['results']))
        report_path = ROOT / 'logs' / 'acceptance-summary.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(report_path)
        print(f'[ACCEPTANCE] Summary: {report_path}', flush=True)
    return 130 if interrupted else (0 if summary['passed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
