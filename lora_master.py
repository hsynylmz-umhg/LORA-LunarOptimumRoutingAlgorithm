"""LORA launcher: reproducible terrain -> validated route -> Webots -> telemetry.

All world coordinates are ENU: X east, Y north, Z up. Terrain samples and
planner nodes share exactly the same positions, spacing and slope limit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import textwrap
import uuid

import numpy as np
from scipy.ndimage import gaussian_filter

from src.algorithm.pathfinder import Pathfinder
from src.telemetry import UDPServer
from src.session import ProjectLock
from src.webots.world_builder import build_world

ROOT = Path(__file__).resolve().parent
WORLD = ROOT / 'webots/worlds/lunar_surface.wbt'
PROCESSED = ROOT / 'data/processed'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=('demo', 'flat', 'terrain'), default='demo')
    p.add_argument('--size', type=int, default=64, help='Terrain samples per side (32..500)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--img', type=Path, help='Real elevation IMG with adjacent PDS3 LBL, never a SLOPE product')
    p.add_argument('--webots', type=Path, help='Webots executable (otherwise auto-detect)')
    p.add_argument('--preview', type=Path, help='Save a rendered Webots image to this path')
    p.add_argument('--prepare-only', action='store_true', help='Generate and validate files, then exit')
    p.add_argument('--no-webots', action='store_true', help='Open panel without starting Webots')
    p.add_argument('--headless', action='store_true', help='Run batch verification and exit with mission result')
    p.add_argument('--timeout', type=float, default=180, help='Batch wall-clock limit in seconds')
    p.add_argument('--sim-timeout', type=float, default=None, help='Mission limit in simulated seconds (default scales with route length)')
    p.add_argument('--port', type=int, default=0, help='Telemetry UDP port; 0 selects an available port')
    return p


def make_terrain(args):
    if not 32 <= args.size <= 500:
        raise ValueError('--size must be between 32 and 500')
    if args.img:
        from src.data.pds_reader import read_elevation_patch
        img = args.img.expanduser()
        if not img.is_file():
            img = ROOT / 'data/raw' / args.img
        heights, source = read_elevation_patch(img, args.size, np.random.default_rng(args.seed))
        spacing = float(source['sample_spacing_m'])
        source['removed_vertical_offset_m'] = float(np.min(heights))
        heights -= source['removed_vertical_offset_m']
        return heights, spacing, source
    n = args.size
    rng = np.random.default_rng(args.seed)
    heights = np.zeros((n, n), dtype=float)
    if args.scenario != 'flat':
        yy, xx = np.mgrid[:n, :n]
        heights = .35 * np.sin(xx / 12) * np.cos(yy / 14)
        heights += gaussian_filter(rng.normal(0, .07, (n, n)), 3)
        heights -= heights.min()
    return heights, 1., {'source': 'synthetic', 'scenario': args.scenario, 'seed': args.seed,
                         'note': 'Demonstration terrain; not NASA/JAXA elevation data.'}


def select_mission(heights, spacing):
    rows, cols = heights.shape
    offset_x, offset_y = -(cols - 1) * spacing / 2, -(rows - 1) * spacing / 2
    planner = Pathfinder(heights, cell_size=spacing, max_slope=20.,
                         world_offset_x=offset_x, world_offset_z=offset_y)
    # Validate connectivity, not just the slope at the two endpoints.
    candidates = [(rows // 2, cols // 4, rows // 2, 3 * cols // 4),
                  (rows // 4, cols // 4, 3 * rows // 4, 3 * cols // 4),
                  (rows // 4, cols // 2, 3 * rows // 4, cols // 2)]
    for sr, sc, gr, gc in candidates:
        start = (offset_x + sc * spacing, offset_y + sr * spacing)
        goal = (offset_x + gc * spacing, offset_y + gr * spacing)
        if planner.plan(start, goal):
            return planner, start, goal, offset_x, offset_y
    raise ValueError('No traversable mission connects the candidate endpoints at the 20 degree slope limit. Use another elevation patch.')


def prepare(args, run_id, port):
    heights, spacing, source = make_terrain(args)
    planner, start, goal, ox, oy = select_mission(heights, spacing)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    mission_path = PROCESSED / 'mission.json'
    map_path = PROCESSED / 'terrain.npy'
    np.save(map_path, heights, allow_pickle=False)
    obstacles = []
    if args.scenario == 'demo' and not args.img:
        obstacles = [dict(x=(start[0] + goal[0]) / 2, y=start[1], radius=.8, height=1.0)]
    stats = planner.current_path_stats()
    sim_timeout = args.sim_timeout if args.sim_timeout is not None else max(240., stats.get('distance_m', 0) / .3 + 60.)
    mission = dict(schema_version=1, run_id=run_id, coordinate_system='ENU',
                   map_path=str(map_path), mission_file=str(mission_path),
                   cell_size=spacing, offset_x=ox, offset_y=oy,
                   start=list(start), goal=list(goal), max_slope=20.,
                   udp_host='127.0.0.1', udp_port=port, seed=args.seed,
                   timeout_sim_s=sim_timeout, exit_on_complete=args.headless, obstacle_count=len(obstacles),
                   source=source, scenario=args.scenario,
                   planned_statistics=stats)
    if args.preview:
        mission['preview_path'] = str(args.preview.resolve())
    mission_path.write_text(json.dumps(mission, indent=2, ensure_ascii=False), encoding='utf-8')
    build_world(heights, mission, WORLD, obstacles=obstacles)
    # Webots must use the SAME interpreter as the launcher, not a global broken Python.
    runtime = ROOT / 'webots/controllers/lora_controller/runtime.ini'
    runtime.write_text('[python]\nCOMMAND = "' + Path(sys.executable).as_posix() + '"\n', encoding='utf-8')
    print(f'[LORA] Source={source["source"]} shape={heights.shape} spacing={spacing:g}m', flush=True)
    print(f'[LORA] Mission {start} -> {goal}; initial path {planner.current_path_stats().get("distance_m", 0):.1f}m', flush=True)
    print(f'[LORA] World: {WORLD}', flush=True)
    return heights, mission, planner


def find_webots(explicit=None):
    candidates = [explicit, os.environ.get('WEBOTS_EXE')]
    if os.environ.get('WEBOTS_HOME'):
        home = Path(os.environ['WEBOTS_HOME'])
        candidates += [home / 'msys64/mingw64/bin/webots.exe', home / 'webots']
    candidates += [shutil.which('webots'), Path('C:/Program Files/Webots/msys64/mingw64/bin/webots.exe'),
                   Path('/usr/local/webots/webots'), Path('/Applications/Webots.app/Contents/MacOS/webots')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError('Webots executable not found. Set --webots or WEBOTS_HOME. Use --prepare-only to inspect the generated world.')


def launch_webots(args, log_path):
    executable = find_webots(args.webots)
    cmd = [str(executable), '--batch', '--stdout', '--stderr',
           '--mode=fast' if args.headless else '--mode=realtime']
    if args.headless:
        cmd += ['--minimize']
        if not args.preview:
            cmd += ['--no-rendering']
    cmd.append(str(WORLD))
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0 if args.headless else 1
        kwargs['startupinfo'] = startup
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    print(f'[LORA] Webots PID={proc.pid}; log={log_path}', flush=True)
    return proc


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == 'nt':
        # Only the process tree created by THIS launcher, never other Webots sessions.
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_result(run_id):
    result_path = PROCESSED / 'mission_result.json'
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        return result if isinstance(result, dict) and result.get('run_id') == run_id else None
    except (OSError, ValueError):
        return None


def run_batch(args, mission, proc, server):
    deadline = time.monotonic() + args.timeout
    result = None
    while time.monotonic() < deadline:
        result = read_result(mission['run_id'])
        if result is not None or proc.poll() is not None:
            break
        time.sleep(.1)
    result = result or read_result(mission['run_id'])
    if result is None:
        result = dict(run_id=mission['run_id'], success=False,
                      reason='wall_timeout' if proc.poll() is None else 'controller_did_not_complete',
                      webots_exit_code=proc.poll())
    else:
        # Give Supervisor.simulationQuit time to flush Webots output normally.
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            result.update(success=False, reason='webots_did_not_exit')
        if proc.poll() not in (None, 0) and result.get('success') is True:
            result.update(success=False, reason='webots_exit_error')
    result['webots_exit_code'] = proc.poll()
    snapshot = server.snapshot()
    result['telemetry_packets_received'] = snapshot['packet_count']
    result['telemetry_path_points'] = len(snapshot['path'])
    result['telemetry_obstacles'] = len(snapshot['obstacles'])
    result['python_version'] = sys.version.split()[0]
    result['scenario'] = mission['scenario']
    result['seed'] = mission['seed']
    report = ROOT / 'logs' / f'validation-{mission["run_id"]}.json'
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f'[LORA] Validation: {report}', flush=True)
    return 0 if result.get('success') is True else 1


def show_panel(heights, mission, planner, server, proc=None, save_path=None):
    # Lazy imports keep batch tests independent of GUI backends and display servers.
    if save_path:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.style.use('dark_background')
    fig, (ax, info) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [3, 1]})
    fig.canvas.manager.set_window_title('LORA | Lunar Rover Mission Control')
    spacing = mission['cell_size']
    ox, oy = mission['offset_x'], mission['offset_y']
    rows, cols = heights.shape
    ax.imshow(heights, origin='lower', cmap='gray', extent=[ox-spacing/2, ox+(cols-.5)*spacing,
              oy-spacing/2, oy+(rows-.5)*spacing], interpolation='bilinear')
    points = planner.waypoints_m
    route, = ax.plot(*zip(*points), color='#79c5ff', lw=1.5, label='Planlanan rota')
    trail, = ax.plot([], [], color='#40e0d0', lw=2, label='Gerçek iz')
    rover, = ax.plot([], [], 'o', color='#73ffb4', ms=8)
    obstacles = ax.scatter([], [], color='#ffb454', marker='x', label='Lidar engeli')
    ax.plot(*mission['start'], '^', color='#73ffb4', ms=10, label='Başlangıç')
    ax.plot(*mission['goal'], '*', color='#ff807f', ms=14, label='Hedef')
    ax.set(xlabel='Doğu / X (m)', ylabel='Kuzey / Y (m)', title='LORA · Ay yüzeyi navigasyonu')
    ax.legend(loc='upper left', fontsize=8)
    info.axis('off')
    label = info.text(.03, .96, '', va='top', transform=info.transAxes, fontsize=10, linespacing=1.7)
    source_label = 'Sentetik demo' if mission['source']['source'] == 'synthetic' else 'PDS yükselti haritası'

    def update(_):
        state = server.snapshot()
        path = state['path']
        if state['last_received'] is not None:
            route.set_data(*zip(*path)) if path else route.set_data([], [])
        if state['trail']:
            trail.set_data(*zip(*state['trail']))
            rover.set_data([state['x']], [state['y']])
        if state['obstacles']:
            obstacles.set_offsets(state['obstacles'])
        status = state['state']
        if proc and proc.poll() is not None and not state['done']:
            status = 'Kontrolcü kapandı; logs dosyasına bakın'
        age = time.monotonic() - state['last_received'] if state['last_received'] else None
        if age is not None and age > 3 and not state['done']:
            status = 'Telemetri kesildi / simülasyon duraklatıldı'
        status = {'completed': 'Tamamlandı', 'complete': 'Tamamlandı', 'navigating': 'Hedefe ilerliyor',
                  'starting': 'Hazırlanıyor', 'waiting': 'Bağlantı bekleniyor',
                  'failed': 'Görev durduruldu', 'no_route': 'Uygun rota yok'}.get(status, status)
        message = state.get('message', '')
        message = {'goal_reached': 'Hedefe ulaşıldı', 'rover_stalled': 'Rover hareket edemiyor',
                   'simulation_timeout': 'Görev süresi doldu', 'no_route': 'Geçilebilir rota bulunamadı'}.get(message, message)
        distance = math.hypot(state.get('x', mission['start'][0])-mission['goal'][0],
                              state.get('y', mission['start'][1])-mission['goal'][1])
        label.set_text(f'LORA  /  MVP\n\n{source_label}\nSeed: {mission["seed"]}\n\n'
                       f'Durum: {textwrap.fill(status, 26)}\n{textwrap.fill(message, 26)}\n\n'
                       f'Hız: {state.get("speed", 0):.2f} m/s\nHedefe: {distance:.2f} m\n'
                       f'Süre: {state.get("time", 0):.1f} s\nEngel: {len(state["obstacles"])}\n\n'
                       'X: Doğu · Y: Kuzey\nZ: Yükseklik\nAy yerçekimi: 1.62 m/s²')
    update(0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return
    animation = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
    fig._lora_animation = animation
    plt.show()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.headless and args.no_webots:
        raise SystemExit('--headless and --no-webots cannot be combined; use --prepare-only')
    if not all(math.isfinite(t) and t > 0 for t in (args.timeout, args.sim_timeout) if t is not None):
        raise SystemExit('Timeouts must be finite positive seconds')
    run_id = uuid.uuid4().hex[:12]
    server = proc = lock = None
    try:
        lock = ProjectLock(ROOT / '.lora.lock')
        server = UDPServer(run_id, port=args.port)
        heights, mission, planner = prepare(args, run_id, server.port)
        if args.prepare_only:
            return 0
        server.start()
        if not args.no_webots:
            proc = launch_webots(args, ROOT / 'logs' / f'webots-{run_id}.log')
        if args.headless:
            code = run_batch(args, mission, proc, server)
            show_panel(heights, mission, planner, server, proc,
                       save_path=ROOT / 'logs' / f'panel-{run_id}.png')
            return code
        show_panel(heights, mission, planner, server, proc)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'[LORA ERROR] {exc}', file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if server:
            server.close()
        stop_process(proc)
        if lock:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
