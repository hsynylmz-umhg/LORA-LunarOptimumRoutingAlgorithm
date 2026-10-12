"""LORA ENU controller: finite sensors, occupancy replanning and safe stops.

Webots passes the absolute mission JSON path in controllerArgs. The controller
uses simulation time, emits versioned UDP, and always writes mission_result.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.algorithm.pathfinder import Pathfinder
from src.algorithm.slam_mapper import SlamMapper
from src.navigation import (LIDAR_UP_M, compass_yaw, front_clearance,
                            project_return, scan_points, telemetry_waypoints, wheel_speeds)
from src.telemetry import encode_packet


class LoraController:
    def __init__(self, robot, mission_path):
        self.robot = robot
        self.mission_path = Path(mission_path).resolve()
        self.config = json.loads(self.mission_path.read_text(encoding="utf-8-sig"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported mission schema_version")
        self.run_id = self.config["run_id"]
        safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(self.run_id))[:80]
        self.log_path = ROOT / "logs" / f"controller-{safe_run_id}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self.log_bytes = 0
        self.last_pose_log = -math.inf
        self.preview_exported = False
        self.goal = tuple(float(v) for v in self.config["goal"])
        if len(self.goal) != 2 or not all(math.isfinite(v) for v in self.goal):
            raise ValueError("Mission goal must contain two finite coordinates")
        self.timestep = int(round(robot.getBasicTimeStep()))
        self.started = robot.getTime()
        self.timeout = float(self.config.get("timeout_sim_s", 300))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_sim_s must be finite and positive")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (self.config.get("udp_host", "127.0.0.1"),
                        int(self.config.get("udp_port", 5005)))
        self.motors = []
        self.devices = {}
        self.last_position = None
        self.previous_sample = None
        self.last_telemetry = -math.inf
        self.last_replan = -math.inf
        self.no_route_since = None
        self.progress_position = None
        self.progress_time = self.started
        self.replans = 0
        self.planned = False
        self.map_changed = False
        self.finished = False
        self.invalid_sensor_frames = 0
        self.sensor_invalid_since = None
        self.valid_sensor_frames = 0
        self.minimum_clearance = math.inf
        self.result = None
        self.obstacle_contact_frames = 0
        self.chassis_contact_frames = 0
        self.obstacle_nodes = []
        self.chassis_node = None
        self.contact_tracking_available = False
        self._setup()

    def send(self, kind, **fields):
        payload = dict(version=1, run_id=self.run_id, type=kind, **fields)
        try:
            self.socket.sendto(encode_packet(payload), self.address)
        except OSError:
            pass  # UI is optional; the result file is the durable outcome.

    def log_event(self, event, **fields):
        """Persist bounded JSONL diagnostics even when Webots loses stdout."""
        record = dict(run_id=self.run_id, time=self.robot.getTime() - self.started,
                      event=event, **fields)
        line = (json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n").encode("utf-8")
        if len(line) > 16384:
            record = dict(run_id=self.run_id, time=self.robot.getTime() - self.started,
                          event=event, truncated=True, detail=line[:12000].decode("utf-8", errors="replace"))
            line = (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8")
        try:
            if self.log_bytes + len(line) > 512 * 1024:
                tail = self.log_path.read_bytes()[-256 * 1024:]
                tail = tail.partition(b"\n")[2]
                self.log_path.write_bytes(tail)
                self.log_bytes = len(tail)
            with self.log_path.open("ab") as stream:
                stream.write(line)
            self.log_bytes += len(line)
        except OSError:
            pass

    def export_preview(self):
        preview = self.config.get("preview_path")
        if not preview or not hasattr(self.robot, "exportImage"):
            return
        try:
            path = Path(preview).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.robot.exportImage(str(path), 90)
            self.preview_exported = True
            self.log_event("preview", path=str(path))
        except Exception as error:
            self.log_event("preview_error", message=str(error))

    def setup_contacts(self):
        count = int(self.config.get("obstacle_count", 0))
        if hasattr(self.robot, "getSelf"):
            node = self.robot.getSelf()
            if node is not None and hasattr(node, "enableContactPointsTracking"):
                node.enableContactPointsTracking(self.timestep, False)
                self.chassis_node = node
        if hasattr(self.robot, "getFromDef"):
            for index in range(count):
                node = self.robot.getFromDef(f"OBSTACLE_{index}")
                if node is not None and hasattr(node, "enableContactPointsTracking"):
                    node.enableContactPointsTracking(self.timestep, False)
                    self.obstacle_nodes.append(node)
        self.contact_tracking_available = (self.chassis_node is not None and
                                           len(self.obstacle_nodes) == count)
        self.log_event("contact_tracking", available=self.contact_tracking_available,
                       requested_obstacles=count, tracked_obstacles=len(self.obstacle_nodes))

    def detect_contacts(self):
        # Descendants are excluded so normal wheel-ground contact is not a chassis contact.
        if self.chassis_node is not None and self.chassis_node.getContactPoints(False):
            self.chassis_contact_frames += 1
        collision = any(node.getContactPoints(False) for node in self.obstacle_nodes)
        if collision:
            self.obstacle_contact_frames += 1
            self.log_event("obstacle_contact", frame=self.obstacle_contact_frames)
        return collision

    def status(self, state, message):
        self.log_event("status", state=state, message=message)
        print(f"[LORA] {state}: {message}", flush=True)
        self.send("status", state=state, message=message)

    def _setup(self):
        for name in ("front left wheel", "front right wheel", "back left wheel", "back right wheel"):
            motor = self.robot.getDevice(name)
            if motor is None:
                raise RuntimeError(f"Required motor is missing: {name}")
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
            self.motors.append(motor)
        for name in ("gps", "compass", "lidar", "inertial unit"):
            device = self.robot.getDevice(name)
            if device is None:
                raise RuntimeError(f"Required sensor is missing: {name}")
            device.enable(self.timestep)
            self.devices[name] = device
        self.gps = self.devices["gps"]
        self.compass = self.devices["compass"]
        self.lidar = self.devices["lidar"]
        self.imu = self.devices["inertial unit"]
        self.resolution = self.lidar.getHorizontalResolution()
        self.fov = self.lidar.getFov()
        self.max_range = self.lidar.getMaxRange()
        map_path = Path(self.config["map_path"])
        if not map_path.is_absolute():
            map_path = self.mission_path.parent / map_path
        self.heightmap = np.load(map_path, allow_pickle=False)
        self.cell_size = float(self.config["cell_size"])
        self.offset_x = float(self.config["offset_x"])
        self.offset_y = float(self.config["offset_y"])
        self.mapper = SlamMapper(self.heightmap, cell_size=self.cell_size, inflation_radius_m=0.7)
        self.planner = Pathfinder(self.heightmap, cell_size=self.cell_size,
                                  max_slope=float(self.config.get("max_slope", 20)),
                                  world_offset_x=self.offset_x, world_offset_z=self.offset_y,
                                  blocked=self.mapper.blocked)
        self.setup_contacts()
        self.status("starting", "Waiting for finite sensors and suspension settling")

    def set_motors(self, left, right):
        if not all(math.isfinite(v) for v in (left, right)):
            left = right = 0.0
        for motor, speed in zip(self.motors, (left, right, left, right)):
            limit = min(6.0, motor.getMaxVelocity())
            motor.setVelocity(max(-limit, min(limit, speed)))

    def stop(self):
        self.set_motors(0.0, 0.0)

    def plan(self, position):
        self.stop()
        self.last_replan = self.robot.getTime()
        self.replans += int(self.planned)
        self.planned = True
        self.map_changed = False
        ok = self.planner.plan(position, self.goal)
        if not ok:
            self.send("path", points=[])
            if self.no_route_since is None:
                self.no_route_since = self.last_replan
                self.status("no_route", "No traversable route; motors stopped")
            return False
        self.no_route_since = None
        self.send("path", points=telemetry_waypoints(self.planner.waypoints_m))
        self.status("navigating", f"Route ready; {len(self.planner.waypoints_m)} waypoints")
        return True

    def learn_obstacles(self, position, height, yaw, points):
        cells = set()
        observations = []
        rows, cols = self.heightmap.shape
        for distance, angle in points:
            if distance > 4.0 or abs(angle) > math.radians(60):
                continue
            x, y = project_return(position, yaw, distance, angle)
            col = int(round((x - self.offset_x) / self.cell_size))
            row = int(round((y - self.offset_y) / self.cell_size))
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            # A known terrain rise is already handled by the slope-aware planner.
            # Occupancy represents returns above the known elevation surface.
            if self.heightmap[row, col] >= height + LIDAR_UP_M - 0.20:
                continue
            cells.add((row, col))
            observations.append((x, y))
        if cells and self.mapper.mark_obstacles(cells):
            self.map_changed = True
            for x, y in observations[::max(1, len(observations) // 12)]:
                self.send("obs", x=x, y=y)

    def measure_motion(self):
        """Read an actual current GPS speed/pose for terminal evidence and telemetry."""
        values = self.gps.getValues()
        north = self.compass.getValues()
        speed = self.gps.getSpeed()
        if (len(values) != 3 or len(north) != 3 or
                not all(math.isfinite(v) for v in (*values, *north, speed)) or speed < 0):
            return None
        try:
            yaw = compass_yaw(north)
        except ValueError:
            return None
        self.last_position = (float(values[0]), float(values[1]))
        return dict(x=self.last_position[0], y=self.last_position[1],
                    speed=float(speed), heading=yaw, time=self.robot.getTime() - self.started)

    def settle_at_goal(self):
        """Verify the rover has physically stopped before reporting mission success."""
        stop_started = self.robot.getTime()
        measurement = self.measure_motion()
        self.status("stopping", "Goal radius reached; verifying physical stop")
        max_steps = max(1, int(3000 // self.timestep))
        for _ in range(max_steps):
            self.stop()
            if self.robot.step(self.timestep) == -1:
                return False, "simulation_ended_during_stop", measurement
            measurement = self.measure_motion()
            if self.detect_contacts():
                return False, "obstacle_collision", measurement
            attitude = self.imu.getRollPitchYaw()
            ranges = self.lidar.getRangeImage()
            valid = (measurement is not None and len(attitude) == 3 and
                     all(math.isfinite(v) for v in attitude) and bool(ranges) and
                     any(v == math.inf or (math.isfinite(v) and v > 0) for v in ranges))
            if not valid:
                self.invalid_sensor_frames += 1
                return False, "sensor_unavailable", measurement
            self.valid_sensor_frames += 1
            if max(abs(attitude[0]), abs(attitude[1])) > math.radians(35):
                return False, "unsafe_rover_tilt", measurement
            points = list(scan_points(ranges, self.resolution, self.fov, self.max_range))
            clearance, _, _ = front_clearance(points)
            self.minimum_clearance = min(self.minimum_clearance, clearance)
            if self.robot.getTime() - stop_started >= .2 and measurement["speed"] < .02:
                if math.dist(self.last_position, self.goal) > .65:
                    return False, "goal_drift_after_stop", measurement
                return True, "goal_reached", measurement
        return False, "rover_did_not_stop", measurement

    def finish(self, success, reason):
        if self.finished:
            return
        self.stop()
        if success:
            success, reason, measurement = self.settle_at_goal()
        else:
            measurement = self.measure_motion()
        self.finished = True
        self.stop()
        self.export_preview()
        duration = self.robot.getTime() - self.started
        distance = math.dist(self.last_position, self.goal) if self.last_position else None
        self.result = dict(run_id=self.run_id, success=bool(success), reason=reason,
                           duration=duration, final_position=self.last_position,
                           final_speed=measurement["speed"] if measurement is not None else None,
                           distance_to_goal=distance, replans=self.replans,
                           obstacle_contact_frames=self.obstacle_contact_frames,
                           chassis_contact_frames=self.chassis_contact_frames,
                           contact_tracking_available=self.contact_tracking_available,
                           tracked_obstacles=len(self.obstacle_nodes),
                           controller_log=str(self.log_path),
                           preview_path=(str(Path(self.config["preview_path"]).resolve())
                                         if self.preview_exported else None),
                           sensors=dict(available=list(self.devices),
                                        valid_frames=self.valid_sensor_frames,
                                        invalid_frames=self.invalid_sensor_frames,
                                        minimum_front_clearance=(self.minimum_clearance
                                            if math.isfinite(self.minimum_clearance) else None)))
        result_path = self.mission_path.with_name("mission_result.json")
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(result_path)
        np.save(self.mission_path.with_name("learned_obstacles.npy"), self.mapper.blocked)
        self.log_event("result", result=self.result)
        if measurement is not None:
            self.send("pos", **measurement)
        self.status("complete" if success else "failed", reason)
        self.send("done", success=bool(success), reason=reason, time=duration,
                  distance_to_goal=distance, replans=self.replans)
        if self.config.get("exit_on_complete", False):
            self.robot.simulationQuit(0 if success else 1)

    def run(self):
        try:
            while self.robot.step(self.timestep) != -1:
                if self.finished:
                    continue
                now = self.robot.getTime()
                elapsed = now - self.started
                if elapsed >= self.timeout:
                    self.finish(False, "simulation_timeout")
                    continue
                if self.detect_contacts():
                    self.finish(False, "obstacle_collision")
                    continue
                if elapsed >= 1.0 and not self.preview_exported:
                    self.export_preview()
                values = self.gps.getValues()
                north = self.compass.getValues()
                attitude = self.imu.getRollPitchYaw()
                ranges = self.lidar.getRangeImage()
                valid = (len(values) == 3 and len(north) == 3 and len(attitude) == 3 and
                         all(math.isfinite(v) for v in (*values, *north, *attitude)) and
                         math.hypot(north[0], north[1]) > 1e-8 and bool(ranges) and
                         any(v == math.inf or (math.isfinite(v) and v > 0) for v in ranges))
                if not valid:
                    self.stop()
                    self.invalid_sensor_frames += 1
                    if self.sensor_invalid_since is None:
                        self.sensor_invalid_since = now
                        self.log_event("sensor_dropout")
                    if now - self.sensor_invalid_since >= 5:
                        self.finish(False, "sensor_unavailable")
                    continue
                try:
                    yaw = compass_yaw(north)
                except ValueError:
                    self.stop()
                    self.invalid_sensor_frames += 1
                    continue
                if self.sensor_invalid_since is not None:
                    self.log_event("sensor_recovered", outage_duration=now - self.sensor_invalid_since)
                self.sensor_invalid_since = None
                self.valid_sensor_frames += 1
                position = (float(values[0]), float(values[1]))
                self.last_position = position
                speed = self.gps.getSpeed()
                if not math.isfinite(speed):
                    speed = (math.dist(position, self.previous_sample[0]) /
                             max(now - self.previous_sample[1], 1e-6)) if self.previous_sample else 0.0
                self.previous_sample = (position, now)
                if now - self.last_pose_log >= 1:
                    self.log_event("pose", position=position, speed=float(speed), yaw=yaw,
                                   goal_distance=math.dist(position, self.goal))
                    self.last_pose_log = now
                if now - self.last_telemetry >= 0.16:
                    self.send("pos", x=position[0], y=position[1], speed=float(speed),
                              heading=yaw, time=elapsed)
                    self.last_telemetry = now
                if elapsed < 0.64:
                    self.stop()
                    continue
                if max(abs(attitude[0]), abs(attitude[1])) > math.radians(35):
                    self.finish(False, "unsafe_rover_tilt")
                    continue
                distance_to_goal = math.dist(position, self.goal)
                if distance_to_goal <= 0.65:
                    self.finish(True, "goal_reached")
                    continue
                if not self.planned:
                    self.progress_position = position
                    self.progress_time = now
                    self.plan(position)
                if self.no_route_since is not None:
                    self.stop()
                    if now - self.no_route_since >= 3:
                        self.finish(False, "no_route")
                    elif now - self.last_replan >= 1:
                        self.plan(position)
                    continue
                points = list(scan_points(ranges, self.resolution, self.fov, self.max_range))
                clearance, left_clearance, right_clearance = front_clearance(points)
                self.minimum_clearance = min(self.minimum_clearance, clearance)
                self.learn_obstacles(position, values[2], yaw, points)
                if self.map_changed and now - self.last_replan >= 0.8:
                    if not self.plan(position):
                        continue
                if math.dist(position, self.progress_position) >= 0.20:
                    self.progress_position = position
                    self.progress_time = now
                elif now - self.progress_time >= 20:
                    self.finish(False, "rover_stalled")
                    continue
                if clearance < 0.30:
                    self.finish(False, "obstacle_too_close")
                    continue
                if clearance < 0.85:
                    # Pivot in place; never drive forward while emergency braking.
                    turn = 1.8 if left_clearance >= right_clearance else -1.8
                    self.set_motors(-turn, turn)
                    continue
                target = self.planner.get_next_move(position, arrival_radius_m=0.35)
                if target is None:
                    self.plan(position)
                    continue
                speed_limit = min(0.64, max(0.12, 0.55 * distance_to_goal))
                if clearance < 2.0:
                    speed_limit *= max(0.2, (clearance - 0.65) / 1.35)
                speed_limit *= max(0.35, math.cos(attitude[0]) * math.cos(attitude[1]))
                self.set_motors(*wheel_speeds(position, yaw, target, speed_limit))
        except Exception as error:
            traceback.print_exc()
            self.log_event("exception", traceback=traceback.format_exc())
            self.finish(False, f"controller_error: {type(error).__name__}: {error}")
        finally:
            self.stop()
            if not self.finished:
                self.finish(False, "simulation_ended")
            self.socket.close()


def main():
    robot = Supervisor()
    mission_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed" / "mission.json"
    controller = None
    try:
        controller = LoraController(robot, mission_path)
        controller.run()
    except Exception as error:
        traceback.print_exc()
        # Even startup failures must be visible to the launcher and acceptance test.
        result = dict(success=False, reason=f"startup_error: {type(error).__name__}: {error}",
                      run_id=None, duration=robot.getTime(), final_position=None,
                      distance_to_goal=None, replans=0, sensors={})
        try:
            config = json.loads(mission_path.read_text(encoding="utf-8-sig"))
            result["run_id"] = config.get("run_id")
            mission_path.with_name("mission_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if config.get("exit_on_complete", False):
                robot.simulationQuit(1)
        except (OSError, ValueError):
            pass
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    main()
