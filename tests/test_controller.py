"""Controller acceptance tests with deterministic kinematic Webots devices.

These exercise actual controller/planner/UDP code; physics is separately checked
by the launcher's --headless mode with a real Webots process.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def controller_module():
    module_path = ROOT / "webots/controllers/lora_controller/lora_controller.py"
    spec = importlib.util.spec_from_file_location("lora_controller_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"controller": types.SimpleNamespace(Supervisor=object)}):
        spec.loader.exec_module(module)
    return module


class Motor:
    def __init__(self):
        self.velocity = 0.0
        self.commands = []

    def setPosition(self, position):
        pass

    def setVelocity(self, velocity):
        self.velocity = velocity
        self.commands.append(velocity)

    def getMaxVelocity(self):
        return 6.4


class Sensor:
    def __init__(self, robot, name):
        self.robot, self.name = robot, name

    def enable(self, timestep):
        pass

    def getValues(self):
        if self.robot.invalid_sensors:
            return [float("nan")] * 3
        if self.name == "gps":
            return [self.robot.x, self.robot.y, .24]
        return [math.sin(self.robot.yaw), math.cos(self.robot.yaw), 0]

    def getSpeed(self):
        return abs(self.robot.speed)

    def getRollPitchYaw(self):
        return [0, 0, self.robot.yaw]

    def getHorizontalResolution(self):
        return 181

    def getFov(self):
        return math.pi

    def getMaxRange(self):
        return 12

    def getRangeImage(self):
        return self.robot.scan


class Robot:
    def __init__(self, invalid_sensors=False, scan=None):
        self.time = self.x = self.y = self.yaw = self.speed = 0.0
        self.exit_code = None
        self.invalid_sensors = invalid_sensors
        self.scan = scan if scan is not None else [float("inf")] * 181
        self.motors = [Motor() for _ in range(4)]
        self.names = dict(zip(("front left wheel", "front right wheel", "back left wheel", "back right wheel"), self.motors))

    def getDevice(self, name):
        return self.names[name] if name in self.names else Sensor(self, name)

    def getTime(self):
        return self.time

    def getBasicTimeStep(self):
        return 32

    def step(self, milliseconds):
        if self.exit_code is not None:
            return -1
        dt = milliseconds / 1000
        left, right = (self.motors[i].velocity for i in (0, 1))
        self.speed = .16 * (left + right) / 2
        self.yaw += .16 * (right - left) / .60 * dt
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.time += dt
        return 0

    def simulationQuit(self, exit_code):
        self.exit_code = exit_code


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = controller_module()

    def make_controller(self, directory, robot, goal=(3, 2), timeout=60, **options):
        path = Path(directory)
        np.save(path / "map.npy", np.zeros((21, 21)))
        mission = dict(schema_version=1, run_id="controller-test", map_path=str(path / "map.npy"),
                       cell_size=1, offset_x=-10, offset_y=-10, start=[0, 0], goal=goal,
                       timeout_sim_s=timeout, exit_on_complete=True, udp_port=59999)
        mission.update(options)
        (path / "mission.json").write_text(json.dumps(mission), encoding="utf-8")
        with patch.object(self.module, "ROOT", path):
            return self.module.LoraController(robot, path / "mission.json")

    def test_full_mission_drives_turns_and_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot()
            controller = self.make_controller(directory, robot)
            packets = []
            controller.send = lambda kind, **fields: packets.append(dict(type=kind, **fields))
            controller.run()
            result = json.loads((Path(directory) / "mission_result.json").read_text())
            self.assertTrue(result["success"], result)
            self.assertLessEqual(result["distance_to_goal"], .65)
            self.assertLess(result["final_speed"], .02)
            final_pose = [p for p in packets if p["type"] == "pos"][-1]
            self.assertEqual([final_pose["x"], final_pose["y"]], result["final_position"])
            self.assertEqual(final_pose["speed"], result["final_speed"])
            self.assertEqual(final_pose["time"], result["duration"])
            self.assertEqual(packets[-1]["type"], "done")
            self.assertEqual(robot.exit_code, 0)
            self.assertFalse(result["contact_tracking_available"])
            self.assertEqual(result["obstacle_contact_frames"], 0)
            log = [json.loads(line) for line in Path(result["controller_log"]).read_text().splitlines()]
            self.assertTrue(any(event["event"] == "result" for event in log))
            self.assertTrue(all(m.velocity == 0 for m in robot.motors))
            self.assertTrue(any(m.commands and max(m.commands) > 0 for m in robot.motors))

    def test_plan_failure_stops_existing_motor_commands_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot()
            controller = self.make_controller(directory, robot)
            controller.set_motors(4, 4)
            controller.planner.blocked[:] = True
            self.assertFalse(controller.plan((0, 0)))
            self.assertTrue(all(m.velocity == 0 for m in robot.motors))
            controller.socket.close()

    def test_sensor_failure_is_finite_time_and_writes_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(invalid_sensors=True)
            controller = self.make_controller(directory, robot)
            controller.run()
            self.assertEqual(controller.result["reason"], "sensor_unavailable")
            self.assertEqual(robot.exit_code, 1)
            self.assertTrue(all(max(map(abs, m.commands)) == 0 for m in robot.motors))

    def test_emergency_obstacle_never_moves_rover_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = [float("inf")] * 181
            scan[90] = .7
            robot = Robot(scan=scan)
            controller = self.make_controller(directory, robot, timeout=2)
            # Isolate reflex behavior from an occupancy-triggered no-route stop.
            controller.learn_obstacles = lambda *args: None
            controller.run()
            self.assertAlmostEqual(robot.x, 0)
            self.assertAlmostEqual(robot.y, 0)
            self.assertTrue(any(abs(v) > 0 for v in robot.motors[0].commands))
            self.assertFalse(controller.result["success"])


    def test_zero_and_negative_lidar_frames_fail_as_sensor_outage(self):
        for bad_range in (0, -1, -math.inf):
            with self.subTest(bad_range=bad_range), tempfile.TemporaryDirectory() as directory:
                robot = Robot(scan=[bad_range] * 181)
                controller = self.make_controller(directory, robot)
                controller.run()
                self.assertEqual(controller.result["reason"], "sensor_unavailable")
                self.assertGreaterEqual(controller.result["duration"], 5)
                self.assertLess(controller.result["duration"], 5.1)
                self.assertTrue(all(max(map(abs, m.commands)) == 0 for m in robot.motors))

    def test_later_sensor_dropout_stops_then_fails_after_five_seconds(self):
        class DropoutRobot(Robot):
            def step(self, milliseconds):
                result = super().step(milliseconds)
                self.invalid_sensors = self.time >= 1
                return result
        with tempfile.TemporaryDirectory() as directory:
            robot = DropoutRobot()
            controller = self.make_controller(directory, robot)
            controller.run()
            self.assertEqual(controller.result["reason"], "sensor_unavailable")
            self.assertGreater(controller.result["sensors"]["valid_frames"], 0)
            self.assertGreaterEqual(controller.result["duration"], 6)
            self.assertLess(controller.result["duration"], 6.1)
            # Rover moved before failure; every command after the first second is zero.
            self.assertTrue(any(v > 0 for v in robot.motors[0].commands[:32]))
            self.assertTrue(all(v == 0 for v in robot.motors[0].commands[35:]))

    def test_obstacle_contact_fails_but_chassis_ground_contact_only_counts(self):
        class ContactNode:
            def __init__(self, robot, since):
                self.robot, self.since = robot, since
                self.enabled = False
            def enableContactPointsTracking(self, timestep, descendants):
                assert descendants is False
                self.enabled = True
            def getContactPoints(self, descendants):
                assert self.enabled and descendants is False
                return [object()] if self.robot.time >= self.since else []
        class ContactRobot(Robot):
            def __init__(self, obstacle_since):
                super().__init__()
                self.chassis = ContactNode(self, 0)
                self.obstacle = ContactNode(self, obstacle_since)
            def getSelf(self):
                return self.chassis
            def getFromDef(self, name):
                return self.obstacle if name == "OBSTACLE_0" else None
        for obstacle_since in (1, math.inf):
            with self.subTest(obstacle_since=obstacle_since), tempfile.TemporaryDirectory() as directory:
                robot = ContactRobot(obstacle_since)
                controller = self.make_controller(directory, robot, obstacle_count=1)
                controller.run()
                result = controller.result
                self.assertTrue(result["contact_tracking_available"])
                self.assertEqual(result["tracked_obstacles"], 1)
                self.assertGreater(result["chassis_contact_frames"], 0)
                if math.isfinite(obstacle_since):
                    self.assertFalse(result["success"])
                    self.assertEqual(result["reason"], "obstacle_collision")
                    self.assertEqual(result["obstacle_contact_frames"], 1)
                else:
                    self.assertTrue(result["success"])
                    self.assertEqual(result["obstacle_contact_frames"], 0)

    def test_preview_export_is_optional_and_controller_log_is_bounded(self):
        class PreviewRobot(Robot):
            def __init__(self):
                super().__init__()
                self.exports = []
            def exportImage(self, filename, quality):
                self.exports.append((filename, quality))
                Path(filename).write_bytes(b"preview-test")
        with tempfile.TemporaryDirectory() as directory:
            robot = PreviewRobot()
            preview = str(Path(directory) / "preview.png")
            controller = self.make_controller(directory, robot, preview_path=preview)
            for i in range(65):
                controller.log_event("bounded_log_test", message="x" * 10000, index=i)
            self.assertLessEqual(controller.log_path.stat().st_size, 512 * 1024)
            for line in controller.log_path.read_text().splitlines():
                json.loads(line)
            controller.run()
            self.assertEqual(len(robot.exports), 2)
            self.assertEqual(controller.result["preview_path"], preview)
            self.assertTrue(Path(preview).exists())


    def test_completion_waits_for_real_braking_and_reports_current_pose(self):
        class CoastingRobot(Robot):
            def step(self, milliseconds):
                if self.exit_code is not None:
                    return -1
                assert all(m.velocity == 0 for m in self.motors)
                dt = milliseconds / 1000
                self.speed *= .5
                self.x += self.speed * dt
                self.time += dt
                return 0
        with tempfile.TemporaryDirectory() as directory:
            robot = CoastingRobot()
            robot.x, robot.speed = 2.5, .4
            controller = self.make_controller(directory, robot, goal=(3, 0))
            controller.last_position = (2.4, 0)  # Deliberately stale telemetry sample.
            packets = []
            controller.send = lambda kind, **fields: packets.append(dict(type=kind, **fields))
            controller.finish(True, "goal_reached")
            result = controller.result
            self.assertTrue(result["success"])
            self.assertGreaterEqual(result["duration"], .2)
            self.assertLessEqual(result["duration"], 3)
            self.assertGreater(result["final_position"][0], 2.5)
            self.assertAlmostEqual(result["final_speed"], robot.speed)
            self.assertGreater(result["final_speed"], 0)  # Must not synthesize zero speed.
            self.assertLess(result["final_speed"], .02)
            final_pose = [p for p in packets if p["type"] == "pos"][-1]
            self.assertEqual(final_pose["x"], robot.x)
            self.assertEqual(final_pose["speed"], robot.speed)
            self.assertEqual(packets[-1]["type"], "done")
            self.assertTrue(all(m.velocity == 0 for m in robot.motors))
            controller.socket.close()

    def test_completion_rejects_unstopped_or_drifted_rover(self):
        class SlidingRobot(Robot):
            def step(self, milliseconds):
                self.speed = .05
                self.x += self.speed * milliseconds / 1000
                self.time += milliseconds / 1000
                return 0
        class DriftRobot(Robot):
            def step(self, milliseconds):
                self.x = .8
                self.speed = 0
                self.time += milliseconds / 1000
                return 0
        for robot_class, reason in ((SlidingRobot, "rover_did_not_stop"),
                                    (DriftRobot, "goal_drift_after_stop")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                robot = robot_class()
                controller = self.make_controller(directory, robot, goal=(0, 0))
                controller.finish(True, "goal_reached")
                self.assertFalse(controller.result["success"])
                self.assertEqual(controller.result["reason"], reason)
                self.assertLessEqual(controller.result["duration"], 3)
                self.assertEqual(robot.exit_code, 1)
                self.assertTrue(all(m.velocity == 0 for m in robot.motors))
                controller.socket.close()

    def test_obstacle_contact_during_braking_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot()
            controller = self.make_controller(directory, robot, goal=(0, 0))
            def contact_during_braking():
                if robot.time >= .1:
                    controller.obstacle_contact_frames += 1
                    return True
                return False
            controller.detect_contacts = contact_during_braking
            controller.finish(True, "goal_reached")
            self.assertFalse(controller.result["success"])
            self.assertEqual(controller.result["reason"], "obstacle_collision")
            self.assertEqual(controller.result["obstacle_contact_frames"], 1)
            self.assertEqual(robot.exit_code, 1)
            self.assertTrue(all(m.velocity == 0 for m in robot.motors))
            controller.socket.close()


if __name__ == "__main__":
    unittest.main()
