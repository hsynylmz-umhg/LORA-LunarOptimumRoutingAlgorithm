import math
import unittest

from src.navigation import (angle_error, bearing, compass_yaw, front_clearance,
                            project_return, scan_points, telemetry_waypoints, wheel_speeds)


class NavigationTests(unittest.TestCase):
    def test_compass_matches_all_cardinal_world_headings(self):
        for yaw in (0, math.pi / 2, math.pi, -math.pi / 2):
            north_in_robot = (math.sin(yaw), math.cos(yaw), 0)
            self.assertAlmostEqual(angle_error(compass_yaw(north_in_robot), yaw), 0)
        with self.assertRaises(ValueError):
            compass_yaw((float("nan"), 1, 0))
        with self.assertRaises(ValueError):
            compass_yaw((0, 0, 1))

    def test_wheel_differential_turns_toward_target(self):
        left, right = wheel_speeds((0, 0), 0, (0, 10))
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        self.assertAlmostEqual(left + right, 0)
        left, right = wheel_speeds((0, 0), 0, (0, -10))
        self.assertGreater(left, 0)
        self.assertLess(right, 0)
        self.assertEqual(wheel_speeds((0, 0), 0, (0, 0)), (0, 0))
        self.assertEqual(wheel_speeds((0, 0), float("nan"), (1, 0)), (0, 0))

    def test_straight_commands_are_bounded_and_slow_near_target(self):
        fast = wheel_speeds((0, 0), 0, (10, 0), 100)
        self.assertEqual(fast, (6, 6))
        slow = wheel_speeds((0, 0), 0, (.1, 0))
        self.assertGreater(slow[0], 0)
        self.assertLess(slow[0], 1)

    def test_lidar_scan_has_correct_left_right_and_multilayer_index(self):
        points = list(scan_points([1, 2, 3, 4, 5, 6], 3, math.pi, 12))
        self.assertEqual(points[0], (1, math.pi / 2))
        self.assertEqual(points[2], (3, -math.pi / 2))
        self.assertEqual(points[3], (4, math.pi / 2))
        self.assertEqual(list(scan_points([float("nan"), float("inf"), -1, 0, 13], 5, math.pi, 12)), [])

    def test_projection_uses_yaw_and_sensor_mount(self):
        self.assertEqual(project_return((1, 2), 0, 3, 0), (4.22, 2))
        x, y = project_return((1, 2), math.pi / 2, 3, 0)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 5.22)
        self.assertEqual(front_clearance([(1, math.pi / 2), (2, 0)]), (2, 2, 2))

    def test_long_telemetry_route_retains_destination(self):
        points = telemetry_waypoints([(i, i / 3) for i in range(6000)])
        self.assertEqual(len(points), 1500)
        self.assertEqual(points[0], [0, 0])
        self.assertEqual(points[-1], [5999, round(5999 / 3, 4)])

    def test_bearing_wrap_boundary(self):
        self.assertAlmostEqual(bearing((0, 0), (0, 1)), math.pi / 2)
        self.assertAlmostEqual(angle_error(-math.pi + .01, math.pi - .01), .02)


if __name__ == "__main__":
    unittest.main()
