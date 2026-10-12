"""World regressions for the geometry and coordinate failures in the old MVP."""
import math
from pathlib import Path
import re
import tempfile
import unittest

from src.webots.world_builder import build_world


class WorldBuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "webots" / "worlds" / "test.wbt"
        self.mission = {
            "cell_size": 2.0,
            "offset_x": 100.0,
            "offset_y": -10.0,
            "start": [103.0, -7.0],
            "goal": [103.0, -6.0],
            "mission_file": str(Path(self.directory.name) / "mission with spaces.json"),
            "seed": 42,
            "run_id": "world regression",
        }
        # Deliberately rectangular and asymmetric so a row/column transpose fails.
        self.heightmap = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]

    def make_world(self, **kwargs):
        build_world(self.heightmap, self.mission, self.path, **kwargs)
        return self.path.read_text(encoding="utf-8")

    def test_terrain_and_spawn_share_world_coordinates(self):
        world = self.make_world()
        terrain = world.split("DEF LUNAR_SURFACE Solid {", 1)[1].split("# Visual marker", 1)[0]
        self.assertIn("translation 100 -10 0", terrain)
        self.assertRegex(terrain, r"xDimension 4\s+yDimension 3")
        self.assertRegex(terrain, r"xSpacing 2\s+ySpacing 2")
        heights = re.search(r"height \[([^]]+)\]", terrain).group(1)
        self.assertEqual(list(map(float, heights.split())), list(range(4)) + list(range(10, 14)) + list(range(20, 24)))
        rover = world.split("DEF LORA_ROVER Robot {", 1)[1]
        x, y, z = map(float, re.search(r"translation ([^\n]+)", rover).group(1).split())
        self.assertEqual((x, y), (103.0, -7.0))
        self.assertAlmostEqual(z, 16.85)  # bilinear height 16.5 + drop clearance .35
        yaw = float(re.search(r"rotation 0 0 1 ([^\n]+)", rover).group(1))
        self.assertAlmostEqual(yaw, math.pi / 2)
        self.assertIn('coordinateSystem "ENU"', world)
        self.assertIn("boundingObject USE TERRAIN_GEOMETRY", terrain)

    def test_camera_optical_axis_points_at_rover_with_level_horizon(self):
        world = self.make_world()
        viewpoint = world.split("Viewpoint {", 1)[1].split("\n}", 1)[0]
        position = tuple(map(float, re.search(r"position ([^\n]+)", viewpoint).group(1).split()))
        ax, ay, az, angle = map(float, re.search(r"orientation ([^\n]+)", viewpoint).group(1).split())
        self.assertAlmostEqual(math.hypot(ax, ay, az), 1.0, places=8)
        # Independently rotate native camera +X with Rodrigues' formula.
        c, sine = math.cos(angle), math.sin(angle)
        forward = (c + ax * ax * (1 - c),
                   az * sine + ay * ax * (1 - c),
                   -ay * sine + az * ax * (1 - c))
        target = (103.0, -7.0, 16.5 + 0.24)
        delta = tuple(target[i] - position[i] for i in range(3))
        distance = math.hypot(*delta)
        for actual, expected in zip(forward, (v / distance for v in delta)):
            self.assertAlmostEqual(actual, expected, places=8)
        # Native camera left is +Y; no roll means its world Z component is zero.
        left_z = ax * sine + az * ay * (1 - c)
        self.assertAlmostEqual(left_z, 0.0, places=8)

    def test_wheels_have_real_locations_clearance_and_consistent_drive_axes(self):
        world = self.make_world()
        joints = world.split("    HingeJoint {")[1:]
        self.assertEqual(len(joints), 4)
        names, wheel_centers = set(), set()
        for joint in joints:
            anchor = tuple(map(float, re.search(r"anchor ([^\n]+)", joint).group(1).split()))
            endpoint = tuple(map(float, re.search(r"endPoint Solid \{\s+translation ([^\n]+)", joint).group(1).split()))
            self.assertEqual(anchor, endpoint)
            wheel_centers.add(endpoint)
            names.add(re.search(r'RotationalMotor \{\s+name "([^"]+)"', joint).group(1))
            axle = tuple(map(float, re.search(r"axis ([^\n]+)", joint).group(1).split()))
            self.assertEqual(axle, (0, 1, 0))  # rolling velocity -omega x (0,0,-r) is +X
            self.assertIn("rotation 1 0 0 -1.57079632679", joint)
            radius = float(re.search(r"boundingObject Cylinder \{ radius ([\d.]+)", joint).group(1))
            chassis_clearance = -0.18 / 2 - (endpoint[2] - radius)
            self.assertGreater(chassis_clearance, 0.1)
            self.assertGreater(float(re.search(r"maxTorque ([\d.]+)", joint).group(1)), 1)
            self.assertIn('contactMaterial "rover tire"', joint)
        self.assertEqual(names, {"front left wheel", "front right wheel", "back left wheel", "back right wheel"})
        self.assertEqual({(x, y) for x, y, _ in wheel_centers}, {(0.24, 0.30), (0.24, -0.30), (-0.24, 0.30), (-0.24, -0.30)})
        self.assertIn("gravity 1.62", world)
        self.assertIn('material2 "lunar regolith"', world)
        self.assertIn("bounce 0", world)

    def test_modern_world_is_self_contained_and_passes_mission_to_controller(self):
        world = self.make_world()
        self.assertTrue(world.startswith("#VRML_SIM R2025a utf8"))
        for obsolete_or_remote in ("EXTERNPROTO", "http://", "https://", "zDimension", "zSpacing", 'followType "Follow"'):
            self.assertNotIn(obsolete_or_remote, world)
        self.assertNotRegex(world, r"\bfps\b")
        self.assertIn('followType "Tracking Shot"', world)
        mission_argument = Path(self.mission["mission_file"]).resolve().as_posix()
        self.assertIn(f'controllerArgs [ "{mission_argument}" ]', world)
        self.assertIn('controller "lora_controller"', world)
        self.assertIn("supervisor TRUE", world)
        self.assertIn('GPS { name "gps" }', world)
        self.assertIn('Compass { name "compass" }', world)
        self.assertIn('InertialUnit { name "inertial unit" }', world)

    def test_lidar_sits_above_body_and_has_no_local_mesh_to_hit(self):
        world = self.make_world()
        lidar = re.search(r"Lidar \{(.*?)\n    \}", world, re.DOTALL).group(1)
        x, y, z = map(float, re.search(r"translation ([^\n]+)", lidar).group(1).split())
        self.assertEqual((x, y), (0.22, 0))
        self.assertGreater(z, 0.18 / 2)
        self.assertNotIn("rotation", lidar)
        self.assertNotIn("Shape", lidar)
        self.assertIn("numberOfLayers 1", lidar)
        self.assertGreater(float(re.search(r"maxRange ([\d.]+)", lidar).group(1)), 5)

    def test_obstacle_is_static_and_based_on_the_local_terrain_height(self):
        world = self.make_world(obstacles=[{"x": 102, "y": -8, "radius": 0.4, "height": 1.2}])
        obstacle = world.split("DEF OBSTACLE_0 Solid {", 1)[1].split("DEF LORA_ROVER", 1)[0]
        self.assertIn("translation 102 -8 11.6", obstacle)
        self.assertIn("radius 0.4 height 1.2", obstacle)
        self.assertIn("boundingObject USE OBSTACLE_GEOMETRY_0", obstacle)
        self.assertNotIn("physics", obstacle)
        markers = world.split("DEF START_MARKER", 1)[1].split("DEF OBSTACLE_0", 1)[0]
        self.assertNotIn("boundingObject", markers)

    def test_invalid_data_fails_before_overwriting_existing_world(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("existing world", encoding="utf-8")
        cases = [
            ([], {}),
            ([[1, 2], [3]], {}),
            ([[1, 2], [3, float("nan")]], {}),
            (self.heightmap, {"cell_size": 0}),
            (self.heightmap, {"offset_y": float("inf")}),
            (self.heightmap, {"start": [0, 0]}),
            (self.heightmap, {"goal": [float("nan"), 0]}),
        ]
        for heightmap, overrides in cases:
            with self.subTest(overrides=overrides, heightmap=heightmap):
                with self.assertRaises(ValueError):
                    build_world(heightmap, dict(self.mission, **overrides), self.path)
                self.assertEqual(self.path.read_text(encoding="utf-8"), "existing world")
        with self.assertRaises(ValueError):
            build_world(self.heightmap, self.mission, self.path, [{"x": 102, "y": -8, "radius": -1, "height": 1}])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "existing world")


if __name__ == "__main__":
    unittest.main()
