import tempfile
from pathlib import Path
import unittest

import numpy as np

from lora_master import make_terrain, parser, select_mission
from src.session import ProjectLock


class LauncherTests(unittest.TestCase):
    def test_synthetic_is_reproducible_and_connected(self):
        args = parser().parse_args(['--seed', '7'])
        one, spacing, metadata = make_terrain(args)
        two, _, _ = make_terrain(args)
        np.testing.assert_array_equal(one, two)
        self.assertEqual(metadata['source'], 'synthetic')
        planner, start, goal, ox, oy = select_mission(one, spacing)
        self.assertEqual(tuple(planner.waypoints_m[0]), start)
        self.assertEqual(tuple(planner.waypoints_m[-1]), goal)

    def test_no_fake_line_when_terrain_unreachable(self):
        terrain = np.indices((32, 32)).sum(axis=0) % 2 * 20.
        with self.assertRaisesRegex(ValueError, 'No traversable mission'):
            select_mission(terrain, 1.)

    def test_project_lock_releases_after_close(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lock'
            first = ProjectLock(path)
            try:
                with self.assertRaises(RuntimeError):
                    ProjectLock(path)
            finally:
                first.close()
            second = ProjectLock(path)
            second.close()
