import unittest

import numpy as np

from src.algorithm.slam_mapper import SlamMapper


class SlamMapperTests(unittest.TestCase):
    def test_conservative_inflation_and_repeat_detection(self):
        terrain = np.zeros((5, 5))
        mapper = SlamMapper(terrain)
        reference = mapper.blocked
        self.assertTrue(mapper.mark_obstacles([(2, 2)]))
        expected = {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
        self.assertEqual(set(map(tuple, np.argwhere(mapper.blocked))), expected)
        self.assertFalse(mapper.mark_obstacles([(2, 2)]))
        self.assertIs(mapper.blocked, reference)
        self.assertTrue(np.array_equal(terrain, np.zeros((5, 5))))

    def test_edges_and_outside_detections(self):
        mapper = SlamMapper(np.zeros((3, 3)))
        self.assertFalse(mapper.mark_obstacles([(-1, 1), (3, 2)]))
        self.assertTrue(mapper.mark_obstacles([(0, 0)]))
        self.assertEqual(int(mapper.blocked.sum()), 3)

    def test_zero_radius_and_atomic_validation(self):
        mapper = SlamMapper(np.zeros((3, 3)), inflation_radius_m=0)
        with self.assertRaises(ValueError):
            mapper.mark_obstacles([(0, 0), (1.5, 2)])
        self.assertFalse(mapper.blocked.any())
        mapper.mark_obstacles([(1, 1)])
        self.assertEqual(int(mapper.blocked.sum()), 1)

    def test_configuration_validation(self):
        for kwargs in ({"cell_size": 0}, {"inflation_radius_m": -1}):
            with self.assertRaises(ValueError):
                SlamMapper(np.zeros((3, 3)), **kwargs)


if __name__ == "__main__":
    unittest.main()
