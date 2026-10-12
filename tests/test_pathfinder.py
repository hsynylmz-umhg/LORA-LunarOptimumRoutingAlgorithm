import heapq
import math
import unittest

import numpy as np

from src.algorithm.pathfinder import Pathfinder


class PathfinderTests(unittest.TestCase):
    def planner(self, terrain=None, **kwargs):
        if terrain is None:
            terrain = np.zeros((5, 5))
        return Pathfinder(terrain, world_offset_x=0, world_offset_y=0, **kwargs)

    def test_native_sample_coordinates_and_round_trip(self):
        pf = self.planner(cell_size=2)
        self.assertEqual(pf._cell_to_meter(2, 3), (6.0, 4.0))
        self.assertEqual(pf._meter_to_cell(5.1, 3.1), (2, 3))
        for r in range(5):
            for c in range(5):
                self.assertEqual(pf._meter_to_cell(*pf._cell_to_meter(r, c)), (r, c))

    def test_out_of_bounds_or_nonfinite_plan_clears_old_route(self):
        pf = self.planner()
        for target in ((4.01, 2), (-0.01, 2), (math.nan, 2), (math.inf, 2)):
            self.assertTrue(pf.plan((0, 0), (4, 4)))
            self.assertFalse(pf.plan((0, 0), target))
            self.assertEqual(pf.waypoints_m, ())
            self.assertIsNone(pf.get_next_move((0, 0)))

    def test_steep_corner_cannot_be_cut(self):
        pf = self.planner(np.array([[0, 10], [10, 0]]))
        self.assertIsNone(pf.find_path((0, 0), (1, 1)))

    def test_occupied_corner_cannot_be_cut(self):
        blocked = np.array([[False, True], [True, False]])
        pf = self.planner(np.zeros((2, 2)), blocked=blocked)
        self.assertIsNone(pf.find_path((0, 0), (1, 1)))

    def test_steep_cross_slope_cannot_be_traversed_at_constant_height(self):
        # Every horizontal edge has zero rise, but a rover would roll sideways.
        terrain = np.repeat(np.array([[0.], [10.], [20.]]), 5, axis=1)
        occupancy = np.zeros(terrain.shape, dtype=bool)
        pf = self.planner(terrain, blocked=occupancy)
        self.assertTrue(pf.terrain_blocked.all())
        self.assertIsNone(pf.find_path((1, 0), (1, 4)))
        self.assertIsNone(pf.find_path((0, 0), (0, 0)))
        self.assertFalse(occupancy.any())

    def test_surface_slope_combines_both_axes_and_handles_singletons(self):
        rows, cols = np.mgrid[:3, :3]
        terrain = math.tan(math.radians(15)) * (rows + cols)
        pf = self.planner(terrain)
        self.assertTrue(pf.terrain_blocked.all())
        self.assertIsNone(pf.find_path((0, 0), (2, 2)))
        for shape in ((1, 1), (1, 5), (5, 1)):
            flat = self.planner(np.zeros(shape))
            self.assertFalse(flat.terrain_blocked.any())
            self.assertIsNotNone(flat.find_path((0, 0), (shape[0] - 1, shape[1] - 1)))

    def test_blocked_endpoint_and_live_occupancy_replan(self):
        blocked = np.zeros((5, 5), dtype=bool)
        pf = self.planner(blocked=blocked)
        self.assertTrue(pf.plan((0, 0), (4, 4)))
        blocked[2, :] = True
        self.assertFalse(pf.replan_from((0, 0)))
        blocked[:] = False
        blocked[0, 0] = True
        self.assertIsNone(pf.find_path((0, 0), (0, 0)))

    def test_flat_route_is_shortest_and_repeatable(self):
        pf = self.planner()
        expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        self.assertEqual(pf.find_path((0, 0), (4, 4)), expected)
        self.assertEqual(pf.find_path((0, 0), (4, 4)), expected)
        self.assertAlmostEqual(pf.path_statistics(expected)["distance_m"], 4 * math.sqrt(2))

    def test_diagonal_slope_statistics_use_diagonal_distance(self):
        pf = self.planner(np.array([[0, 0.1], [0.1, 0.2]]))
        stats = pf.path_statistics([(0, 0), (1, 1)])
        self.assertAlmostEqual(stats["distance_m"], math.sqrt(2))
        self.assertAlmostEqual(stats["max_slope_deg"], math.degrees(math.atan(0.2 / math.sqrt(2))))
        self.assertEqual(stats["climb_m"], 0.2)

    def test_astar_matches_dijkstra_cost_on_slope_penalties(self):
        # Independent shortest-path oracle, with a deliberately asymmetric hill.
        terrain = np.array([[0, 0, 0, 0, 0], [0, .05, .1, .1, 0],
                            [0, .1, .3, .1, 0], [0, .1, .1, .05, 0],
                            [0, 0, 0, 0, 0]])
        pf = self.planner(terrain)
        start, goal = (2, 0), (2, 4)
        costs, queue = {start: 0.0}, [(0.0, start)]
        def edge(a, b):
            distance = math.dist(a, b)
            slope = math.degrees(math.atan(abs(terrain[a] - terrain[b]) / distance))
            return distance * (1 + 5 * (slope / 20) ** 2)
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != costs[current]:
                continue
            if current == goal:
                break
            r, c = current
            for nr in range(max(0, r - 1), min(5, r + 2)):
                for nc in range(max(0, c - 1), min(5, c + 2)):
                    neighbour = nr, nc
                    if neighbour == current:
                        continue
                    candidate = cost + edge(current, neighbour)
                    if candidate < costs.get(neighbour, math.inf):
                        costs[neighbour] = candidate
                        heapq.heappush(queue, (candidate, neighbour))
        path = pf.find_path(start, goal)
        self.assertAlmostEqual(sum(edge(a, b) for a, b in zip(path, path[1:])), costs[goal])

    def test_waypoints_bound_turn_skipping_and_preserve_exact_goal(self):
        pf = self.planner()
        self.assertTrue(pf.plan((0, 0), (3.8, 3.7)))
        self.assertEqual(pf.waypoints_m[-1], (3.8, 3.7))
        self.assertEqual(pf.get_next_move((0, 0), arrival_radius_m=100), (1.0, 1.0))
        self.assertTrue(pf.plan((1, 1), (1.4, 1)))
        self.assertEqual(pf.get_next_move((1, 1)), (1.4, 1.0))
        self.assertIsNone(pf.get_next_move((1.4, 1)))

    def test_random_terrain_matches_independent_matrix_dijkstra(self):
        # Build an all-pairs graph from geometry, then use array-based Dijkstra.
        # No planner cost, neighbour, heuristic, or traversability helper is used.
        # This catches lost relaxations, inconsistent heuristics and slope-cost
        # mistakes across multiple cell sizes, blocked corners and disconnected maps.
        coordinates = np.array(list(np.ndindex((5, 6))))
        differences = np.abs(coordinates[:, None, :] - coordinates[None, :, :])
        cardinal = differences.sum(axis=2) == 1
        adjacent = (differences.max(axis=2) == 1)
        for seed in range(80):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                terrain = rng.uniform(-.4, .4, (5, 6))
                blocked = rng.random((5, 6)) < .17
                blocked[0, 0] = blocked[-1, -1] = False
                spacing = (.5, 1., 2.)[seed % 3]
                distances = np.linalg.norm(differences, axis=2) * spacing
                height_change = np.abs(terrain.ravel()[:, None] - terrain.ravel()[None, :])
                angles = np.degrees(np.arctan2(height_change, distances))
                valid = adjacent & (angles <= 20)
                # Independent finite differences for the full surface slope;
                # do not call np.gradient or the planner's terrain mask.
                row_grade, col_grade = np.zeros_like(terrain), np.zeros_like(terrain)
                row_grade[1:-1] = (terrain[2:] - terrain[:-2]) / (2 * spacing)
                row_grade[0] = (terrain[1] - terrain[0]) / spacing
                row_grade[-1] = (terrain[-1] - terrain[-2]) / spacing
                col_grade[:, 1:-1] = (terrain[:, 2:] - terrain[:, :-2]) / (2 * spacing)
                col_grade[:, 0] = (terrain[:, 1] - terrain[:, 0]) / spacing
                col_grade[:, -1] = (terrain[:, -1] - terrain[:, -2]) / spacing
                surface_slope = np.degrees(np.arctan(np.sqrt(row_grade ** 2 + col_grade ** 2)))
                occupied = (blocked | (surface_slope > 20)).ravel()
                valid &= ~(occupied[:, None] | occupied[None, :])
                cardinal_valid = valid & cardinal
                for a, b in zip(*np.where(valid & ~cardinal)):
                    r1, c1 = coordinates[a]
                    r2, c2 = coordinates[b]
                    corners = (r1 * 6 + c2, r2 * 6 + c1)
                    valid[a, b] = all(cardinal_valid[a, k] and cardinal_valid[k, b]
                                      for k in corners)
                edges = np.where(valid, distances * (1 + 5 * (angles / 20) ** 2), np.inf)
                best = np.full(len(coordinates), np.inf)
                visited = np.zeros(len(coordinates), dtype=bool)
                best[0] = 0.
                for _ in coordinates:
                    node = int(np.argmin(np.where(visited, np.inf, best)))
                    if visited[node] or not np.isfinite(best[node]):
                        break
                    visited[node] = True
                    best = np.minimum(best, best[node] + edges[node])
                pf = self.planner(terrain, cell_size=spacing, blocked=blocked)
                path = pf.find_path((0, 0), (4, 5))
                if not np.isfinite(best[-1]):
                    self.assertIsNone(path)
                else:
                    self.assertIsNotNone(path)
                    indices = [r * 6 + c for r, c in path]
                    total = sum(edges[a, b] for a, b in zip(indices, indices[1:]))
                    self.assertAlmostEqual(total, best[-1], places=10)

    def test_invalid_map_config_and_cells_are_rejected(self):
        for terrain in (np.zeros(3), np.zeros((0, 2)), [[0, math.nan]], [[math.inf]]):
            with self.subTest(terrain=terrain), self.assertRaises(ValueError):
                self.planner(terrain)
        for kwargs in ({"cell_size": 0}, {"max_slope": 0}, {"max_slope": 90},
                       {"slope_penalty": -1}, {"heuristic_w": .9},
                       {"blocked": np.zeros((2, 2), dtype=bool)},
                       {"blocked": np.zeros((5, 5))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.planner(**kwargs)
        pf = self.planner()
        for cell in ((-1, 0), (5, 0), (1.5, 0), (True, 0)):
            with self.subTest(cell=cell), self.assertRaises(ValueError):
                pf.find_path(cell, (0, 0))


if __name__ == "__main__":
    unittest.main()
