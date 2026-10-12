"""Deterministic slope-aware A* on an ENU terrain sample grid.

Rows run along northing (Y), columns along easting (X), and values are
elevations (Z) in metres. ``world_offset_z`` is retained as a legacy name
for the second *horizontal* offset; new callers may use ``world_offset_y``.
"""

import heapq
import math
import operator

import numpy as np

MAX_SLOPE_DEG = 20.0
SLOPE_PENALTY = 5.0
CELL_SIZE_M = 1.0
HEURISTIC_WEIGHT = 1.0
WORLD_OFFSET_X = -250.0
WORLD_OFFSET_Z = -250.0
ARRIVAL_RADIUS_M = 0.35

_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0),
               (1, 1), (1, -1), (-1, -1), (-1, 1))


class Pathfinder:
    """Plan minimum distance-plus-slope-cost routes with optional occupancy.

    ``blocked`` is a shared boolean array: in-place obstacle updates affect
    subsequent plans. A heuristic weight of 1 gives optimal A*; larger values
    explicitly select weighted A* and trade optimality for speed.
    A separate sampled surface-slope mask excludes excessive lateral slopes.
    This is a conservative grid check, not a full chassis/contact simulation.
    """

    def __init__(self, heightmap, cell_size=CELL_SIZE_M,
                 max_slope=MAX_SLOPE_DEG, slope_penalty=SLOPE_PENALTY,
                 heuristic_w=HEURISTIC_WEIGHT, world_offset_x=WORLD_OFFSET_X,
                 world_offset_z=WORLD_OFFSET_Z, blocked=None, *,
                 world_offset_y=None):
        terrain = np.asarray(heightmap, dtype=np.float64)
        if terrain.ndim != 2 or not terrain.size:
            raise ValueError("heightmap must be a nonempty 2D array")
        if not np.isfinite(terrain).all():
            raise ValueError("heightmap must contain only finite elevations")
        values = (cell_size, max_slope, slope_penalty, heuristic_w,
                  world_offset_x, world_offset_z)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("planner configuration must be finite")
        if cell_size <= 0 or not 0 < max_slope < 90:
            raise ValueError("cell_size must be positive and 0 < max_slope < 90")
        if slope_penalty < 0 or heuristic_w < 1:
            raise ValueError("slope_penalty must be >= 0 and heuristic_w >= 1")
        if world_offset_y is not None and not math.isfinite(float(world_offset_y)):
            raise ValueError("world_offset_y must be finite")
        self.heightmap = terrain.copy()
        self.cell_size = float(cell_size)
        self.max_slope_deg = float(max_slope)
        self.slope_penalty = float(slope_penalty)
        self.heuristic_w = float(heuristic_w)
        self.offset_x = float(world_offset_x)
        self.offset_y = float(world_offset_z if world_offset_y is None else world_offset_y)
        self.offset_z = self.offset_y  # legacy horizontal-axis alias
        self.n_rows, self.n_cols = terrain.shape
        if blocked is None:
            blocked = np.zeros(terrain.shape, dtype=bool)
        if not isinstance(blocked, np.ndarray) or blocked.dtype != np.bool_:
            raise ValueError("blocked must be a boolean numpy array")
        if blocked.shape != terrain.shape:
            raise ValueError("blocked must have the heightmap shape")
        self.blocked = blocked
        self._max_grade = math.tan(math.radians(self.max_slope_deg))
        # Along-edge slope alone misses rollover risk on a contour traverse.
        # Calculate the total local surface gradient and retain it separately
        # from mutable LIDAR occupancy. Singleton axes have no measurable grade.
        with np.errstate(over="ignore", invalid="ignore"):
            grade_y = (np.gradient(self.heightmap, self.cell_size, axis=0)
                       if self.n_rows > 1 else np.zeros_like(self.heightmap))
            grade_x = (np.gradient(self.heightmap, self.cell_size, axis=1)
                       if self.n_cols > 1 else np.zeros_like(self.heightmap))
            grade = np.hypot(grade_x, grade_y)
        self.terrain_blocked = (~np.isfinite(grade) |
                                (grade > self._max_grade + 1e-12))
        self._path = []
        self._path_idx = 0
        self._waypoints = ()
        self._goal_m = None
        self.last_iterations = 0

    def _in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _validate_cell(self, cell):
        try:
            r, c = cell
            if isinstance(r, (bool, np.bool_)) or isinstance(c, (bool, np.bool_)):
                raise TypeError
            r, c = operator.index(r), operator.index(c)
        except (TypeError, ValueError) as exc:
            raise ValueError("grid coordinates must be two integers") from exc
        if not self._in_bounds(r, c):
            raise ValueError(f"grid coordinate {(r, c)} is outside the terrain")
        return r, c

    def _meter_to_cell(self, x_m, z_m):
        """Convert horizontal (X, Y) metres to the nearest terrain sample.

        Positions beyond the physical sample extent are rejected, never clipped.
        The second argument keeps its legacy name for existing callers.
        """
        x, y = float(x_m), float(z_m)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("world coordinates must be finite")
        col = (x - self.offset_x) / self.cell_size
        row = (y - self.offset_y) / self.cell_size
        eps = 1e-9
        if not (-eps <= col <= self.n_cols - 1 + eps and
                -eps <= row <= self.n_rows - 1 + eps):
            raise ValueError("world coordinate is outside the terrain")
        return int(math.floor(row + 0.5)), int(math.floor(col + 0.5))

    def _cell_to_meter(self, row, col):
        row, col = self._validate_cell((row, col))
        return (self.offset_x + col * self.cell_size,
                self.offset_y + row * self.cell_size)

    def _slope_cost(self, r1, c1, r2, c2, base_cost):
        distance = self.cell_size * base_cost
        dh = abs(float(self.heightmap[r2, c2] - self.heightmap[r1, c1]))
        if (self.blocked[r2, c2] or self.terrain_blocked[r2, c2] or
                dh > self._max_grade * distance + 1e-12):
            return None
        slope = math.degrees(math.atan2(dh, distance))
        return distance * (1 + self.slope_penalty * (slope / self.max_slope_deg) ** 2)

    def _diagonal_clear(self, r, c, nr, nc):
        # Both adjacent cardinal passages must be usable. This prevents crossing
        # between blocked cells or jumping diagonally over steep terrain corners.
        for ar, ac in ((r, nc), (nr, c)):
            if (self._slope_cost(r, c, ar, ac, 1.0) is None or
                    self._slope_cost(ar, ac, nr, nc, 1.0) is None):
                return False
        return True

    def _heuristic(self, r, c, gr, gc):
        dr, dc = abs(r - gr), abs(c - gc)
        octile = max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        return self.heuristic_w * self.cell_size * octile

    def find_path(self, start, goal):
        """Return grid samples from start through goal, or None if unreachable."""
        start, goal = self._validate_cell(start), self._validate_cell(goal)
        self.last_iterations = 0
        if (self.blocked[start] or self.blocked[goal] or
                self.terrain_blocked[start] or self.terrain_blocked[goal]):
            return None
        best = np.full(self.heightmap.shape, math.inf)
        best[start] = 0.0
        parents = {}
        h = self._heuristic(*start, *goal)
        heap = [(h, h, 0, 0.0, start)]
        serial = 0
        while heap:
            _, _, _, cost, current = heapq.heappop(heap)
            if cost != best[current]:
                continue
            self.last_iterations += 1
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            r, c = current
            for dr, dc in _DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                diagonal = bool(dr and dc)
                move = self._slope_cost(r, c, nr, nc, math.sqrt(2) if diagonal else 1.0)
                if move is None or (diagonal and not self._diagonal_clear(r, c, nr, nc)):
                    continue
                new_cost = cost + move
                if new_cost < best[nr, nc]:
                    best[nr, nc] = new_cost
                    parents[(nr, nc)] = current
                    serial += 1
                    h = self._heuristic(nr, nc, *goal)
                    heapq.heappush(heap, (new_cost + h, h, serial, new_cost, (nr, nc)))
        return None

    @property
    def waypoints_m(self):
        """Immutable world-space route, including the exact requested endpoint."""
        return self._waypoints

    def plan(self, start_m, goal_m):
        self._path, self._waypoints, self._path_idx = [], (), 0
        try:
            start = self._meter_to_cell(*start_m)
            goal = self._meter_to_cell(*goal_m)
            self._goal_m = tuple(float(v) for v in goal_m)
        except (ValueError, TypeError, OverflowError):
            self._goal_m = None
            return False
        path = self.find_path(start, goal)
        if path is None:
            return False
        self._path = path
        points = [self._cell_to_meter(*cell) for cell in path]
        if len(path) == 1:
            points = [self._goal_m]
        elif math.dist(points[-1], self._goal_m) > 1e-9:
            points.append(self._goal_m)
        self._waypoints = tuple(points)
        return True

    def get_next_move(self, current_pos_m, arrival_radius_m=ARRIVAL_RADIUS_M):
        if not math.isfinite(arrival_radius_m) or arrival_radius_m <= 0:
            raise ValueError("arrival_radius_m must be positive and finite")
        # Keep accepted radii below half a sample interval so a tight turn cannot
        # be skipped by a caller using a multi-metre goal tolerance.
        radius = min(arrival_radius_m, self.cell_size * 0.49)
        position = tuple(float(v) for v in current_pos_m)
        if len(position) != 2 or not all(math.isfinite(v) for v in position):
            raise ValueError("current position must contain two finite coordinates")
        while self._path_idx < len(self._waypoints):
            waypoint = self._waypoints[self._path_idx]
            if math.dist(position, waypoint) > radius:
                return waypoint
            self._path_idx += 1
        return None

    def replan_from(self, current_pos_m, goal_m=None):
        target = self._goal_m if goal_m is None else goal_m
        return False if target is None else self.plan(current_pos_m, target)

    def path_statistics(self, path):
        if not path:
            return {}
        cells = [self._validate_cell(cell) for cell in path]
        heights = np.array([self.heightmap[cell] for cell in cells])
        delta = np.diff(heights)
        distances = [math.dist(a, b) * self.cell_size for a, b in zip(cells, cells[1:])]
        if any(d == 0 for d in distances):
            raise ValueError("a route cannot contain consecutive duplicate samples")
        slopes = [math.degrees(math.atan2(abs(dh), d)) for dh, d in zip(delta, distances)]
        return {"step_count": len(cells), "distance_m": sum(distances),
                "start_height_m": float(heights[0]), "end_height_m": float(heights[-1]),
                "net_elevation_m": float(heights[-1] - heights[0]),
                "max_slope_deg": max(slopes, default=0.0),
                "avg_slope_deg": sum(slopes) / len(slopes) if slopes else 0.0,
                "climb_m": float(np.maximum(delta, 0).sum()),
                "descent_m": float(-np.minimum(delta, 0).sum())}

    def current_path_stats(self):
        return self.path_statistics(self._path)
