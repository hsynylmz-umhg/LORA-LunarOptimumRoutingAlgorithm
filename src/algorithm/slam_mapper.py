"""Conservative LIDAR occupancy updates; this is not a localization SLAM system.

Elevation and obstacle occupancy are deliberately separate. Heights are never
blurred or changed in response to an obstacle detection.
"""

import math
import operator

import numpy as np


class SlamMapper:
    def __init__(self, heightmap, cell_size=1.0, inflation_radius_m=0.7):
        terrain = np.asarray(heightmap)
        if terrain.ndim != 2 or not terrain.size or not np.isfinite(terrain).all():
            raise ValueError("heightmap must be a nonempty finite 2D array")
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        if not math.isfinite(inflation_radius_m) or inflation_radius_m < 0:
            raise ValueError("inflation_radius_m must be nonnegative and finite")
        self.cell_size = float(cell_size)
        self.inflation_radius_m = float(inflation_radius_m)
        self.blocked = np.zeros(terrain.shape, dtype=bool)
        # Round up to preserve clearance on coarse terrain grids. At 1m spacing,
        # a 0.7m radius occupies the sample and its four cardinal neighbours.
        radius = math.ceil(inflation_radius_m / cell_size)
        self._offsets = tuple((dr, dc)
                              for dr in range(-radius, radius + 1)
                              for dc in range(-radius, radius + 1)
                              if dr * dr + dc * dc <= radius * radius)

    def mark_obstacles(self, cells):
        """Mark/inflate (row, col) detections in place; return whether map changed.

        Valid integer detections beyond the map are ignored. Malformed detections
        raise before any mutation, so callers cannot accidentally partly update.
        """
        parsed = []
        for cell in cells:
            try:
                r, c = cell
                if isinstance(r, (bool, np.bool_)) or isinstance(c, (bool, np.bool_)):
                    raise TypeError
                parsed.append((operator.index(r), operator.index(c)))
            except (TypeError, ValueError) as exc:
                raise ValueError("obstacle cells must contain two integers") from exc
        rows, cols = self.blocked.shape
        changed = False
        for r, c in parsed:
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            for dr, dc in self._offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not self.blocked[nr, nc]:
                    self.blocked[nr, nc] = True
                    changed = True
        return changed
