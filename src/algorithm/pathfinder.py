# filepath: LORA/src/algorithm/pathfinder.py
"""
L.O.R.A. — Lunar Optimum Routing Algorithm
Aşama 3-A (FINAL): A* Tabanlı Rota Planlayıcı

DEĞİŞİKLİKLER (Final):
  + plan(start_m, goal_m)           → metre koordinatıyla rota planla
  + get_next_move(current_pos_m)    → Webots her timestep'te çağırır;
                                       bir sonraki waypoint'i metre döndürür
  + replan_from(current_pos_m)      → LIDAR override sonrası acil yeniden plan
  + _meter_to_cell() / _cell_to_meter() → koordinat köprüsü
  Mevcut find_path() ve path_statistics() KORUNDU — hiçbir satır silinmedi.
"""

import heapq
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  PARAMETRELERİ BURADAN DEĞİŞTİR
# ─────────────────────────────────────────────

MAX_SLOPE_DEG    = 20.0   # Bu açının üstündeki hücreler geçilemez.
SLOPE_PENALTY    = 5.0    # Eğim maliyet çarpanı.

# v2 harita: 500 grid / 500 m → 1.0 m/hücre
# (Eski v1 harita için 2000m/500grid = 4.0 m kullanın)
CELL_SIZE_M      = 1.0

HEURISTIC_WEIGHT = 1.2    # 1.0 = optimal, >1.0 = daha hızlı/az optimal

# ElevationGrid merkez ofseti — world_builder.py ile aynı olmalı
# 500×500 m dünya, (0,0) merkez → offset = -250 m
WORLD_OFFSET_X   = -250.0
WORLD_OFFSET_Z   = -250.0

# Waypoint "varıldı" kabul mesafesi (metre)
ARRIVAL_RADIUS_M = 2.5

# 8 yönlü hareket (dr, dc, base_cost)
_DIRECTIONS = [
    ( 0,  1, 1.0), ( 0, -1, 1.0), ( 1,  0, 1.0), (-1,  0, 1.0),
    ( 1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
    (-1,  1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


# ─────────────────────────────────────────────
#  YARDIMCI VERİ YAPISI
# ─────────────────────────────────────────────

@dataclass(order=True)
class _Node:
    """A* öncelik kuyruğu için karşılaştırılabilir düğüm."""
    f_cost: float
    g_cost: float             = field(compare=False)
    pos:    tuple             = field(compare=False)
    parent: Optional["_Node"] = field(default=None, compare=False)


# ─────────────────────────────────────────────
#  ANA SINIF
# ─────────────────────────────────────────────

class Pathfinder:
    """
    A* tabanlı, eğim-duyarlı rota planlayıcı.

    Webots entegrasyonu (birincil kullanım):
        pf = Pathfinder(heightmap)
        pf.plan(start_m=(gps_x0, gps_z0), goal_m=(gps_x1, gps_z1))

        # Her Webots timestep:
        target = pf.get_next_move(current_pos_m=(gps_x, gps_z))
        if target is None:
            print("Hedefe ulaşıldı / rota tükendi")

    Standalone test (grid koordinatı):
        path = pf.find_path(start=(r0,c0), goal=(r1,c1))
        stats = pf.path_statistics(path)
    """

    def __init__(self,
                 heightmap:      np.ndarray,
                 cell_size:      float = CELL_SIZE_M,
                 max_slope:      float = MAX_SLOPE_DEG,
                 slope_penalty:  float = SLOPE_PENALTY,
                 heuristic_w:    float = HEURISTIC_WEIGHT,
                 world_offset_x: float = WORLD_OFFSET_X,
                 world_offset_z: float = WORLD_OFFSET_Z):
        """
        Args:
            heightmap:      2D numpy yükseklik matrisi (satır=Z, sütun=X).
            cell_size:      Hücre başına metre.
            max_slope:      Geçilemez eğim eşiği (derece).
            slope_penalty:  Eğim maliyet çarpanı.
            heuristic_w:    A* sezgisel ağırlık faktörü.
            world_offset_x: ElevationGrid X ofseti (metre).
            world_offset_z: ElevationGrid Z ofseti (metre).
        """
        if heightmap.ndim != 2:
            raise ValueError("heightmap 2 boyutlu numpy dizisi olmalıdır.")

        self.heightmap      = heightmap.astype(np.float64)
        self.cell_size      = cell_size
        self.max_slope_deg  = max_slope
        self.slope_penalty  = slope_penalty
        self.heuristic_w    = heuristic_w
        self.offset_x       = world_offset_x
        self.offset_z       = world_offset_z
        self.n_rows, self.n_cols = heightmap.shape

        # Eğim → maksimum geçilebilir dH (metre)
        self._max_dh_ortho = math.tan(math.radians(max_slope)) * cell_size
        self._max_dh_diag  = math.tan(math.radians(max_slope)) * cell_size * math.sqrt(2)

        # Aktif rota cache (grid hücre listesi) ve pointer
        self._path:     list[tuple[int, int]] = []
        self._path_idx: int   = 0
        self._goal_m:   Optional[tuple[float, float]] = None

        print(f"[Pathfinder] Grid:{self.n_rows}×{self.n_cols} | "
              f"Cell:{cell_size}m | MaxSlope:{max_slope}° | "
              f"Offset:({world_offset_x},{world_offset_z})")

    # ══════════════════════════════════════════
    #  KOORDİNAT DÖNÜŞÜM YARDIMCILARI
    # ══════════════════════════════════════════

    def _meter_to_cell(self, x_m: float, z_m: float) -> tuple[int, int]:
        """
        Webots dünya koordinatı (metre) → grid (row, col).
          col ← X ekseni (Webots'ta doğu yönü)
          row ← Z ekseni (Webots'ta güney yönü)
        Sınır kırpma uygulanır.
        """
        col = int((x_m - self.offset_x) / self.cell_size)
        row = int((z_m - self.offset_z) / self.cell_size)
        row = max(0, min(self.n_rows - 1, row))
        col = max(0, min(self.n_cols - 1, col))
        return row, col

    def _cell_to_meter(self, row: int, col: int) -> tuple[float, float]:
        """
        Grid (row, col) → Webots dünya koordinatı (x_m, z_m).
        +0.5 ofseti hücrenin merkezini verir.
        """
        x_m = (col + 0.5) * self.cell_size + self.offset_x
        z_m = (row + 0.5) * self.cell_size + self.offset_z
        return x_m, z_m

    # ══════════════════════════════════════════
    #  A* ÇEKİRDEĞİ — İÇ YARDIMCILAR
    # ══════════════════════════════════════════

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _slope_cost(self, r1: int, c1: int,
                    r2: int, c2: int,
                    base_cost: float) -> Optional[float]:
        """
        İki hücre arasındaki eğim maliyetini hesaplar.
        Eğim eşiği aşılırsa None döndürür (geçilemez duvar).
        """
        dh     = abs(float(self.heightmap[r2, c2]) - float(self.heightmap[r1, c1]))
        max_dh = self._max_dh_diag if base_cost > 1.0 else self._max_dh_ortho
        if dh > max_dh:
            return None
        slope_deg = math.degrees(math.atan2(dh, self.cell_size * base_cost))
        penalty   = (slope_deg / self.max_slope_deg) ** 2 * self.slope_penalty
        return base_cost + penalty

    def _heuristic(self, r: int, c: int, gr: int, gc: int) -> float:
        """Ağırlıklı Öklid sezgiseli."""
        return self.heuristic_w * math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

    @staticmethod
    def _reconstruct(node: _Node) -> list[tuple[int, int]]:
        """Geri iz sürerek tam yolu döndürür."""
        path, cur = [], node
        while cur is not None:
            path.append(cur.pos)
            cur = cur.parent
        return list(reversed(path))

    # ══════════════════════════════════════════
    #  GRID KOORDİNATLI ANA ROTA PLANLAMA
    #  (ORIJINAL — KORUNDU, HİÇBİR SATIR SİLİNMEDİ)
    # ══════════════════════════════════════════

    def find_path(self,
                  start: tuple[int, int],
                  goal:  tuple[int, int]) -> Optional[list[tuple[int, int]]]:
        """
        Başlangıçtan hedefe A* ile en güvenli yolu bulur.

        Args:
            start: (row, col) başlangıç hücre koordinatı.
            goal:  (row, col) hedef hücre koordinatı.

        Returns:
            [(r0,c0), (r1,c1), ...] koordinat listesi veya None.
        """
        sr, sc = start
        gr, gc = goal

        for label, (r, c) in [("Başlangıç", (sr, sc)), ("Hedef", (gr, gc))]:
            if not self._in_bounds(r, c):
                raise ValueError(f"{label} ({r},{c}) harita sınırları dışında!")

        open_heap: list[_Node]       = []
        g_costs:   dict[tuple, float] = {start: 0.0}
        visited:   set[tuple]         = set()

        heapq.heappush(open_heap, _Node(
            f_cost=self._heuristic(sr, sc, gr, gc),
            g_cost=0.0, pos=start
        ))

        iterations = 0
        while open_heap:
            current    = heapq.heappop(open_heap)
            cr, cc     = current.pos
            iterations += 1

            if current.pos in visited:
                continue
            visited.add(current.pos)

            if current.pos == goal:
                path   = self._reconstruct(current)
                dist_m = (len(path) - 1) * self.cell_size
                print(f"[Pathfinder] Rota bulundu ✓ | "
                      f"{len(path)} adım | ~{dist_m:.0f} m | "
                      f"{iterations} iterasyon")
                return path

            for dr, dc, base_move_cost in _DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if not self._in_bounds(nr, nc) or (nr, nc) in visited:
                    continue
                move_cost = self._slope_cost(cr, cc, nr, nc, base_move_cost)
                if move_cost is None:
                    continue
                new_g = current.g_cost + move_cost
                if new_g < g_costs.get((nr, nc), math.inf):
                    g_costs[(nr, nc)] = new_g
                    heapq.heappush(open_heap, _Node(
                        f_cost=new_g + self._heuristic(nr, nc, gr, gc),
                        g_cost=new_g, pos=(nr, nc), parent=current
                    ))

        print(f"[Pathfinder] ⚠ Rota bulunamadı! ({iterations} iterasyon)")
        return None

    # ══════════════════════════════════════════
    #  WEBOTS ARAYÜZİ — YENİ EKLENEN METODLAR
    # ══════════════════════════════════════════

    def plan(self,
             start_m: tuple[float, float],
             goal_m:  tuple[float, float]) -> bool:
        """
        Metre koordinatlarıyla yeni bir rota planlar; dahili path cache'e yazar.

        Webots kontrolcüsü simülasyon başında (ve LIDAR override sonrası) çağırır.

        Args:
            start_m: (x, z) Webots GPS koordinatı — başlangıç.
            goal_m:  (x, z) Webots GPS koordinatı — hedef.

        Returns:
            True: rota bulundu ve cache'e yazıldı.
            False: ulaşılamaz (harita engeli veya grid dışı).
        """
        self._goal_m   = goal_m
        start_cell     = self._meter_to_cell(*start_m)
        goal_cell      = self._meter_to_cell(*goal_m)

        path = self.find_path(start_cell, goal_cell)
        if path:
            self._path     = path
            self._path_idx = 0
            return True

        self._path = []
        return False

    def get_next_move(self,
                      current_pos_m:    tuple[float, float],
                      arrival_radius_m: float = ARRIVAL_RADIUS_M
                      ) -> Optional[tuple[float, float]]:
        """
        Webots kontrolcüsünün her timestep'te çağırdığı navigasyon arayüzü.

        GPS konumunu alır → bir sonraki hedef waypoint'i metre olarak döndürür.

        Mantık:
          1. GPS → grid hücresi dönüşümü.
          2. Zaten geçilmiş waypoint'leri atla (arrival_radius kontrolü).
          3. Sonraki waypoint'in metre koordinatını döndür.
          4. Tüm path tükendi → None (hedefe ulaşıldı).

        Args:
            current_pos_m:    (gps_x, gps_z) rover'ın anlık Webots koordinatı.
            arrival_radius_m: Bu mesafe içine girince waypoint "geçildi" sayılır.

        Returns:
            (target_x_m, target_z_m) — bir sonraki waypoint (metre).
            None                     — hedefe ulaşıldı veya rota yok.
        """
        if not self._path:
            return None

        # Zaten geçilmiş waypoint'leri ilerle
        while self._path_idx < len(self._path) - 1:
            wp_x, wp_z = self._cell_to_meter(*self._path[self._path_idx])
            dist = math.sqrt((current_pos_m[0] - wp_x) ** 2 +
                             (current_pos_m[1] - wp_z) ** 2)
            if dist <= arrival_radius_m:
                self._path_idx += 1
            else:
                break

        if self._path_idx >= len(self._path):
            return None

        target_m = self._cell_to_meter(*self._path[self._path_idx])

        # Son waypoint'e yaklaştıysak None döndür (görev tamam)
        if self._path_idx == len(self._path) - 1:
            dist = math.sqrt((current_pos_m[0] - target_m[0]) ** 2 +
                             (current_pos_m[1] - target_m[1]) ** 2)
            if dist <= arrival_radius_m:
                return None

        return target_m

    def replan_from(self,
                    current_pos_m: tuple[float, float],
                    goal_m: Optional[tuple[float, float]] = None) -> bool:
        """
        LIDAR override veya SLAM güncellmesi sonrası acil yeniden planlama.

        Args:
            current_pos_m: Rover'ın mevcut GPS konumu (metre).
            goal_m:        Yeni hedef. None → önceki hedef kullanılır.

        Returns:
            True: yeni rota bulundu.
        """
        target = goal_m if goal_m is not None else self._goal_m
        if target is None:
            print("[Pathfinder] replan_from: hedef tanımlı değil!")
            return False
        return self.plan(current_pos_m, target)

    # ══════════════════════════════════════════
    #  İSTATİSTİK (ORIJINAL — KORUNDU)
    # ══════════════════════════════════════════

    def path_statistics(self, path: list[tuple[int, int]]) -> dict:
        """
        Bulunan rota için istatistik sözlüğü döndürür.
        Görsel raporlama veya loglama için kullanılabilir.
        """
        if not path or len(path) < 2:
            return {}

        heights    = [float(self.heightmap[r, c]) for r, c in path]
        elevations = np.diff(heights)
        slopes_deg = [
            math.degrees(math.atan2(abs(dh), self.cell_size))
            for dh in elevations
        ]

        return {
            "step_count":      len(path),
            "distance_m":      (len(path) - 1) * self.cell_size,
            "start_height_m":  heights[0],
            "end_height_m":    heights[-1],
            "net_elevation_m": heights[-1] - heights[0],
            "max_slope_deg":   max(slopes_deg),
            "avg_slope_deg":   sum(slopes_deg) / len(slopes_deg),
            "climb_m":         sum(max(0,  dh) for dh in elevations),
            "descent_m":       sum(abs(min(0, dh)) for dh in elevations),
        }

    def current_path_stats(self) -> dict:
        """Aktif rotanın istatistiklerini döndürür (plan() sonrası)."""
        return self.path_statistics(self._path) if self._path else {}