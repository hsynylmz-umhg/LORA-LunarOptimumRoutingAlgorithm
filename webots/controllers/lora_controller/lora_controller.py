# filepath: LORA/webots/controllers/lora_controller/lora_controller.py
"""
L.O.R.A. — Lunar Optimum Routing Algorithm  |  v1.0 HACKATHON FINAL
════════════════════════════════════════════════════════════════════
  Düzeltmeler (v1.0):
    FIX-4a  Rota bulunamazsa araç KESINLIKLE DURUR → "ROTA YOK" logu
    FIX-4b  Her replan sonrası path listesi UDP TYPE_PATH ile yayınlanır
            → lora_master.py UI'ında mavi çizgi güncellenir
    (FIX-1/2/3 master tarafında)
════════════════════════════════════════════════════════════════════
"""

from controller import Robot

import sys
import math
import time
import socket
import struct
import numpy as np
from pathlib import Path

_CTRL_DIR = Path(__file__).resolve().parent
_SRC_DIR  = _CTRL_DIR.parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR / "algorithm"))

from pathfinder  import Pathfinder
from slam_mapper import SlamMapper

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRELERİ BURADAN DEĞİŞTİR
# ══════════════════════════════════════════════════════════════════════════════

TIME_STEP = 32

GRID_SIZE   = 500
WORLD_SIZE  = 500
CELL_SIZE_M = 1.0
OFFSET_X    = -WORLD_SIZE / 2.0
OFFSET_Z    = -WORLD_SIZE / 2.0

_BASE_DIR     = _CTRL_DIR.parent.parent.parent
_MISSION_FILE = _BASE_DIR / "data" / "processed" / "mission_params.txt"
LOW_MAP_PATH  = _BASE_DIR / "data" / "raw" / "low_detail_map.csv"
HIGH_MAP_PATH = _BASE_DIR / "data" / "raw" / "high_detail_map.csv"

DEFAULT_START = (-220.0, -220.0)
DEFAULT_GOAL  = ( 220.0,  220.0)

MOTOR_FL = "front left wheel"
MOTOR_FR = "front right wheel"
MOTOR_BL = "back left wheel"
MOTOR_BR = "back right wheel"

MAX_SPEED    = 6.28
CRUISE_SPEED = 4.0
TURN_SPEED   = 2.0

KP = 0.045
KD = 0.008
MAX_STEER = MAX_SPEED * 0.85

# Dinamik eğim fiziği
PITCH_DOWN   = -8.0    # dereceyi geç → hızı kıs
PITCH_UP     =  8.0    # dereceyi geç → hızı artır
SCALE_DOWN   = 0.42    # iniş çarpanı
SCALE_UP     = 1.18    # çıkış çarpanı (MAX_SPEED ile sınırlı)

# LIDAR
LIDAR_NAME  = "lidar"
LIDAR_MAX   = 30.0
LIDAR_EMRG  = 1.2     # acil dur
LIDAR_AVOD  = 4.0     # yavaşla
LIDAR_CLR   = 8.0     # temiz
LIDAR_ARC   = 60.0
LIDAR_TURN  = 35.0
LIDAR_OVR_S = 2.8

# UDP
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005

TYPE_POS  = 1
TYPE_OBS  = 2
TYPE_DON  = 3
TYPE_PATH = 4    # FIX-4b: rota listesi

GOAL_R    = 5.0
WP_R      = 2.5
LOG_N     = 40


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def _read_mission():
    if not _MISSION_FILE.exists():
        return DEFAULT_START, DEFAULT_GOAL, UDP_HOST, UDP_PORT
    cfg = {}
    for ln in _MISSION_FILE.read_text().splitlines():
        if "=" in ln:
            k, v = ln.strip().split("=", 1)
            cfg[k.strip()] = v.strip()
    s = (float(cfg.get("START_X", DEFAULT_START[0])),
         float(cfg.get("START_Z", DEFAULT_START[1])))
    g = (float(cfg.get("GOAL_X",  DEFAULT_GOAL[0])),
         float(cfg.get("GOAL_Z",  DEFAULT_GOAL[1])))
    return s, g, cfg.get("UDP_HOST", UDP_HOST), int(cfg.get("UDP_PORT", UDP_PORT))


def compass_hdg(north):
    return math.degrees(math.atan2(north[0], -north[2])) % 360.0

def adiff(tgt, cur):
    return ((tgt - cur + 180.0) % 360.0) - 180.0

def bearing(pos, tgt):
    return math.degrees(math.atan2(tgt[0]-pos[0], -(tgt[1]-pos[1]))) % 360.0

def gps_cell(x, z):
    c = max(0, min(GRID_SIZE-1, int((x - OFFSET_X) / CELL_SIZE_M)))
    r = max(0, min(GRID_SIZE-1, int((z - OFFSET_Z) / CELL_SIZE_M)))
    return r, c


# ══════════════════════════════════════════════════════════════════════════════
#  UDP GÖNDERİCİ
# ══════════════════════════════════════════════════════════════════════════════

class UDPSender:
    def __init__(self, host, port):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._a = (host, port)

    def pos(self, x, z, spd, hdg):
        try:
            self._s.sendto(struct.pack("<Bffff", TYPE_POS, x, z, spd, hdg), self._a)
        except OSError: pass

    def obs(self, ox, oz, h):
        try:
            self._s.sendto(struct.pack("<Bfff", TYPE_OBS, ox, oz, h), self._a)
        except OSError: pass

    def done(self):
        try:
            self._s.sendto(struct.pack("<B", TYPE_DON), self._a)
        except OSError: pass

    def path(self, waypoints_m: list):
        """
        FIX-4b ▸ A* rota listesini UDP ile master'a gönderir.
        Format: uint8(4) | uint16(count) | [float32 x, float32 z] × count
        500 waypoint sınırı — daha uzun rotalar kırpılır.
        """
        pts = waypoints_m[:500]
        n   = len(pts)
        buf = struct.pack("<BH", TYPE_PATH, n)
        for px, pz in pts:
            buf += struct.pack("<ff", px, pz)
        try:
            self._s.sendto(buf, self._a)
        except OSError: pass

    def close(self):
        self._s.close()


# ══════════════════════════════════════════════════════════════════════════════
#  LIDAR İŞLEMCİ
# ══════════════════════════════════════════════════════════════════════════════

class LidarProc:
    def __init__(self, dev, fov_deg, nrays):
        self.dev    = dev
        self.angles = [
            math.radians(-fov_deg/2 + fov_deg*i/max(nrays-1,1))
            for i in range(nrays)
        ]

    def analyze(self):
        ranges   = self.dev.getRangeImage()
        half_arc = math.radians(LIDAR_ARC)
        L, R     = [], []
        for i, a in enumerate(self.angles):
            if abs(a) > half_arc: continue
            d = (ranges[i] if ranges else LIDAR_MAX)
            if d <= 0 or d > LIDAR_MAX: d = LIDAR_MAX
            (L if a < 0 else R).append(d)
        lm = min(L) if L else LIDAR_MAX
        rm = min(R) if R else LIDAR_MAX
        mn = min(lm, rm)
        side = "clear"
        if mn < LIDAR_AVOD:
            side = "left" if lm < rm else ("right" if rm < lm else "center")
        return {"min": mn, "left": lm, "right": rm,
                "side": side, "blocked": mn < LIDAR_EMRG}

    def slam_pts(self, gx, gz, hdg, hy):
        ranges   = self.dev.getRangeImage()
        half_arc = math.radians(LIDAR_ARC)
        out = []
        for i, rel in enumerate(self.angles):
            if abs(rel) > half_arc: continue
            d = (ranges[i] if ranges else LIDAR_MAX)
            if d <= 0 or d >= LIDAR_MAX * 0.95: continue
            aa  = math.radians(hdg) + rel
            ox  = gx + d * math.sin(aa)
            oz  = gz - d * math.cos(aa)
            r,c = gps_cell(ox, oz)
            out.append((r, c, hy + 0.5))
        return out

    def obs_gps(self, gx, gz, hdg):
        ranges   = self.dev.getRangeImage()
        half_arc = math.radians(LIDAR_ARC)
        out = []
        for i, rel in enumerate(self.angles):
            if abs(rel) > half_arc: continue
            d = (ranges[i] if ranges else LIDAR_MAX)
            if d <= 0 or d >= LIDAR_AVOD: continue
            aa = math.radians(hdg) + rel
            out.append((gx + d * math.sin(aa), gz - d * math.cos(aa)))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  ANA KONTROLCÜ
# ══════════════════════════════════════════════════════════════════════════════

class LoraController(Robot):

    def __init__(self):
        super().__init__()
        print("=" * 62)
        print("  L.O.R.A. v1.0 — Kontrolcü Başlıyor")
        print("=" * 62)

        self._ts       = TIME_STEP
        self._step     = 0
        self._done     = False
        self._t0       = time.time()
        self._prev_err = 0.0

        # FIX-4a ▸ rota yok sayacı ve durum bayrağı
        self._no_path  = False       # True iken araç durur
        self._no_path_logged = False

        # LIDAR override
        self._ovr      = False
        self._ovr_t    = 0.0
        self._ovr_dir  = 1.0
        self._last_spd = (0.0, 0.0)

        start_m, goal_m, udp_h, udp_p = _read_mission()
        self._goal_m = goal_m
        self._udp    = UDPSender(udp_h, udp_p)
        print(f"  Start: {start_m}  →  Goal: {goal_m}")

        self._init_hw()
        self._init_lora(start_m, goal_m)
        print("\n✅ Hazır.\n")

    # ── Donanım ───────────────────────────────────────────────────────────────

    def _init_hw(self):
        print("[1/3] Donanım...")

        self._gps = self.getDevice("gps"); self._gps.enable(TIME_STEP)
        self._cmp = self.getDevice("compass"); self._cmp.enable(TIME_STEP)

        self._imu = self.getDevice("inertial unit")
        if self._imu:
            self._imu.enable(TIME_STEP); self._has_imu = True
            print("  IMU: ✓")
        else:
            self._has_imu = False
            print("  IMU: yok — eğim fiziği devre dışı")

        ld = self.getDevice(LIDAR_NAME)
        if ld:
            ld.enable(TIME_STEP); ld.enablePointCloud()
            self._lidar    = LidarProc(ld, math.degrees(ld.getFov()),
                                        ld.getNumberOfPoints())
            self._has_lidar = True
            print(f"  LIDAR: FOV={math.degrees(ld.getFov()):.0f}° ✓")
        else:
            self._has_lidar = False
            print("  LIDAR: yok")

        self._fl = self.getDevice(MOTOR_FL)
        self._fr = self.getDevice(MOTOR_FR)
        self._bl = self.getDevice(MOTOR_BL)
        self._br = self.getDevice(MOTOR_BR)
        for m in (self._fl, self._fr, self._bl, self._br):
            m.setPosition(float("inf")); m.setVelocity(0.0)
        print("  Motorlar: ✓")

    # ── L.O.R.A. ──────────────────────────────────────────────────────────────

    def _init_lora(self, start_m, goal_m):
        print("[2/3] Haritalar...")

        def ld(p):
            if not p.exists():
                raise FileNotFoundError(f"{p} yok — generate_terrain çalıştır.")
            d = np.loadtxt(str(p), delimiter=" ", dtype=np.float64)
            print(f"  {p.name}: {d.shape}")
            return d

        low = ld(LOW_MAP_PATH)
        ld(HIGH_MAP_PATH)   # varlık kontrolü

        print("[3/3] Pathfinder + SlamMapper...")
        self._mapper = SlamMapper(LOW_MAP_PATH)
        self._pf     = Pathfinder(low, cell_size=CELL_SIZE_M,
                                   world_offset_x=OFFSET_X,
                                   world_offset_z=OFFSET_Z)
        self._plan(start_m)

    def _plan(self, current_m=None):
        """
        FIX-4a ▸ Rota bulunamazsa self._no_path = True.
        FIX-4b ▸ Rota bulununca UDP TYPE_PATH paketi gönderilir.
        """
        if current_m is None:
            v = self._gps.getValues()
            current_m = (v[0], v[2])

        ok = self._pf.plan(current_m, self._goal_m)

        if not ok:
            self._no_path = True
            print("[Pathfinder] ⚠ ROTA YOK — araç durdu.")
            return False

        self._no_path        = False
        self._no_path_logged = False

        # FIX-4b ▸ Bulunan rotayı UDP ile gönder
        if self._pf._path:
            wpts_m = [self._pf._cell_to_meter(r, c) for r, c in self._pf._path]
            self._udp.path(wpts_m)
            s = self._pf.current_path_stats()
            print(f"  [Plan] {s.get('step_count','?')} adım | "
                  f"~{s.get('distance_m',0):.0f} m | "
                  f"MaxEğim:{s.get('max_slope_deg',0):.1f}°")
        return True

    # ── Motor ─────────────────────────────────────────────────────────────────

    def _set(self, L, R):
        L = max(-MAX_SPEED, min(MAX_SPEED, L))
        R = max(-MAX_SPEED, min(MAX_SPEED, R))
        self._fl.setVelocity(L); self._bl.setVelocity(L)
        self._fr.setVelocity(R); self._br.setVelocity(R)
        self._last_spd = (L, R)

    def _stop(self):
        self._set(0.0, 0.0)

    # ── Pitch / Eğim ──────────────────────────────────────────────────────────

    def _pitch_scale(self):
        if not self._has_imu: return 1.0
        pitch = math.degrees(self._imu.getRollPitchYaw()[1])
        if pitch < PITCH_DOWN:
            t = min(1.0, (PITCH_DOWN - pitch) / 15.0)
            return SCALE_DOWN + (1.0 - SCALE_DOWN) * (1.0 - t)
        if pitch > PITCH_UP:
            return min(SCALE_UP, 1.0 + 0.15 * min(1.0, (pitch - PITCH_UP) / 15.0))
        return 1.0

    def _est_speed(self):
        return (abs(self._last_spd[0]) + abs(self._last_spd[1])) / 2.0 * 0.1975

    # ── PD Sürüş ─────────────────────────────────────────────────────────────

    def _steer(self, gx, gz, hdg, tgt, scale=1.0):
        des   = bearing((gx, gz), tgt)
        err   = adiff(des, hdg)
        d_err = err - self._prev_err
        st    = max(-MAX_STEER, min(MAX_STEER, KP * err + KD * d_err))
        self._prev_err = err
        base  = CRUISE_SPEED * scale
        self._set(base - st, base + st)

    # ── LIDAR Override ────────────────────────────────────────────────────────

    def _lidar_override(self, info, hdg, gx, gz, gy):
        dt = TIME_STEP / 1000.0

        if self._ovr:
            self._ovr_t -= dt
            if self._ovr_t <= 0:
                self._ovr = False
                self._plan()
                print("[LIDAR] Override bitti.")
                return False
            esc = hdg + self._ovr_dir * LIDAR_TURN
            err = adiff(esc, hdg)
            st  = max(-MAX_STEER, min(MAX_STEER, KP * err))
            self._set(TURN_SPEED - st, TURN_SPEED + st)
            return True

        mn, side = info["min"], info["side"]

        if mn < LIDAR_EMRG:
            self._stop()
            print(f"[LIDAR] 🛑 {mn:.2f}m — override!")
            for ox, oz in self._lidar.obs_gps(gx, gz, hdg):
                self._udp.obs(ox, oz, gy + 0.5)
            self._ovr     = True
            self._ovr_t   = LIDAR_OVR_S
            self._ovr_dir = -1.0 if side == "left" else 1.0
            return True

        if mn < LIDAR_AVOD:
            scale  = max(0.25, (mn - LIDAR_EMRG) / (LIDAR_AVOD - LIDAR_EMRG))
            bias   = -1.0 if side == "left" else 1.0
            st_add = bias * TURN_SPEED * (1.0 - scale)
            if self._step % 10 == 0:
                for ox, oz in self._lidar.obs_gps(gx, gz, hdg):
                    self._udp.obs(ox, oz, gy + 0.5)
            self._set(CRUISE_SPEED * scale - st_add,
                      CRUISE_SPEED * scale + st_add)
            return True

        return False

    # ── Ana Döngü ─────────────────────────────────────────────────────────────

    def run(self):
        while self.step(self._ts) != -1:

            if self._done:
                self._stop(); continue

            self._step += 1
            v   = self._gps.getValues()
            gx, gy, gz = v[0], v[1], v[2]
            hdg = compass_hdg(self._cmp.getValues())
            ps  = self._pitch_scale()

            # Varış
            d_goal = math.sqrt((gx - self._goal_m[0])**2 +
                               (gz - self._goal_m[1])**2)
            if d_goal <= GOAL_R:
                self._stop(); self._done = True
                self._udp.done()
                self._mapper.save_learned_map()
                self._print_final(); continue

            # ── FIX-4a: ROTA YOK → KESINLIKLE DUR ───────────────────────────
            if self._no_path:
                self._stop()
                if not self._no_path_logged:
                    print("[Ctrl] ⛔ ROTA YOK - BEKLENİYOR  "
                          "(Pathfinder engeli aşamıyor)")
                    self._no_path_logged = True
                # Her 200 adımda bir yeniden dene
                if self._step % 200 == 0:
                    self._plan((gx, gz))
                continue
            # ─────────────────────────────────────────────────────────────────

            # LIDAR
            override = False
            info     = {"min": LIDAR_MAX, "side": "clear", "blocked": False}
            if self._has_lidar:
                info = self._lidar.analyze()
                if info["min"] < LIDAR_CLR:
                    pts = self._lidar.slam_pts(gx, gz, hdg, gy)
                    if pts:
                        self._mapper.update_map_region(pts)
                    if self._step % 25 == 0:
                        self._plan((gx, gz))
                override = self._lidar_override(info, hdg, gx, gz, gy)

            # A* Navigasyon
            if not override:
                tgt = self._pf.get_next_move((gx, gz), WP_R)
                if tgt is None:
                    # FIX-4a: yeniden plan dene
                    ok = self._plan((gx, gz))
                    if not ok:
                        # _no_path = True → bir sonraki iterasyonda durulur
                        pass
                    continue
                self._steer(gx, gz, hdg, tgt, ps)

            # UDP
            if self._step % 3 == 0:
                self._udp.pos(gx, gz, self._est_speed(), hdg)

            # Log
            if self._step % LOG_N == 0:
                t = time.time() - self._t0
                r, c = gps_cell(gx, gz)
                print(
                    f"[{t:6.1f}s] "
                    f"({gx:+7.1f},{gz:+7.1f}) | "
                    f"({r:3d},{c:3d}) | "
                    f"H:{hdg:5.1f}° | "
                    f"Goal:{d_goal:6.1f}m | "
                    f"L:{info['min']:4.1f}m | "
                    f"P:{ps:.2f}x | "
                    f"OVR:{'Y' if override else 'N'} | "
                    f"NoPath:{'Y' if self._no_path else 'N'}"
                )

    # ── Final ─────────────────────────────────────────────────────────────────

    def _print_final(self):
        e = time.time() - self._t0
        s = self._mapper.get_update_stats()
        print("\n" + "=" * 62)
        print("  🏁  GÖREV TAMAMLANDI!")
        print(f"  Süre      : {e:.1f}s  |  Adım: {self._step:,}")
        print(f"  SLAM güncelleme: {s['total_updates']}  |  Öğrenilen hücre: {s['dirty_cell_count']}")
        ps = self._pf.current_path_stats()
        if ps:
            print(f"  Rota: ~{ps['distance_m']:.0f} m")
        print("=" * 62)
        self._udp.close()


if __name__ == "__main__":
    LoraController().run()