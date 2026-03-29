# filepath: LORA/lora_master.py
"""
L.O.R.A. — Lunar Optimum Routing Algorithm  |  v1.0 HACKATHON FINAL
════════════════════════════════════════════════════════════════════
  python lora_master.py                     # JAXA haritasından rastgele
  python lora_master.py --img SLDEM*.IMG    # Belirli dosya
  python lora_master.py --no-webots         # Sadece UI testi

  Düzeltmeler (v1.0):
    FIX-1  JAXA yükseklik skalası → patch * HEIGHT_SCALE (0.40)
    FIX-2  WBT aydınlatma → AmbientLight + yüksek ambientIntensity
    FIX-3  Start/Goal eşiği → MAX_SLOPE_DEG = 28° + garantili rastgele fallback
    FIX-4  2D UI → A* rotası mavi çizgi olarak çiziliyor (UDP path listesi)
════════════════════════════════════════════════════════════════════
"""

import os
import sys
import math
import time
import socket
import struct
import random
import argparse
import subprocess
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")          # sorun çıkarsa "Qt5Agg" deneyin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter

# ── L.O.R.A. algoritma modülleri (pathfinder sadece UI path için) ─────────────
_BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE_DIR / "src" / "algorithm"))
try:
    from pathfinder import Pathfinder
    _PF_AVAILABLE = True
except ImportError:
    _PF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 1 — KONFİGÜRASYON  (tüm ayarlar burada)
# ══════════════════════════════════════════════════════════════════════════════

# ── Dizinler ──────────────────────────────────────────────────────────────────
DATA_RAW   = _BASE_DIR / "data" / "raw"
DATA_PROC  = _BASE_DIR / "data" / "processed"
WBT_PATH   = _BASE_DIR / "webots" / "worlds" / "lunar_surface.wbt"

# ── Harita ────────────────────────────────────────────────────────────────────
PATCH_SIZE    = 500       # 500×500 hücre
CELL_SIZE_M   = 1.0       # metre/hücre → toplam 500×500 m
WORLD_M       = PATCH_SIZE * CELL_SIZE_M

# FIX-1 ▸ JAXA ham verisi çok sert → yüksekliği bu katsayıyla ölçekle
# 0.40 = tepe/çukur yüksekliklerini %60 baskıla; artırırsan daha sarp olur
HEIGHT_SCALE  = 0.40

# PDS IMG boyutları (SLDEM2015_256 serisi)
PDS_LINES     = 7680
PDS_SAMPLES   = 30720

# ── Görev ─────────────────────────────────────────────────────────────────────
MIN_DIST_M    = 300       # Start-Goal arası min mesafe
MAX_DIST_M    = 350       # Start-Goal arası max mesafe

# FIX-3 ▸ JAXA haritası engebeli → eşiği 28°'ye çektik
MAX_SLOPE_DEG = 28.0

# ── UDP ───────────────────────────────────────────────────────────────────────
UDP_HOST    = "127.0.0.1"
UDP_PORT    = 5005
UDP_BUFSIZE = 2048        # path listesi için büyütüldü

# ── UI ────────────────────────────────────────────────────────────────────────
UI_UPDATE_MS = 120
TRAIL_MAX    = 3000
OBSTACLE_CLR = "#FF8C00"
PATH_CLR     = "#4488ff"  # A* rotası rengi (açık mavi)

# ── Webots ────────────────────────────────────────────────────────────────────
WEBOTS_EXE = r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
# Linux/Mac → WEBOTS_EXE = "webots"

# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 2 — HARİTA ÜRETİMİ
# ══════════════════════════════════════════════════════════════════════════════

def _list_img_files() -> list:
    return list(DATA_RAW.glob("*.IMG")) + list(DATA_RAW.glob("*.img"))


def _slope_map(hm: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(hm, CELL_SIZE_M)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))


def _load_patch(img_path: Path, sr: int, sc: int) -> np.ndarray:
    raw   = np.memmap(str(img_path), dtype="float32", mode="r",
                      shape=(PDS_LINES, PDS_SAMPLES))
    patch = np.array(raw[sr:sr + PATCH_SIZE, sc:sc + PATCH_SIZE],
                     dtype=np.float64)
    return patch


def _find_flat_patch(img_path: Path, max_attempts: int = 25):
    """
    Eğim yüzdesi en düşük 500×500 kesiti bulur.
    FIX-3: eşiği esnetildi (MAX_SLOPE_DEG = 28°).
    """
    margin  = 50
    best    = None
    best_p  = 999.0

    for _ in range(max_attempts):
        sr = random.randint(margin, PDS_LINES  - PATCH_SIZE - margin)
        sc = random.randint(margin, PDS_SAMPLES - PATCH_SIZE - margin)
        p  = _load_patch(img_path, sr, sc)

        if np.isnan(p).any() or p.max() == p.min():
            continue

        # FIX-1 ▸ ölçekleme burada uygulanır
        p = p * HEIGHT_SCALE

        slopes = _slope_map(p)
        p85    = float(np.percentile(slopes, 85))

        if p85 < best_p:
            best_p       = p85
            best         = (p, sr, sc)

        if p85 <= MAX_SLOPE_DEG:
            print(f"  [Harita] Kesit bulundu: satır={sr}, sütun={sc}, "
                  f"p85_eğim={p85:.1f}°")
            return p, sr, sc

    # Tüm denemeler başarısız → en iyi bulunanı döndür
    if best:
        p, sr, sc = best
        print(f"  [Harita] ⚠ Eşik tutmadı — en düşük eğimli kesit: "
              f"satır={sr}, sütun={sc}, p85={best_p:.1f}°")
        return p, sr, sc

    # Tamamen boş → sentetik
    print("  [Harita] ⚠ Hiç uygun kesit yok → sentetik üretiliyor.")
    return _generate_synthetic(), 0, 0


def _generate_synthetic() -> np.ndarray:
    """IMG yoksa yedek sentetik harita (Perlin veya Gaussian)."""
    try:
        from noise import pnoise2
        N = PATCH_SIZE
        f = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                f[i, j] = pnoise2(i / 150.0, j / 150.0, octaves=3)
        f = (f - f.min()) / (f.max() - f.min()) * (20.0 * HEIGHT_SCALE)
    except ImportError:
        rng = np.random.default_rng(42)
        f   = gaussian_filter(rng.normal(0, 3.0, (PATCH_SIZE, PATCH_SIZE)), 20)
        f   = f * HEIGHT_SCALE
    return f


def generate_heightmap(img_path=None) -> np.ndarray:
    """
    IMG dosyasından ölçeklenmiş heightmap üretir;
    hem high_detail hem low_detail CSV'lerini kaydeder.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    img_files = _list_img_files()

    if not img_files:
        print("  [Harita] IMG bulunamadı → sentetik.")
        patch = _generate_synthetic()
    else:
        chosen = img_path if img_path else random.choice(img_files)
        print(f"  [Harita] Seçilen: {chosen.name}")
        patch, _, _ = _find_flat_patch(chosen)

    np.savetxt(str(DATA_RAW / "high_detail_map.csv"), patch,
               fmt="%.4f", delimiter=" ")
    np.savetxt(str(DATA_RAW / "low_detail_map.csv"),
               gaussian_filter(patch, sigma=8.0),
               fmt="%.4f", delimiter=" ")

    print(f"  [Harita] Yük. aralığı: {patch.min():.1f}–{patch.max():.1f} m")
    return patch


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 3 — GÖREV NOKTASI SEÇİMİ
# ══════════════════════════════════════════════════════════════════════════════

def _local_slope_ok(hm: np.ndarray, r: int, c: int, radius: int = 12) -> bool:
    r0, r1 = max(0, r - radius), min(hm.shape[0], r + radius)
    c0, c1 = max(0, c - radius), min(hm.shape[1], c + radius)
    return float(np.mean(_slope_map(hm[r0:r1, c0:c1]))) <= MAX_SLOPE_DEG


def pick_start_goal(hm: np.ndarray, max_tries: int = 800):
    """
    FIX-3 ▸ Eşik 28°; 800 deneme; başarısız olursa haritanın
    en düz dört bölgesinden biri seçilir — hep köşe değil.
    """
    N      = hm.shape[0]
    margin = 25
    min_c  = int(MIN_DIST_M / CELL_SIZE_M)
    max_c  = int(MAX_DIST_M / CELL_SIZE_M)

    for _ in range(max_tries):
        sr = random.randint(margin, N - margin)
        sc = random.randint(margin, N - margin)
        if not _local_slope_ok(hm, sr, sc):
            continue

        angle  = random.uniform(0, 2 * math.pi)
        dist_c = random.randint(min_c, max_c)
        gr     = int(sr + dist_c * math.cos(angle))
        gc     = int(sc + dist_c * math.sin(angle))

        if not (margin <= gr < N - margin and margin <= gc < N - margin):
            continue
        if not _local_slope_ok(hm, gr, gc):
            continue

        d = math.sqrt((gr - sr) ** 2 + (gc - sc) ** 2) * CELL_SIZE_M
        if MIN_DIST_M <= d <= MAX_DIST_M:
            print(f"  [Görev] Start:({sr},{sc})  Goal:({gr},{gc})  {d:.0f}m")
            return (sr, sc), (gr, gc)

    # Garantili fallback: haritayı 4 kadrana böl, her kadranda en düz noktayı bul
    print("  [Görev] ⚠ Rastgele bulunamadı → kadran bazlı seçim yapılıyor.")
    slopes = _slope_map(hm)
    half   = N // 2
    quads  = [
        (0,    0,    half, half),   # sol-üst
        (0,    half, half, N),      # sağ-üst
        (half, 0,    N,    half),   # sol-alt
        (half, half, N,    N),      # sağ-alt
    ]
    def _best_in_quad(r0, c0, r1, c1):
        sub  = slopes[r0:r1, c0:c1]
        flat = np.argmin(sub)
        lr, lc = np.unravel_index(flat, sub.shape)
        return (r0 + lr + margin // 2, c0 + lc + margin // 2)

    random.shuffle(quads)
    sq = _best_in_quad(*quads[0])
    gq = _best_in_quad(*quads[1])
    dist = math.sqrt((gq[0]-sq[0])**2 + (gq[1]-sq[1])**2) * CELL_SIZE_M
    print(f"  [Görev] Kadran Start:{sq}  Goal:{gq}  {dist:.0f}m")
    return sq, gq


def cells_to_gps(row: int, col: int):
    x = (col + 0.5) * CELL_SIZE_M - WORLD_M / 2.0
    z = (row + 0.5) * CELL_SIZE_M - WORLD_M / 2.0
    return x, z


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 4 — WBT OLUŞTURMA
# ══════════════════════════════════════════════════════════════════════════════

def _height_str(hm: np.ndarray, per_line: int = 20) -> str:
    flat  = hm.flatten().tolist()
    lines = [" ".join(f"{v:.4f}" for v in flat[i:i+per_line])
             for i in range(0, len(flat), per_line)]
    return "\n              ".join(lines)


def inject_and_write_wbt(hm, start_gps, goal_gps, start_y, goal_y):
    """
    FIX-2 ▸ Karanlık ekran düzeltildi:
      - AmbientLight düğümü eklendi (genel dolgu ışığı)
      - DirectionalLight.ambientIntensity = 0.85
      - PBRAppearance yerine klasik Appearance + Material kullanıldı
        (PBR bazı Webots sürümlerinde büyük haritada karardı)
      - Material.diffuseColor açık gri, emissiveColor hafif parlak
    """
    n_rows, n_cols = hm.shape
    ox  = -WORLD_M / 2.0
    oz  = -WORLD_M / 2.0
    hs  = _height_str(hm)
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Rover spawn yüksekliği: zemin yüksekliği + küçük offset
    rover_y = start_y + 0.8

    wbt = f"""\
#VRML_SIM R2023b utf8
# L.O.R.A. v1.0 — {ts}
# Start GPS: {start_gps}   Goal GPS: {goal_gps}

WorldInfo {{
  title "L.O.R.A. — Lunar Surface v1.0"
  gravity -1.62
  CFM 0.00001
  ERP 0.6
  fps 60
}}

Viewpoint {{
  follow           "LORA_ROVER"
  followType       "Follow"
  followSmoothness 0.2
  orientation      -0.57 0.0 0.82 1.2
  position         {start_gps[0]:.2f} {start_y + 20:.2f} {start_gps[1] - 25:.2f}
  near             0.3
  far              5000
}}

# FIX-2 ▸ Genel ortam ışığı — ekranı karartmaz
Background {{
  skyColor [ 0.01 0.01 0.02 ]
  luminosity 0.08
}}

# Güneş ışığı — alçak açı, Ay güney kutbu atmosfersiz
DirectionalLight {{
  ambientIntensity 0.85
  color            1.0 0.97 0.90
  direction        -0.45 -0.30 -0.84
  intensity        2.8
  castShadows      FALSE
}}

# Dolgu ışığı — gölge bölgelerini aydınlatır
DirectionalLight {{
  ambientIntensity 0.0
  color            0.55 0.60 0.75
  direction        0.70 -0.20 0.50
  intensity        0.9
  castShadows      FALSE
}}

# FIX-2 ▸ Sabit ortam ışığı düğümü (tüm yüzeyleri dengeli aydınlatır)
PointLight {{
  location    0 500 0
  intensity   0.5
  radius      5000
  castShadows FALSE
}}

DEF LUNAR_SURFACE Solid {{
  translation {ox:.2f} 0 {oz:.2f}
  children [
    Shape {{
      # FIX-2 ▸ PBR yerine klasik Appearance — kararmıyor
      appearance Appearance {{
        material Material {{
          diffuseColor  0.72 0.70 0.67
          specularColor 0.05 0.05 0.05
          emissiveColor 0.06 0.06 0.05
          shininess     0.05
          ambientIntensity 1.0
        }}
      }}
      geometry DEF TERRAIN ElevationGrid {{
        xDimension {n_cols}
        zDimension {n_rows}
        xSpacing   {CELL_SIZE_M}
        zSpacing   {CELL_SIZE_M}
        smooth     TRUE
        height [
              {hs}
        ]
      }}
    }}
  ]
  boundingObject USE TERRAIN
  locked TRUE
}}

# ── Rover ─────────────────────────────────────────────────────────────────────
DEF LORA_ROVER Robot {{
  translation {start_gps[0]:.4f} {rover_y:.4f} {start_gps[1]:.4f}
  rotation    0 1 0 0
  name        "LORA_ROVER"
  controller  "lora_controller"
  children [
    Shape {{
      appearance Appearance {{
        material Material {{
          diffuseColor  0.75 0.58 0.15
          specularColor 0.40 0.35 0.10
          emissiveColor 0.05 0.04 0.01
          shininess     0.4
        }}
      }}
      geometry Box {{ size 0.55 0.28 0.65 }}
    }}
    GPS           {{ name "gps" }}
    Compass       {{ name "compass" }}
    InertialUnit  {{ name "inertial unit" }}
    Lidar {{
      name "lidar"
      translation 0 0.25 0.3
      horizontalResolution 512
      fieldOfView          3.14159
      verticalFieldOfView  0.08
      numberOfLayers       1
      maxRange             30
      noise                0.01
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 0 1 anchor -0.28 0 0.22 }}
      device [ RotationalMotor {{ name "front left wheel"  maxVelocity 6.4 }} ]
      endPoint Solid {{
        children [ Shape {{ geometry Cylinder {{ radius 0.10 height 0.08 }} }} ]
        name "fl_wheel"
        boundingObject Cylinder {{ radius 0.10 height 0.08 }}
        physics Physics {{ density -1 mass 3 }}
      }}
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 0 1 anchor  0.28 0 0.22 }}
      device [ RotationalMotor {{ name "front right wheel" maxVelocity 6.4 }} ]
      endPoint Solid {{
        children [ Shape {{ geometry Cylinder {{ radius 0.10 height 0.08 }} }} ]
        name "fr_wheel"
        boundingObject Cylinder {{ radius 0.10 height 0.08 }}
        physics Physics {{ density -1 mass 3 }}
      }}
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 0 1 anchor -0.28 0 -0.22 }}
      device [ RotationalMotor {{ name "back left wheel"  maxVelocity 6.4 }} ]
      endPoint Solid {{
        children [ Shape {{ geometry Cylinder {{ radius 0.10 height 0.08 }} }} ]
        name "bl_wheel"
        boundingObject Cylinder {{ radius 0.10 height 0.08 }}
        physics Physics {{ density -1 mass 3 }}
      }}
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 0 1 anchor  0.28 0 -0.22 }}
      device [ RotationalMotor {{ name "back right wheel" maxVelocity 6.4 }} ]
      endPoint Solid {{
        children [ Shape {{ geometry Cylinder {{ radius 0.10 height 0.08 }} }} ]
        name "br_wheel"
        boundingObject Cylinder {{ radius 0.10 height 0.08 }}
        physics Physics {{ density -1 mass 3 }}
      }}
    }}
  ]
  boundingObject Box {{ size 0.55 0.28 0.65 }}
  physics Physics {{ density -1 mass 185 }}
}}

DEF START_MARKER Solid {{
  translation {start_gps[0]:.4f} {start_y + 4.0:.4f} {start_gps[1]:.4f}
  children [
    Shape {{
      appearance Appearance {{
        material Material {{ diffuseColor 0.1 1.0 0.2 emissiveColor 0.0 0.5 0.0 }}
      }}
      geometry Sphere {{ radius 1.8 subdivision 2 }}
    }}
  ]
  name "start_marker"
}}

DEF GOAL_MARKER Solid {{
  translation {goal_gps[0]:.4f} {goal_y + 4.0:.4f} {goal_gps[1]:.4f}
  children [
    Shape {{
      appearance Appearance {{
        material Material {{ diffuseColor 1.0 0.1 0.05 emissiveColor 0.5 0.0 0.0 }}
      }}
      geometry Sphere {{ radius 1.8 subdivision 2 }}
    }}
  ]
  name "goal_marker"
}}
"""
    WBT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WBT_PATH.write_text(wbt, encoding="utf-8")
    print(f"  [WBT] Güncellendi → {WBT_PATH}")


def launch_webots():
    try:
        proc = subprocess.Popen(
            [WEBOTS_EXE, str(WBT_PATH)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"  [Webots] PID={proc.pid}")
        return proc
    except FileNotFoundError:
        print(f"  [Webots] ⚠ '{WEBOTS_EXE}' bulunamadı — yalnızca UI çalışır.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 5 — A* ROTA ÖN-HESAPLAMASI (UI için)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ui_path(hm: np.ndarray,
                    start_cell: tuple,
                    goal_cell:  tuple) -> list:
    """
    FIX-4 ▸ UI'da A* rotasını göstermek için master tarafında
    Pathfinder çalıştırılır (kontrolcüden bağımsız).
    Koordinatlar piksel (col, row) olarak döndürülür.
    """
    if not _PF_AVAILABLE:
        # Pathfinder import edilemezse düz çizgi
        return [
            (start_cell[1], start_cell[0]),
            (goal_cell[1],  goal_cell[0]),
        ]

    try:
        low_map = np.loadtxt(
            str(DATA_RAW / "low_detail_map.csv"), delimiter=" ", dtype=np.float64
        )
        pf   = Pathfinder(low_map,
                          cell_size=CELL_SIZE_M,
                          world_offset_x=-WORLD_M / 2.0,
                          world_offset_z=-WORLD_M / 2.0)
        path = pf.find_path(start_cell, goal_cell)
        if path:
            # (row, col) → (px_x, px_z) piksel koordinatı
            return [(c, r) for r, c in path]
    except Exception as e:
        print(f"  [UI Path] Pathfinder hatası: {e}")

    # Fallback: düz çizgi
    return [(start_cell[1], start_cell[0]), (goal_cell[1], goal_cell[0])]


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 6 — UDP TELEMETRI
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryPacket:
    TYPE_POS  = 1   # uint8 | float32 x | float32 z | float32 speed | float32 heading
    TYPE_OBS  = 2   # uint8 | float32 ox | float32 oz | float32 h
    TYPE_DON  = 3   # uint8
    TYPE_PATH = 4   # uint8 | uint16 count | [float32 x, float32 z] × count

    @staticmethod
    def decode(data: bytes):
        if not data:
            return None
        t = data[0]
        try:
            if t == TelemetryPacket.TYPE_POS and len(data) >= 17:
                _, x, z, spd, hdg = struct.unpack_from("<Bffff", data)
                return {"type": "pos", "x": x, "z": z, "speed": spd, "heading": hdg}
            elif t == TelemetryPacket.TYPE_OBS and len(data) >= 13:
                _, ox, oz, oh = struct.unpack_from("<Bfff", data)
                return {"type": "obs", "x": ox, "z": oz, "height": oh}
            elif t == TelemetryPacket.TYPE_DON:
                return {"type": "done"}
            elif t == TelemetryPacket.TYPE_PATH and len(data) >= 3:
                count = struct.unpack_from("<H", data, 1)[0]
                pts   = []
                off   = 3
                for _ in range(count):
                    if off + 8 > len(data):
                        break
                    px, pz = struct.unpack_from("<ff", data, off)
                    pts.append((px, pz))
                    off += 8
                return {"type": "path", "points": pts}
        except struct.error:
            pass
        return None


class UDPServer(threading.Thread):
    def __init__(self, shared: dict):
        super().__init__(daemon=True)
        self.shared = shared
        self._sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_HOST, UDP_PORT))
        self._sock.settimeout(0.08)
        print(f"  [UDP] {UDP_HOST}:{UDP_PORT} dinleniyor...")

    def run(self):
        while not self.shared.get("quit"):
            try:
                data, _ = self._sock.recvfrom(UDP_BUFSIZE)
                pkt = TelemetryPacket.decode(data)
                if not pkt:
                    continue
                t = pkt["type"]
                if t == "pos":
                    self.shared.update({
                        "pos_x":   pkt["x"],
                        "pos_z":   pkt["z"],
                        "speed":   pkt["speed"],
                        "heading": pkt["heading"],
                    })
                    self.shared["trail_x"].append(pkt["x"])
                    self.shared["trail_z"].append(pkt["z"])
                    if len(self.shared["trail_x"]) > TRAIL_MAX:
                        self.shared["trail_x"].pop(0)
                        self.shared["trail_z"].pop(0)
                elif t == "obs":
                    self.shared["obstacles"].append((pkt["x"], pkt["z"]))
                elif t == "done":
                    self.shared["mission_done"] = True
                elif t == "path":
                    self.shared["udp_path"] = pkt["points"]
            except (socket.timeout, OSError):
                pass
        self._sock.close()


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 7 — 2D CANLI KONTROL PANELİ
# ══════════════════════════════════════════════════════════════════════════════

class LivePanel:
    """
    FIX-4 ▸ A* rotası:
      - İlk olarak master'ın kendi Pathfinder çalıştırmasıyla çizilir (ui_path).
      - Kontrolcü UDP TYPE_PATH paketi gönderirse o kullanılır (daha güncel).
    """

    def __init__(self, hm, start_gps, goal_gps, shared, ui_path):
        self.hm        = hm
        self.start_gps = start_gps
        self.goal_gps  = goal_gps
        self.shared    = shared
        self._half     = WORLD_M / 2.0

        self.fig, axes = plt.subplots(
            1, 2, figsize=(12, 7),
            gridspec_kw={"width_ratios": [3, 1]}
        )
        self.fig.patch.set_facecolor("#0b0b12")
        self.fig.canvas.manager.set_window_title("L.O.R.A.  v1.0 — Canlı Kontrol Paneli")

        self.ax_map  = axes[0]
        self.ax_info = axes[1]

        self._build_map(ui_path)
        self._build_info()

        self._anim = FuncAnimation(
            self.fig, self._update,
            interval=UI_UPDATE_MS, blit=False, cache_frame_data=False
        )

    # ── harita paneli ──────────────────────────────────────────────────────────

    def _gps_px(self, x, z):
        return (x + self._half) / CELL_SIZE_M, (z + self._half) / CELL_SIZE_M

    def _build_map(self, ui_path):
        ax = self.ax_map
        ax.set_facecolor("#0b0b12")

        ls  = LightSource(azdeg=315, altdeg=40)
        rgb = ls.shade(self.hm, cmap=plt.cm.gray, vert_exag=4, blend_mode="soft")
        ax.imshow(rgb, origin="lower",
                  extent=[0, PATCH_SIZE, 0, PATCH_SIZE],
                  interpolation="bilinear", zorder=1)

        # FIX-4 ▸ A* rotası (önce master hesabı)
        if ui_path and len(ui_path) >= 2:
            px = [p[0] for p in ui_path]
            pz = [p[1] for p in ui_path]
            ax.plot(px, pz, color=PATH_CLR, linewidth=1.2,
                    alpha=0.65, zorder=4, label="A* Rota")

        # FIX-4 ▸ UDP ile gelen yenilenen rota (aynı satır nesnesini günceller)
        self._path_line, = ax.plot([], [], color="#88bbff", linewidth=1.0,
                                    alpha=0.5, zorder=4)

        sx, sz = self._gps_px(*self.start_gps)
        gx, gz = self._gps_px(*self.goal_gps)
        ax.plot(sx, sz, "^", color="#00ff66", ms=13, zorder=8, label="Start")
        ax.plot(gx, gz, "*", color="#ff3333", ms=15, zorder=8, label="Goal")

        self._trail_ln,  = ax.plot([], [], color="#00ccff", lw=1.1, alpha=0.8, zorder=6)
        self._rover_dot, = ax.plot([], [], "o", color="#00ffaa", ms=10, zorder=10)

        self._obs_sc = ax.scatter([], [], marker="^", color=OBSTACLE_CLR,
                                   s=55, zorder=9, label="LIDAR Engel")

        ax.set_title("🌑  L.O.R.A. — Ay Yüzeyi (Canlı)",
                     color="#ddd8cc", fontsize=11, pad=6)
        ax.set_xlabel("X (hücre)", color="#666", fontsize=8)
        ax.set_ylabel("Z (hücre)", color="#666", fontsize=8)
        ax.tick_params(colors="#444")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a28")
        ax.legend(loc="upper left", fontsize=8,
                  facecolor="#151520", edgecolor="#2a2a40",
                  labelcolor="white")

    def _build_info(self):
        ax = self.ax_info
        ax.set_facecolor("#0b0b12")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title("Telemetri", color="#ddd8cc", fontsize=10, pad=6)

        kw = dict(color="#b8ccea", fontsize=9, fontfamily="monospace", va="top")
        self._t_pos  = ax.text(0.05, 0.96, "GPS: —",      transform=ax.transAxes, **kw)
        self._t_spd  = ax.text(0.05, 0.86, "Hız: —",      transform=ax.transAxes, **kw)
        self._t_hdg  = ax.text(0.05, 0.78, "Yön: —",      transform=ax.transAxes, **kw)
        self._t_obs  = ax.text(0.05, 0.70, "Engel: 0",    transform=ax.transAxes, **kw)
        self._t_dst  = ax.text(0.05, 0.62, "Hedefe: —",   transform=ax.transAxes, **kw)
        self._t_stat = ax.text(0.05, 0.42, "BEKLENIYOR",
                               transform=ax.transAxes,
                               color="#ffdd44", fontsize=11,
                               fontfamily="monospace", va="top", fontweight="bold")

        # Küçük eğim renk çubuğu göstergesi
        ax.text(0.05, 0.18,
                "■ A* Rota\n■ Rover İzi\n▲ LIDAR Engel",
                transform=ax.transAxes,
                color="#888", fontsize=8, fontfamily="monospace", va="top")

    # ── animasyon ──────────────────────────────────────────────────────────────

    def _update(self, _frame):
        s    = self.shared
        half = self._half

        # İz
        if s["trail_x"]:
            pxs = [(v + half) / CELL_SIZE_M for v in s["trail_x"]]
            pzs = [(v + half) / CELL_SIZE_M for v in s["trail_z"]]
            self._trail_ln.set_data(pxs, pzs)
            self._rover_dot.set_data([pxs[-1]], [pzs[-1]])

        # UDP'den gelen güncel rota
        if s.get("udp_path"):
            ppx = [(p[0] + half) / CELL_SIZE_M for p in s["udp_path"]]
            ppz = [(p[1] + half) / CELL_SIZE_M for p in s["udp_path"]]
            self._path_line.set_data(ppx, ppz)

        # Engeller
        obs = s.get("obstacles", [])
        if obs:
            self._obs_sc.set_offsets(
                [((o[0]+half)/CELL_SIZE_M, (o[1]+half)/CELL_SIZE_M) for o in obs]
            )

        # Telemetri
        x   = s.get("pos_x", 0.0)
        z   = s.get("pos_z", 0.0)
        spd = s.get("speed", 0.0)
        hdg = s.get("heading", 0.0)
        d   = math.sqrt((x - self.goal_gps[0])**2 + (z - self.goal_gps[1])**2)

        self._t_pos.set_text(f"X: {x:+7.1f} m\nZ: {z:+7.1f} m")
        self._t_spd.set_text(f"Hız    : {spd:.2f} m/s")
        self._t_hdg.set_text(f"Yön    : {hdg:.1f}°")
        self._t_obs.set_text(f"Engel  : {len(obs)}")
        self._t_dst.set_text(f"Hedefe : {d:.1f} m")

        if s.get("mission_done"):
            self._t_stat.set_text("✅ GÖREV\nTAMAMLANDI!")
            self._t_stat.set_color("#00ff88")
        elif s["trail_x"]:
            self._t_stat.set_text("🚀 AKTİF\nSEYİR")
            self._t_stat.set_color("#44ddff")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 8 — ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="L.O.R.A. v1.0 Master Controller")
    ap.add_argument("--img",       default="auto")
    ap.add_argument("--no-webots", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print("  L.O.R.A. v1.0 — HACKATHON FINAL")
    print("=" * 62)

    # 1. Heightmap
    print("\n[1/5] Heightmap üretiliyor...")
    chosen = None
    if args.img != "auto":
        p = DATA_RAW / args.img
        chosen = p if p.exists() else None
    hm = generate_heightmap(chosen)

    # 2. Görev noktaları
    print("\n[2/5] Start & Goal seçiliyor...")
    start_cell, goal_cell = pick_start_goal(hm)
    start_gps = cells_to_gps(*start_cell)
    goal_gps  = cells_to_gps(*goal_cell)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    (DATA_PROC / "mission_params.txt").write_text(
        f"START_X={start_gps[0]:.4f}\nSTART_Z={start_gps[1]:.4f}\n"
        f"GOAL_X={goal_gps[0]:.4f}\nGOAL_Z={goal_gps[1]:.4f}\n"
        f"UDP_HOST={UDP_HOST}\nUDP_PORT={UDP_PORT}\n"
    )

    # 3. WBT
    print("\n[3/5] WBT oluşturuluyor...")
    start_y = float(hm[start_cell[0], start_cell[1]])
    goal_y  = float(hm[goal_cell[0],  goal_cell[1]])
    inject_and_write_wbt(hm, start_gps, goal_gps, start_y, goal_y)

    # 4. A* ön-hesaplama (UI)
    print("\n[4/5] UI için A* rotası hesaplanıyor...")
    ui_path = compute_ui_path(hm, start_cell, goal_cell)
    print(f"  [UI Path] {len(ui_path)} waypoint")

    # 5. Webots
    webots_proc = None
    if not args.no_webots:
        print("\n[5/5] Webots başlatılıyor...")
        webots_proc = launch_webots()
        time.sleep(2.0)
    else:
        print("\n[5/5] --no-webots → atlandı.")

    # UDP + UI
    shared = {
        "pos_x": start_gps[0], "pos_z": start_gps[1],
        "speed": 0.0, "heading": 0.0,
        "trail_x": [start_gps[0]], "trail_z": [start_gps[1]],
        "obstacles": [], "udp_path": [],
        "mission_done": False, "quit": False,
    }
    UDPServer(shared).start()
    panel = LivePanel(hm, start_gps, goal_gps, shared, ui_path)

    try:
        panel.show()
    except KeyboardInterrupt:
        pass
    finally:
        shared["quit"] = True
        if webots_proc:
            webots_proc.terminate()
        print("\n  [Master] Kapatıldı.")


if __name__ == "__main__":
    main()