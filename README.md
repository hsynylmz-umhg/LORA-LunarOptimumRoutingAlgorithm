# L.O.R.A. — Lunar Optimum Routing Algorithm

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Webots](https://img.shields.io/badge/Webots-R2023b-E87722?logo=webots&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Hackathon%20Final-blueviolet)

**Otonom Ay Gezgini — Gerçek NASA/JAXA Topografya Verisiyle A\* Navigasyon + LIDAR Engel Kaçınma + Canlı UDP Telemetri**

</div>

---

## 🌑 Proje Özeti

L.O.R.A., Ay'ın güney kutbu bölgesinde çalışacak otonom bir rover (keşif aracı) için geliştirilmiş tam yığın bir navigasyon sistemidir. Gerçek JAXA SLDEM2015 topografya verisinden 500×500 metrelik alanlar kırparak Webots simülasyon ortamında Pioneer 3-AT rover'ı eğim-duyarlı A\* algoritması, LIDAR tabanlı anlık engel kaçınma ve SLAM benzeri harita öğrenimi ile otonom olarak hedefe ulaştırır.

```
JAXA .IMG  →  Heightmap  →  A* Rota  →  Webots Sim  →  UDP Telemetri  →  2D Canlı Panel
```

---

## 🏗️ Sistem Mimarisi

```
LORA/
├── lora_master.py                  ← Orkestra Şefi + 2D Canlı UI
├── data/
│   ├── raw/
│   │   ├── SLDEM2015_256_*.IMG    ← JAXA Ay topografya verisi
│   │   ├── high_detail_map.csv   ← Gerçek yüzey (sürpriz engellerle)
│   │   └── low_detail_map.csv    ← Rover'ın zihinsel haritası
│   └── processed/
│       ├── mission_params.txt    ← Dinamik Start/Goal (master tarafından yazılır)
│       └── learned_map.csv       ← SLAM sonrası öğrenilmiş harita
├── src/
│   ├── algorithm/
│   │   ├── pathfinder.py         ← A* + Eğim Cezası + Webots arayüzü
│   │   ├── slam_mapper.py        ← Gaussian yayılımlı harita öğrenimi
│   │   ├── obstacle_detector.py  ← Sanal sensör + kaçınma yön hesabı
│   │   └── lora_core.py          ← Algoritmik orkestratör (Webots'suz test)
│   ├── data/
│   │   ├── generate_terrain.py   ← Sentetik Perlin terrain üretici
│   │   └── pds_reader.py         ← JAXA .IMG okuyucu (memory-mapped)
│   └── webots/
│       └── world_builder.py      ← CSV → ElevationGrid .wbt üretici
└── webots/
    ├── worlds/lunar_surface.wbt  ← Otomatik üretilen simülasyon dünyası
    └── controllers/
        └── lora_controller/
            └── lora_controller.py ← Pioneer 3-AT Webots kontrolcüsü (FINAL)
```

---

## ⭐ Kritik Teknik Özellikler

### 1. A\* Eğim Cezası (Slope-Aware A\*)

Standart A\* sadece mesafeyi minimize eder. L.O.R.A.'nın Pathfinder'ı her adımda **eğim açısını maliyet fonksiyonuna** ekler:

```python
# pathfinder.py — _slope_cost()
slope_deg = math.degrees(math.atan2(dh, cell_size * base_cost))
penalty   = (slope_deg / MAX_SLOPE_DEG) ** 2 * SLOPE_PENALTY
return base_cost + penalty   # eğim ne kadar dik, maliyet o kadar büyük
```

20°'den dik geçişler tamamen engellenir (`return None` → geçilemez duvar). Bu sayede rover devrilme riski taşıyan yamaçlara hiç girmez; düz ova güzergahlarını, dik kraterleri geçmeye tercih eder.

### 2. Dinamik LIDAR Engel Kaçınma (3 Katmanlı Refleks)

Global A\* rotası haritada olmayan sürpriz boulderları göremez. LIDAR her timestep'te bağımsız çalışır ve 3 mod üretir:

| Mesafe | Eylem |
|--------|-------|
| `< 1.2 m` | 🛑 Acil fren + sabit yön kaçınma (2.8 s) |
| `1.2–4.0 m` | Hızı kıs + engel tarafından uzağa steer |
| `> 8.0 m` | Tam hız, saf A\* navigasyon |

Kaçınma tamamlandıktan sonra **`replan_from()`** çağrılır → güncellenmiş SLAM haritasıyla yeni rota hesaplanır.

### 3. Dinamik Eğim Fiziği (Pitch Modülasyonu)

InertialUnit sensöründen okunan pitch açısı gerçek zamanlı hız ölçekler:

```python
# lora_controller.py — _pitch_speed_scale()
if pitch_deg < -8°:   # Yokuş aşağı → fren
    scale = DOWNHILL_SPEED_SCALE  # 0.45×
elif pitch_deg > +8°: # Yokuş yukarı → tork
    scale = UPHILL_SPEED_SCALE    # 1.15×
```

Bu sayede rover hem krater iç duvarlarında kontrolsüz kayıp yaşamaz, hem de yüksek eğimli tırmanışlarda yeterli tork üretir.

### 4. Rastgele Topografya Üretimi (JAXA .IMG Tabanlı)

`lora_master.py` her çalıştırmada NASA/JAXA SLDEM2015 veri setinden rastgele bir 500×500'lük kesit seçer; düz bölge kriterini (p85 eğim ≤ %12°) geçmezse başka bir koordinat dener. Bu mekanizma sayesinde her simülasyon farklı bir Ay yüzeyi parçasında gerçekleşir.

### 5. UDP Canlı Telemetri Arayüzü

```
lora_controller.py  ──UDP:5005──►  lora_master.py
  send_pos(x, z, speed, heading)    → 2D haritada mavi iz
  send_obstacle(ox, oz, height)     → Turuncu üçgen (LIDAR tespiti)
  send_done()                       → "✅ GÖREV TAMAMLANDI"
```

Binary paket formatı (little-endian):
```
TYPE_POS (1): uint8 | float32 x | float32 z | float32 speed | float32 heading
TYPE_OBS (2): uint8 | float32 ox | float32 oz | float32 height
TYPE_DON (3): uint8
```

---

## 🚀 Kurulum & Çalıştırma

### Gereksinimler

```bash
pip install numpy scipy matplotlib noise
# Webots R2023b: https://cyberbotics.com
```

### Adım 1 — JAXA Verisi (Opsiyonel)

JAXA SELENE/SLDEM2015 verisini indirin ve `data/raw/` klasörüne koyun:
- `SLDEM2015_256_SL_60N_90N_000_120.IMG` (veya benzeri)
- IMG yoksa sistem otomatik olarak Perlin noise ile sentetik harita üretir.

### Adım 2 — Tüm Sistemi Başlat

```bash
# Rastgele IMG seç, Webots'u başlat, canlı UI'ı aç
python lora_master.py

# Belirli IMG dosyasıyla
python lora_master.py --img SLDEM2015_256_SL_60N_90N_000_120.IMG

# Sadece UI testi (Webots olmadan)
python lora_master.py --no-webots
```

### Alternatif — Adım Adım

```bash
# 1. Sentetik harita üret
python src/data/generate_terrain.py

# 2. Webots dünyasını oluştur
python src/webots/world_builder.py

# 3. Webots'u aç → lunar_surface.wbt dosyasını yükle
# 4. Kontrolcü otomatik başlar

# Standalone algoritma testi (Webots olmadan)
python src/algorithm/lora_core.py
```

---

## 📡 360° Kamera Kurulumu (Webots)

`.wbt` dosyasındaki `Viewpoint` düğümünü aşağıdaki gibi yapılandırın:

```vrml
Viewpoint {
  follow           "LORA_ROVER"     # DEF ismiyle eşleşmeli
  followType       "Follow"         # Rover'ın arkasında sabit mesafe
  followSmoothness 0.25             # 0=anlık, 1=çok kaygan
  orientation      -0.707 0.0 0.707 1.57
  position         0 60 -90
  near             0.5
  far              3000
}
```

| Fare Hareketi | Eylem |
|---------------|-------|
| `ALT + Sol Tık + Sürükle` | 360° yatay döndür |
| `ALT + Orta Tık + Sürükle` | Yakınlaştır / Uzaklaştır |
| `Sağ Tık + Sürükle` | Yukarı / Aşağı bak |
| `Scroll` | Zoom |

`followType` seçenekleri: `"None"` · `"Tracking Shot"` · `"Mounted Shot"` · `"Pan and Tilt"` · **`"Follow"`** (önerilen)

---

## 🔬 Algoritma Akışı

```
Her Webots Timestep (32 ms):
  ┌─ GPS + Compass → pose (x, z, heading°)
  ├─ InertialUnit  → pitch° → hız skalası
  ├─ LIDAR analiz  ─── engel < 1.2m ──► ACİL FREN + UDP obs bildir
  │                 ─── engel < 4.0m ──► Yavaşla + taraflı steer
  │                 ─── temiz       ──►  A* navigasyon
  ├─ A* get_next_move(GPS) → waypoint metre
  ├─ PD steering (KP=0.045, KD=0.008) → sol/sağ tekerlek hızı
  ├─ SLAM update_map_region(LIDAR pts)
  └─ UDP send_pos() → 2D canlı panel
```

---

## 📊 Performans Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Grid | 500×500 | 1 m/hücre |
| A\* sezgisel ağırlık | 1.2 | Optimal'e yakın, hızlı |
| Maks geçilebilir eğim | 20° | Pioneer 3-AT limiti |
| LIDAR menzil | 30 m | Sick LMS 291 eşdeğeri |
| SLAM Gaussian σ | 1.5 hücre | Engel yayılım yumuşaklığı |
| UDP güncelleme | ~10 Hz | 3 adımda bir paket |
| TIME\_STEP | 32 ms | ~31 FPS simülasyon |

---

## 👥 Takım

**L.O.R.A. Hackathon Takımı**
- Proje Yöneticisi & 3D Tasarım: *[İsim]*
- Yazılım, Algoritma & Simülasyon: *[İsim]*

---

## 📄 Lisans

MIT License — Akademik ve araştırma amaçlı kullanıma açıktır.

---

<div align="center">
<sub>Veriler: JAXA SELENE SLDEM2015 · Simülasyon: Cyberbotics Webots R2023b · Robot: Pioneer 3-AT</sub>
</div>