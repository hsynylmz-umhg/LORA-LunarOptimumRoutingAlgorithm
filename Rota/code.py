import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


# https://planetarynames.wr.usgs.gov/Page/moon1to10mShadedRelief


try:
    import imageio
except Exception:
    imageio = None

# --- Varsayilan Ayarlar ---
DOSYA_ADI = "mesut.jpg"

# OPTIMIZASYON: 80 MB+ görseller için agresif küçültme.
# Orijinal kodda 2 idi — bu değer rota grid boyutunu doğrudan belirler.
# 2 → ~25M piksel grid = çok yavaş / bellek taşması
# 8 → ~1.5M piksel grid = hızlı ve yeterli hassasiyet
HIZLI_MOD_ORANI = 8

# OPTIMIZASYON: Gösterim için de ayrı bir küçültme oranı.
# Matplotlib'e 80 MB ham piksel vermek çizimi donduruyor.
# Bu değer sadece ekranda gösterilen görseli etkiler, rota hassasiyetini değil.
GOSTERIM_ORANI = 4
MAX_GRID_MEGAPIKSEL = 2.0
MAX_GOSTERIM_MEGAPIKSEL = 1.5

TARAMA_YARICAPI = 40
HEDEF_VIDEO_SURE_SANIYE = 7.5
Image.MAX_IMAGE_PIXELS = None

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
TOPO_ADAY_DOSYALAR = (
    "mesut2.jpg",
    "meust2.jpg",
    "mesut2.png",
    "topografya.png",
    "topography.png",
    "moon_sp_page-0001 (1).jpg",
)


def gorsel_yolunu_coz(resim_yolu: str) -> Path:
    aday = Path(resim_yolu)
    if aday.exists():
        return aday

    # Kullanici uzanti vermediyse bilinen uzantilari dene.
    if aday.suffix == "":
        for uzanti in DESTEKLENEN_UZANTILAR:
            alternatif = aday.with_suffix(uzanti)
            if alternatif.exists():
                return alternatif

    # Varsayilan ad bulunamazsa proje klasorunde mesut.* aramasi yap.
    proje_klasoru = Path.cwd()
    for uzanti in DESTEKLENEN_UZANTILAR:
        alternatif = proje_klasoru / f"mesut{uzanti}"
        if alternatif.exists():
            return alternatif

    raise FileNotFoundError(
        "Gorsel bulunamadi. Ayni klasore mesut.jpg veya mesut.png koyun "
        "ya da --image ile acik dosya yolu verin."
    )


def topo_gorseli_bul(topo_yolu: str | None, ana_gorsel: Path) -> Path | None:
    if topo_yolu:
        return gorsel_yolunu_coz(topo_yolu)

    proje_klasoru = Path.cwd()

    for ad in TOPO_ADAY_DOSYALAR:
        aday = proje_klasoru / ad
        if aday.exists() and aday.resolve() != ana_gorsel.resolve():
            return aday

    for aday in sorted(proje_klasoru.iterdir()):
        if not aday.is_file() or aday.suffix.lower() not in DESTEKLENEN_UZANTILAR:
            continue
        if aday.resolve() == ana_gorsel.resolve():
            continue
        ad_kucuk = aday.name.lower()
        if "topo" in ad_kucuk or "topograf" in ad_kucuk or "moon" in ad_kucuk:
            return aday

    return None


def normalize_harita(veri: np.ndarray, alt: float = 1.0, ust: float = 99.0) -> np.ndarray:
    alt_deger = np.percentile(veri, alt)
    ust_deger = np.percentile(veri, ust)
    if ust_deger <= alt_deger:
        return np.zeros_like(veri, dtype=np.float32)
    return np.clip((veri - alt_deger) / (ust_deger - alt_deger), 0.0, 1.0).astype(np.float32)





def rgbden_topo_yukseklik(topo_rgb: np.ndarray) -> np.ndarray:
    """
    Topografya haritasındaki RGB renklerini yükseklik/derinlik değerine çevirir.
    LOLA renk skalası: 
    - Kırmızı/Turuncu: yüksek (+8000m)
    - Sarı/Yeşil: orta (0m)
    - Mavi: düşük (-4000m)
    - Mor/Lacivert: çok düşük (-8000m)
    """
    r = topo_rgb[:, :, 0].astype(np.float32) / 255.0
    g = topo_rgb[:, :, 1].astype(np.float32) / 255.0
    b = topo_rgb[:, :, 2].astype(np.float32) / 255.0
    
    # HSV'ye dönüştür
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    eps = 1e-6
    
    hue = np.zeros_like(r)
    
    # Hue hesaplaması
    maske_r = (cmax == r) & (delta > eps)
    hue[maske_r] = (60.0 * ((g[maske_r] - b[maske_r]) / delta[maske_r])) % 360
    
    maske_g = (cmax == g) & (delta > eps)
    hue[maske_g] = 60.0 * ((b[maske_g] - r[maske_g]) / delta[maske_g]) + 120.0
    
    maske_b = (cmax == b) & (delta > eps)
    hue[maske_b] = 60.0 * ((r[maske_b] - g[maske_b]) / delta[maske_b]) + 240.0
    
    saturation = np.zeros_like(cmax, dtype=np.float32)
    gecerli = cmax > eps
    saturation[gecerli] = delta[gecerli] / cmax[gecerli]
    
    # Hue → Yükseklik eşlemesi
    yukseklik = np.zeros_like(hue)
    
    # 0° - 60°: Kırmızı/Turuncu (yüksek)
    m = (hue < 60) & (saturation > 0.2)
    yukseklik[m] = 0.8 + 0.2 * (hue[m] / 60.0)
    
    # 60° - 120°: Sarı/Yeşil (orta)
    m = (hue >= 60) & (hue < 120) & (saturation > 0.2)
    yukseklik[m] = 0.5 + 0.3 * ((120 - hue[m]) / 60.0)
    
    # 120° - 180°: Cyan/Açık Mavi (orta-düşük)
    m = (hue >= 120) & (hue < 180) & (saturation > 0.2)
    yukseklik[m] = 0.3 + 0.2 * ((180 - hue[m]) / 60.0)
    
    # 180° - 240°: Mavi (düşük)
    m = (hue >= 180) & (hue < 240) & (saturation > 0.2)
    yukseklik[m] = 0.1 + 0.2 * ((240 - hue[m]) / 60.0)
    
    # 240° - 300°: Mor/Lacivert (çok düşük)
    m = (hue >= 240) & (hue < 300) & (saturation > 0.25)
    yukseklik[m] = 0.05
    
    # 300° - 360°: Kırmızı-Leylak (yüksek)
    m = (hue >= 300) & (saturation > 0.2)
    yukseklik[m] = 0.75 + 0.25 * ((hue[m] - 300) / 60.0)
    
    # Düşük saturation = beyaz/gri → orta yükseklik
    yukseklik[saturation < 0.15] = 0.5
    
    return np.clip(yukseklik, 0.0, 1.0).astype(np.float32)


class MesutPngNavigasyon:
    def __init__(
        self,
        resim_yolu: str,
        topo_yolu: str | None = None,
        hizli_mod_orani: int = HIZLI_MOD_ORANI,
        gosterim_orani: int = GOSTERIM_ORANI,
        video_dizin: str = "kayitlar",
        video_fps: int = 20,
    ):
        self.resim_yolu = gorsel_yolunu_coz(resim_yolu)
        self.topo_yolu = topo_gorseli_bul(topo_yolu, self.resim_yolu)
        self.hizli_mod_orani = max(1, int(hizli_mod_orani))
        self.gosterim_orani = max(1, int(gosterim_orani))
        self.video_dizin = Path(video_dizin)
        self.cikti_dizin = Path("ciktilar")
        self.video_fps = max(1, int(video_fps))
        self.noktalar = []
        self.cizimler = []
        self.kayit_aktif = False
        self.kayit_yazici = None
        self.kayit_dosyasi = None
        
        # Animasyon state'i
        self.animasyon_rota = None
        self.animasyon_index = 0
        self.animasyon_cizim_line = None
        self.animasyon_marker = None
        self.animasyon_aktif = False
        self._yolu_hazirla()

    def _yolu_hazirla(self) -> None:
        print(f"'{self.resim_yolu.name}' dosyasi yukleniyor...")
        with Image.open(self.resim_yolu) as img:
            # JPEG'de draft, yuksek cozunurlukte bellek ve yukleme suresini azaltir.
            if img.format and img.format.upper() == "JPEG":
                img.draft("L", (4096, 4096))

            gri = img.convert("L")
            genislik, yukseklik = gri.size
            print(f"Orijinal boyut: {genislik}x{yukseklik} piksel")

            if genislik < 1 or yukseklik < 1:
                raise ValueError("Gorsel boyutu gecersiz.")

            toplam_piksel = genislik * yukseklik
            min_grid_orani = math.ceil(math.sqrt(toplam_piksel / (MAX_GRID_MEGAPIKSEL * 1_000_000)))
            min_gosterim_orani = math.ceil(
                math.sqrt(toplam_piksel / (MAX_GOSTERIM_MEGAPIKSEL * 1_000_000))
            )
            etkin_grid_orani = max(self.hizli_mod_orani, min_grid_orani, 1)
            etkin_gosterim_orani = max(self.gosterim_orani, min_gosterim_orani, 1)

            if etkin_grid_orani > self.hizli_mod_orani or etkin_gosterim_orani > self.gosterim_orani:
                print(
                    "Buyuk gorsel algilandi. Donmayi engellemek icin otomatik olcekleme uygulandi: "
                    f"grid_orani={etkin_grid_orani}, gosterim_orani={etkin_gosterim_orani}"
                )

            # OPTIMIZASYON: Gösterim görseli — ekrana verilecek küçültülmüş kopya.
            # Orijinal kodda tam çözünürlük kullanılıyordu; bu Matplotlib'i donduruyordu.
            gosterim_boyut = (
                max(1, genislik // etkin_gosterim_orani),
                max(1, yukseklik // etkin_gosterim_orani),
            )
            gosterim_gorseli = img.convert("RGB").resize(gosterim_boyut, Image.Resampling.BICUBIC)
            self.gosterim_haritasi = np.asarray(gosterim_gorseli, dtype=np.uint8)

            # OPTIMIZASYON: Rota grid — hesaplama için ayrıca küçültülmüş kopya.
            hedef_boyut = (
                max(1, genislik // etkin_grid_orani),
                max(1, yukseklik // etkin_grid_orani),
            )
            hesaplama_gorseli = gri.resize(hedef_boyut, Image.Resampling.BICUBIC)
            hesaplama_gorseli = hesaplama_gorseli.filter(ImageFilter.GaussianBlur(radius=1.0))

        topo_yukseklik_norm = None
        if self.topo_yolu is not None:
            print(f"Topografya haritasi kullaniliyor: '{self.topo_yolu.name}'")
            with Image.open(self.topo_yolu) as topo_img:
                topo_rgb = topo_img.convert("RGB").resize(hedef_boyut, Image.Resampling.BICUBIC)
                topo_haritasi = np.asarray(
                    topo_rgb.filter(ImageFilter.GaussianBlur(radius=1.0)),
                    dtype=np.float32,
                )
                topo_yukseklik_norm = rgbden_topo_yukseklik(topo_haritasi)
        else:
            print("Topografya haritasi bulunamadi. Rota yalnizca ana gorselden hesaplanacak.")

        self.harita = np.asarray(hesaplama_gorseli, dtype=np.float32)
        if self.harita.ndim != 2 or self.gosterim_haritasi.ndim != 3:
            raise ValueError("Hesaplama haritasi 2B, gosterim haritasi 3B RGB olmali.")

        dy, dx = np.gradient(self.harita)
        egim_ana = np.hypot(dx, dy)
        ana_egim_norm = normalize_harita(egim_ana)

        normalize = self.harita / 255.0
        koyuluk_cezasi = np.where(normalize < 0.15, 0.8, 0.0).astype(np.float32)

        # --- TOPOGRAFYA ANALIZI ---
        # Rota mantiginin duzlesmemesi icin topo etkisini sadece egimde degil,
        # mutlak yukseklik/derinlikte de kullaniyoruz.
        topo_egim_norm = np.zeros_like(self.harita, dtype=np.float32)
        topo_puruz_norm = np.zeros_like(self.harita, dtype=np.float32)
        derinlik_cezasi = np.zeros_like(self.harita, dtype=np.float32)
        zirve_cezasi = np.zeros_like(self.harita, dtype=np.float32)
        if topo_yukseklik_norm is not None:
            topo_yukseklik_norm = np.clip(topo_yukseklik_norm.astype(np.float32), 0.0, 1.0)

            # Topografya verisinden egrimi hesapla (engebe seviyesi)
            tdy, tdx = np.gradient(topo_yukseklik_norm)
            topo_egim = np.hypot(tdx, tdy)

            # Ikinci turev (pürüzlülük): keskin kırık bölgelerden kaçınma
            ddy, ddx = np.gradient(topo_egim)
            topo_puruz = np.hypot(ddx, ddy)

            topo_egim_norm = normalize_harita(topo_egim)
            topo_puruz_norm = normalize_harita(topo_puruz)

            # Mavi/mor (dusuk deger) alanlar daha pahali, beyaz/sari/yesil daha ucuz.
            derinlik_cezasi = np.clip((0.48 - topo_yukseklik_norm) / 0.48, 0.0, 1.0)
            # Asiri parlak zirvelerde de hafif ceza uygula; rota daha dengeli olur.
            zirve_cezasi = np.clip((topo_yukseklik_norm - 0.93) / 0.07, 0.0, 1.0)

        # --- ENTEGRE MALIYET MATRISI ---
        # Topografya agirligi yuksek tutuldu: rota artik duz cizgiye kacmaz.
        maliyet = (
            1.0
            + (1.1 * ana_egim_norm)
            + (3.2 * topo_egim_norm)
            + (2.3 * topo_puruz_norm)
            + (3.0 * derinlik_cezasi)
            + (0.8 * zirve_cezasi)
            + koyuluk_cezasi
        )
        maliyet = np.clip(maliyet, 1.0, 14.0)

        # Dinamik araligi artiriyoruz ki A* topo farkini net hissetsin.
        self.maliyet = np.clip((maliyet * 80).astype(np.int32), 1, np.iinfo(np.int32).max)

        self.grid_boyut = (int(self.harita.shape[1]), int(self.harita.shape[0]))
        self.gosterim_boyut = (
            int(self.gosterim_haritasi.shape[1]),
            int(self.gosterim_haritasi.shape[0]),
        )

        # OPTIMIZASYON: Gösterim ↔ grid koordinat dönüşüm oranı.
        # Orijinal kodda gosterim=orijinal çözünürlük olduğundan oran farklıydı.
        # Şimdi her iki boyut da küçültülmüş, oranlar buna göre güncellendi.
        self.oran_x = self.grid_boyut[0] / self.gosterim_boyut[0]
        self.oran_y = self.grid_boyut[1] / self.gosterim_boyut[1]

        print(
            "Sistem hazir. "
            f"Gosterim: {self.gosterim_boyut[0]}x{self.gosterim_boyut[1]} | "
            f"Rota grid: {self.grid_boyut[0]}x{self.grid_boyut[1]}"
        )

    def _yeni_grid(self) -> Grid:
        # OPTIMIZASYON: Her seferinde .tolist() çağrısı büyük grid'lerde yavaştı.
        # maliyet_liste'yi bir kez saklayıp tekrar kullanıyoruz.
        if not hasattr(self, "_maliyet_liste"):
            print("Grid önbelleğe alınıyor (ilk seferinde biraz bekleyin)...")
            self._maliyet_liste = self.maliyet.tolist()
        return Grid(matrix=self._maliyet_liste)

    def _koordinat_gecerli_mi(self, x: int, y: int) -> bool:
        return 0 <= x < self.gosterim_boyut[0] and 0 <= y < self.gosterim_boyut[1]

    def _gosterimden_gride(self, x: int, y: int) -> tuple[int, int]:
        gx = min(self.grid_boyut[0] - 1, max(0, int(x * self.oran_x)))
        gy = min(self.grid_boyut[1] - 1, max(0, int(y * self.oran_y)))
        return gx, gy

    def _gridden_gosterime(self, gx: float, gy: float) -> tuple[float, float]:
        x = (gx + 0.5) / self.oran_x
        y = (gy + 0.5) / self.oran_y
        return x, y

    def _path_koordinatlari(self, path) -> list[tuple[int, int]]:
        koordinatlar = []
        for nokta in path:
            if isinstance(nokta, (tuple, list)) and len(nokta) >= 2:
                koordinatlar.append((int(nokta[0]), int(nokta[1])))
                continue

            # pathfinding bazi surumlerde GridNode nesnesi dondurur.
            x = getattr(nokta, "x", None)
            y = getattr(nokta, "y", None)
            if x is None or y is None:
                raise TypeError(f"Beklenmeyen rota nokta tipi: {type(nokta)}")
            koordinatlar.append((int(x), int(y)))
        return koordinatlar

    def _rotayi_yumusat(self, rota: list[tuple[int, int]], pencere: int = 5) -> list[tuple[float, float]]:
        if len(rota) < 3:
            return [self._gridden_gosterime(x, y) for x, y in rota]

        yaricap = max(1, pencere // 2)
        yumusak = []
        for i in range(len(rota)):
            sol = max(0, i - yaricap)
            sag = min(len(rota), i + yaricap + 1)
            parcax = [p[0] for p in rota[sol:sag]]
            parcay = [p[1] for p in rota[sol:sag]]
            ortx = float(np.mean(parcax))
            orty = float(np.mean(parcay))
            yumusak.append(self._gridden_gosterime(ortx, orty))

        yumusak[0] = self._gridden_gosterime(*rota[0])
        yumusak[-1] = self._gridden_gosterime(*rota[-1])
        return yumusak

    def _temizle(self) -> None:
        for cizim in self.cizimler:
            try:
                cizim.remove()
            except ValueError:
                pass
        self.cizimler.clear()
        self.noktalar.clear()

        if self.animasyon_cizim_line is not None:
            try:
                self.animasyon_cizim_line.remove()
            except (ValueError, AttributeError):
                pass
            self.animasyon_cizim_line = None

        if self.animasyon_marker is not None:
            try:
                self.animasyon_marker.remove()
            except (ValueError, AttributeError):
                pass
            self.animasyon_marker = None

        self.animasyon_rota = None
        self.animasyon_index = 0
        self.animasyon_aktif = False

        if self.kayit_aktif:
            self._kayit_durdur()

    def _kare_al(self, fig) -> np.ndarray:
        # draw() ve buffer_rgba() birlikte kullanildiginda backend kaynakli genislik/yukseklik
        # uyumsuzlugu engellenir; numpy dogrudan dogru sekli alir.
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Beklenmeyen frame boyutu: {rgba.shape}")
        return np.ascontiguousarray(rgba[:, :, :3])

    def _kayit_baslat(self, fig) -> None:
        if self.kayit_aktif:
            return
        if imageio is None:
            print("Video kaydi icin imageio paketi gerekli. requirements.txt ile kurulum yapin.")
            return
        self.video_dizin.mkdir(parents=True, exist_ok=True)
        zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kayit_dosyasi = self.video_dizin / f"rota_kaydi_{zaman}.mp4"
        self.kayit_yazici = imageio.get_writer(str(self.kayit_dosyasi), fps=self.video_fps)
        self.kayit_aktif = True
        print(f"Kayit basladi: {self.kayit_dosyasi} (otomatik, V ile de durdurabilirsiniz)")
        try:
            self.kayit_yazici.append_data(self._kare_al(fig))
        except Exception as exc:
            print(f"Video baslatma hatasi: {exc}")
            self._kayit_durdur()

    def _kayit_durdur(self) -> None:
        if not self.kayit_aktif:
            return
        self.kayit_aktif = False
        if self.kayit_yazici is not None:
            self.kayit_yazici.close()
            self.kayit_yazici = None
        if self.kayit_dosyasi is not None:
            print(f"Kayit tamamlandi: {self.kayit_dosyasi}")

    def _kare_tick(self, fig) -> None:
        if self.kayit_aktif and self.kayit_yazici is not None:
            try:
                self.kayit_yazici.append_data(self._kare_al(fig))
            except Exception as exc:
                print(f"Video kare yakalama hatasi: {exc}")
                self._kayit_durdur()

    def _cikti_kaydet(self, fig) -> None:
        self.cikti_dizin.mkdir(parents=True, exist_ok=True)
        zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
        dosya = self.cikti_dizin / f"rota_cikti_{zaman}.png"
        fig.savefig(dosya, dpi=180, bbox_inches="tight")
        print(f"Cikti kaydedildi: {dosya}")

    def _timer_tick(self, fig, ax) -> None:
        # Once animasyonu ilerlet, sonra frame'i yaz: boylece video her zaman guncel kareyi alir.
        self._animasyon_tick(fig, ax)
        self._kare_tick(fig)

    def _animasyon_tick(self, fig, ax) -> None:
        """Rota animasyonunu frame-by-frame göstermek için."""
        if not self.animasyon_aktif or self.animasyon_rota is None:
            return
        
        # Kullanici sure ayari yapmadan, rota uzunluguna gore otomatik hizlanir.
        hedef_kare_sayisi = max(1, int(self.video_fps * HEDEF_VIDEO_SURE_SANIYE))
        hizli_mod = max(1, int(math.ceil(len(self.animasyon_rota) / hedef_kare_sayisi)))
        self.animasyon_index += hizli_mod
        
        if self.animasyon_index >= len(self.animasyon_rota):
            # Animasyon bitti
            self.animasyon_index = len(self.animasyon_rota) - 1
            self.animasyon_aktif = False
            # Son frame'de bitis noktasini belirgin bir sekilde birak.
            son_x, son_y = self.animasyon_rota[-1]

            # Son rotayi eksiksiz goster.
            son_xlar = [p[0] for p in self.animasyon_rota]
            son_yler = [p[1] for p in self.animasyon_rota]
            if self.animasyon_cizim_line is None:
                self.animasyon_cizim_line, = ax.plot([], [], color="cyan", linewidth=2.8, zorder=4)
            self.animasyon_cizim_line.set_data(son_xlar, son_yler)

            if self.animasyon_marker is None:
                self.animasyon_marker = ax.scatter([son_x], [son_y], color="yellow", s=90, zorder=6)
            else:
                self.animasyon_marker.set_offsets(np.array([[son_x, son_y]], dtype=np.float32))
                self.animasyon_marker.set_color("yellow")

            fig.canvas.draw()

            # Son goruntunun videoya kesin yazilmasi icin ekstra kare bas.
            if self.kayit_aktif and self.kayit_yazici is not None:
                try:
                    son_kare = self._kare_al(fig)
                    self.kayit_yazici.append_data(son_kare)
                    self.kayit_yazici.append_data(son_kare)
                except Exception as exc:
                    print(f"Final kare yazma hatasi: {exc}")
                self._kayit_durdur()
            return
        
        # Mevcut noktaya kadar rota çiz
        rota_parcasi_x = [p[0] for p in self.animasyon_rota[:self.animasyon_index + 1]]
        rota_parcasi_y = [p[1] for p in self.animasyon_rota[:self.animasyon_index + 1]]
        
        # Her karede yeniden plot etmek yerine mevcut cizimi guncelle (daha hizli).
        if self.animasyon_cizim_line is None:
            self.animasyon_cizim_line, = ax.plot([], [], color="cyan", linewidth=2.8, zorder=4)
        self.animasyon_cizim_line.set_data(rota_parcasi_x, rota_parcasi_y)
        
        # Hareketli pozisyon marker'i tek nesne olarak guncellenir.
        if self.animasyon_index > 0:
            noktalar = np.array([[rota_parcasi_x[-1], rota_parcasi_y[-1]]], dtype=np.float32)
            if self.animasyon_marker is None:
                self.animasyon_marker = ax.scatter(noktalar[:, 0], noktalar[:, 1], color="red", s=45, zorder=6)
            else:
                self.animasyon_marker.set_offsets(noktalar)
                self.animasyon_marker.set_color("red")
        
        fig.canvas.draw_idle()
    def gorev_baslat(self) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Gorsel kalite icin bicubic interpolation; performans icin gosterim boyutu zaten dusuruluyor.
        ax.imshow(self.gosterim_haritasi, interpolation="bicubic")
        ax.set_title("MESUT - Guvenli Rota (Ana + Topografya)")

        fig.canvas.draw()

        timer = fig.canvas.new_timer(interval=max(25, int(1000 / self.video_fps)))
        timer.add_callback(lambda: self._timer_tick(fig, ax))
        timer.start()

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return

            x, y = int(event.xdata), int(event.ydata)
            if not self._koordinat_gecerli_mi(x, y):
                return

            if event.button == 3:  # Sag tik: sifirla
                self._temizle()
                plt.draw()
                return

            if event.button == 2:  # Orta tik: tarama dairesi
                t = np.linspace(0, 2 * np.pi, 64)
                daire_x = x + TARAMA_YARICAPI * np.cos(t)
                daire_y = y + TARAMA_YARICAPI * np.sin(t)
                line = ax.plot(daire_x, daire_y, color="orange", linestyle="--", linewidth=2)
                self.cizimler.append(line[0])
                plt.draw()
                return

            if event.button == 1:  # Sol tik: rota baslat/bitir
                if len(self.noktalar) >= 2:
                    return

                self.noktalar.append((x, y))
                renk = "lime" if len(self.noktalar) == 1 else "red"
                nokta_cizim = ax.scatter(x, y, color=renk, s=60, zorder=5)
                self.cizimler.append(nokta_cizim)
                plt.draw()

                if len(self.noktalar) == 2:
                    print("Rota hesaplaniyor...")
                    grid = self._yeni_grid()
                    baslangic = grid.node(*self._gosterimden_gride(*self.noktalar[0]))
                    bitis = grid.node(*self._gosterimden_gride(*self.noktalar[1]))
                    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
                    path, _ = finder.find_path(baslangic, bitis, grid)

                    if path:
                        koordinat_rota = self._path_koordinatlari(path)
                        yumusak_rota = self._rotayi_yumusat(koordinat_rota, pencere=5)
                        print(f"Guvenli rota bulundu. Adim sayisi: {len(koordinat_rota)}")

                        # Onceki video kaydi aciksa yeni rota oncesi kapat.
                        if self.kayit_aktif:
                            self._kayit_durdur()

                        # Onceki animasyon cizimlerini temizle.
                        if self.animasyon_cizim_line is not None:
                            try:
                                self.animasyon_cizim_line.remove()
                            except (ValueError, AttributeError):
                                pass
                            self.animasyon_cizim_line = None
                        if self.animasyon_marker is not None:
                            try:
                                self.animasyon_marker.remove()
                            except (ValueError, AttributeError):
                                pass
                            self.animasyon_marker = None
                        
                        # Animasyon başlat
                        self.animasyon_rota = yumusak_rota
                        self.animasyon_index = 0
                        self.animasyon_aktif = True
                        
                        # Otomatik video kaydetmeyi başlat
                        hedef_kare_sayisi = max(1, int(self.video_fps * HEDEF_VIDEO_SURE_SANIYE))
                        hizli_mod = max(1, int(math.ceil(len(yumusak_rota) / hedef_kare_sayisi)))
                        tahmini_sure = len(yumusak_rota) / max(1, self.video_fps * hizli_mod)
                        print(f"Animasyon başlatıldı. Tahmini video süresi: {tahmini_sure:.1f} saniye")
                        if imageio is not None and not self.kayit_aktif:
                            self._kayit_baslat(fig)
                    else:
                        print("Uyari: Secilen noktalar arasinda rota bulunamadi.")
                    plt.draw()

        def onkey(event):
            if not event.key:
                return
            tus = event.key.lower()
            if tus == "v":
                if self.kayit_aktif:
                    self._kayit_durdur()
                else:
                    self._kayit_baslat(fig)
            elif tus == "s":
                self._cikti_kaydet(fig)

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkey)
        fig.canvas.mpl_connect("close_event", lambda _event: self._kayit_durdur())
        print("Video kaydi icin V, PNG cikti icin S tusuna bas")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesut JPG/PNG navigasyon araci")
    parser.add_argument("--image", default=DOSYA_ADI, help="Kullanilacak gorsel dosyasi")
    parser.add_argument(
        "--topo-image",
        default=None,
        help="Topografya haritasi dosyasi (opsiyonel, verilmezse otomatik aranir)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=HIZLI_MOD_ORANI,
        help="Rota grid icin kucultme orani (varsayilan: 8)",
    )
    parser.add_argument(
        "--display-scale",
        type=int,
        default=GOSTERIM_ORANI,
        help="Ekran gosterimi icin kucultme orani (varsayilan: 4)",
    )
    parser.add_argument(
        "--video-dir",
        default="kayitlar",
        help="Kayit dosyalarinin yazilacagi klasor",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="Video kayit FPS degeri (varsayilan: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        nav = MesutPngNavigasyon(
            args.image,
            topo_yolu=args.topo_image,
            hizli_mod_orani=args.scale,
            gosterim_orani=args.display_scale,
            video_dizin=args.video_dir,
            video_fps=args.video_fps,
        )
        nav.gorev_baslat()
    except Exception as exc:
        print(f"HATA: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()