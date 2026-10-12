# L.O.R.A. — Lunar Optimum Routing Algorithm

Ay yerçekiminde çalışan dört tekerlekli rover, eğim ve engel duyarlı A*, lidar ile engel öğrenimi ve canlı görev paneli. Bu MVP, **Webots R2025a** için düzeltildi. Varsayılan 64×64 örneklik (63×63 m) demo tekrarlanabilir; haritada önceden gösterilmeyen bir kaya içerir.

## Hızlı başlatma — Windows

Kurulmuş bu kopyada **`Start-LORA.cmd` dosyasına çift tıklayın**. Webots ve canlı panel birlikte açılır. Görev tamamlanınca rover durur; paneli kapatmak bu başlatıcının Webots sürecini de kapatır.

Yeni bir bilgisayarda Webots R2025a ve bağımsız Python 3.11+ kurun, ardından depo klasöründe:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\Start-LORA.cmd
```

`setup.ps1` proje içindeki `.venv` ortamını kullanır. Anaconda kaynaklı DLL hatasında bağımsız yorumlayıcıyı belirtin: `setup.ps1 -Python 'C:\Python314\python.exe'`. Var olan `.venv` yeniden kullanılacağından, bozuk bir ortamı önce farklı bir ada taşıyın. Sistemin Python veya Webots tercihlerini değiştirmek gerekmez; başlatıcı kontrolcüye kendi yorumlayıcısını `runtime.ini` üzerinden verir.

Bu makinede doğrulanan ortam: Windows, Webots R2025a, Python 3.14.3. Tam paket sürümleri `requirements-lock.txt` içindedir; bu kilit dosyası doğrulanan Python sürümünü yeniden üretmek içindir. Genel bağımlılık aralıkları `requirements.txt` dosyasındadır.

## Çalıştırma seçenekleri

```powershell
# Varsayılan: hafif engebeli yüzey + önceden bilinmeyen kaya
.\.venv\Scripts\python.exe lora_master.py

# Düz yüzeyde fizik ve motor kontrolü
.\.venv\Scripts\python.exe lora_master.py --scenario flat

# Penceresiz kabul testi; görev sonunda başarılıysa çıkış kodu 0
.\.venv\Scripts\python.exe lora_master.py --headless --scenario demo

# Eski projenin 500×500 boyutu; görev süresi rota uzunluğuna göre artar
.\.venv\Scripts\python.exe lora_master.py --size 500 --headless

# Dünya, harita ve görev dosyasını üret; Webots açmadan çık
.\.venv\Scripts\python.exe lora_master.py --prepare-only

# Canlı paneli aç; simülatörü otomatik açma
.\.venv\Scripts\python.exe lora_master.py --no-webots

# Farklı tekrarlanabilir yüzey
.\.venv\Scripts\python.exe lora_master.py --seed 7
```

Webots başka klasördeyse `--webots 'D:\Webots\msys64\mingw64\bin\webots.exe'` veya `WEBOTS_HOME` kullanın. `--timeout` batch duvar saati sınırıdır; `--sim-timeout` simülasyon süresi sınırıdır. Varsayılan simülasyon süresi rota uzunluğuna göre belirlenir. Büyük görevlerde gerekirse `--timeout 600` verin. Aynı klasörden ikinci görev başlatılması engellenir; birinci paneli kapatın.

## Bileşenler ve ilişkileri

```text
lora_master.py
  ├─ sentetik yüzey / src/data/pds_reader.py → metre cinsinden heightmap
  ├─ src/algorithm/pathfinder.py → erişilebilir start/goal + ilk A* rotası
  ├─ data/processed/terrain.npy + mission.json → ortak görev sözleşmesi
  ├─ src/webots/world_builder.py → webots/worlds/lunar_surface.wbt
  ├─ runtime.ini → aynı .venv ile Webots kontrolcüsü
  └─ src/telemetry.py → canlı Matplotlib paneli / batch raporu

webots/controllers/lora_controller/lora_controller.py
  ├─ GPS + Compass + InertialUnit + Lidar → src/navigation.py
  ├─ src/algorithm/slam_mapper.py → şişirilmiş engel occupancy katmanı
  ├─ aynı Pathfinder + paylaşılan occupancy → yeniden rota planlama
  ├─ diferansiyel sürüş → 4 motor
  └─ UDP + mission_result.json + controller günlüğü → ölçülebilir sonuç
```

Tek koordinat sözleşmesi **ENU**: X doğu, Y kuzey, Z yükseklik. Rover yerel +X yönünde ilerler, +Y soludur. `heightmap[row,col]` noktası `(offset_x+col*cell_size, offset_y+row*cell_size)` konumundadır. Heading, +X ekseninden saat yönünün tersine radyandır. Eski Pathfinder `world_offset_z` parametresi uyumluluk için ikinci **yatay** eksenin takma adı olarak korunmuştur; yeni kod `world_offset_y` kullanabilir.

## Düzeltilen temel sorunlar

- Eksik `slam_mapper.py` eklendi; yalnızca derlenmiş `.pyc` dosyasına bağımlılık kaldırıldı.
- Karışık Y-up/Z-up dünya, hatalı negatif gravity, eski ElevationGrid alanları ve geçersiz `Follow` değeri düzeltildi.
- Tekerleklerin gerçek konumları, aksları ve silindir yönleri tanımlandı; gövdeye yerden açıklık verildi.
- Pusula, lidar, harita ve motor yönleri aynı ENU sözleşmesine alındı.
- İlk plan gerçek GPS ölçümünden başlıyor; finite sensörler ve oturma süresi bekleniyor.
- Öğrenilen engeller sonraki A* planına gerçekten katılıyor; arazi yüksekliği yapay engel yüksekliğiyle bozulmuyor.
- Rota yokken veya sensör verisi bozukken motorlar duruyor. Yakın engelde ileri kaçış yerine yerinde dönüş uygulanıyor; görevde süre, eğilme ve takılma kontrolleri var.
- UDP dinleyici simülatörden önce başlıyor; paketler sürüm/run_id ile doğrulanıyor. Büyük rotalarda hedef korunuyor, kesik paketler kabul edilmiyor. Paylaşılan panel verisi kilit altında okunuyor.
- UI rota yokken sahte düz çizgi üretmiyor; güncel rota, ölçülen hız ve telemetri kesintisi gösteriliyor.
- A* toplam arazi eğimini de denetliyor; köşe kesmiyor, sınır dışı konumu harita içine gizlice taşımıyor ve gerçek çapraz mesafeyi hesaplıyor. Varsayılan admissible sezgi ağırlığı 1; tanımlı grid maliyetinde en düşük maliyetli rota aranıyor.

## Veri gerçeği ve kapsam

Depodaki `SLDEM2015_256_SL_*.LBL` etiketleri **SLOPE** ürünlerini tarif eder: bunlar yükseklik haritası değildir. Etiketlerdeki 0.118451 km/pixel değeri 118.451 m/pixel'dir; 1 m/pixel gibi kullanılamaz. Depoda IMG dosyaları yoktur. Varsayılan demo açıkça **sentetik** olarak etiketlenir, NASA/JAXA arazisi iddiası taşımaz.

`--img` yolu, bitişik aynı isimli `.LBL` etiketi olan gerçek yükselti verisi içindir. Okuyucu boyutları, endian bilgisini, örnek tipini, ölçek/ofseti, birimleri, eksik örnekleri ve dosya boyunu doğrular. Fiziksel model kare yerel metre koordinatları gerektirir: `LOCAL_CARTESIAN` / `LOCAL_ENU` projeksiyonu olarak tanımlanan bir yerel yükselti rasteri kabul edilir. Ham küresel/polar PDS haritası önce gerçek bir yerel metre projeksiyonuna dönüştürülmelidir; yalnızca etiketini yeniden adlandırmak doğru değildir. Dikey sıfırlama ofseti görev metaverisinde saklanır.

Bu sürüm, hafif eğimli sentetik arazi ve statik engeller üzerinde doğrulanan bir **simülasyon MVP'sidir**. Her Ay arazisi, hareketli engel, lastik-regolit etkileşimi veya gerçek rover için fiziksel doğrulama iddiası taşımaz. `slam_mapper` burada GPS ve pusula ile konumlanan bir engel hafızasıdır; tam bir SLAM/konum kestirimi sistemi değildir. A* optimumluğu tanımlanan ayrık maliyet içindir; gerçek enerji tüketiminin global optimumu anlamına gelmez.

## Test ve kanıt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/verify_webots.py
```

Her batch çalıştırma yeni `run_id` üretir. Sonuçlar:

- `data/processed/mission_result.json`: kontrolcünün atomik yazdığı son görev sonucu.
- `logs/validation-<run_id>.json`: varış, süre, sensörler, telemetri ve engel istatistikleri.
- `logs/controller-<run_id>.log`: bounded JSONL durum/pozisyon olayları.
- `logs/panel-<run_id>.png`: gerçek telemetriden görev paneli görüntüsü.
- `logs/acceptance-summary.json`: otomatik Webots kabul testlerinin toplu sonucu.

Başarı: hedefe en fazla 0.65 m mesafe, sonrasında sıfır motor komutu ve ölçülen hızın 0.02 m/s altında olması. Engelli kabul koşusunda ayrıca lidar algılaması ve yeniden planlama beklenir. Ölçümler, önceki çalıştırmadan kalan sonuçla karışmaması için `run_id` ile eşleştirilir. Batch başarısızlığında çıkış kodu sıfır olmaz.

## Eski prototip dosyaları

`Rota/code.py`, görüntü renklerinden rota çizme prototipidir. `csv_to_wbt.py` ve `webots_icin_heights.txt` eski manuel aktarma araçlarıdır. `data/raw/*_detail_map.csv`, `mission_params.txt` ve `logs/slam_updates.log` eski kayıtlar olarak korunur. **Yeni çalıştırma zinciri bunları kullanmaz.** README'nin eski sürümündeki bulunmayan `generate_terrain.py`, `lora_core.py` komutları yerine yukarıdaki başlatıcı kullanılır.

Takım: Mesut Azak · Hüseyin Yılmaz · Yusuf Çağdaş Canpolat · Havva Şahin.

## Teknik referanslar

- [Webots WorldInfo ve ENU](https://cyberbotics.com/doc/reference/worldinfo)
- [Python kontrolcüsü ve runtime.ini](https://cyberbotics.com/doc/guide/using-python)
- [Webots Supervisor doğrulama API'si](https://cyberbotics.com/doc/reference/supervisor)
- [NASA PDS MAP_SCALE tanımı](https://pds.nasa.gov/datastandards/documents/dd/all/current/ch11s103.html)
