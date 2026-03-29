# filepath: LORA/src/data/pds_reader.py
"""
L.O.R.A. — PDS Image Reader (32-bit PC_REAL)
JAXA SLDEM2015 verilerini okur ve kesit çıkarır.
"""

import numpy as np
import os

def extract_lunar_patch(img_path, output_csv, start_row=1000, start_col=5000, patch_size=500):
    # LBL dosyasından alınan sabitler
    TOTAL_LINES = 7680
    TOTAL_SAMPLES = 30720
    
    print(f"[*] '{img_path}' dosyası aranıyor...")
    
    if not os.path.exists(img_path):
        print(f"⚠ HATA: Dosya bulunamadı! Lütfen '{img_path}' yolunu kontrol et.")
        return

    try:
        print("[*] Veri belleğe haritalanıyor (Memory Mapping)...")
        # data_type 'float32' (32-bit PC_REAL)
        raw_data = np.memmap(img_path, dtype='float32', mode='r', 
                             shape=(TOTAL_LINES, TOTAL_SAMPLES))
        
        print(f"[*] {patch_size}x{patch_size} boyutunda kesit alınıyor...")
        patch = raw_data[start_row:start_row+patch_size, start_col:start_col+patch_size]
        
        print("[*] CSV dosyasına yazılıyor (Bu birkaç saniye sürebilir)...")
        np.savetxt(output_csv, patch, fmt="%.4f", delimiter=" ")
        
        print(f"\n✓ BAŞARILI: '{output_csv}' oluşturuldu!")
        print(f"✓ Kesit Aralığı: Satır[{start_row}:{start_row+patch_size}], Sütun[{start_col}:{start_col+patch_size}]")
        
    except Exception as e:
        print(f"\n⚠ BEKLENMEYEN HATA: {e}")

# --- KODUN ÇALIŞMA (TETİKLENME) NOKTASI ---
if __name__ == "__main__":
    print("="*50)
    print(" LORA JAXA Veri İşleyici Başlatıldı")
    print("="*50)
    
    # Resimdeki 921 MB'lık dosyalardan birinin tam adı
    img_file = 'data/raw/SLDEM2015_256_SL_60N_90N_000_120.IMG' 
    output_file = 'data/raw/high_detail_map.csv'
    
    # Çıktı klasörünün var olduğundan emin olalım
    os.makedirs('data/raw', exist_ok=True)
    
    # Fonksiyonu çağırarak işlemi başlatıyoruz
    extract_lunar_patch(img_file, output_file, start_row=1000, start_col=5000, patch_size=500)