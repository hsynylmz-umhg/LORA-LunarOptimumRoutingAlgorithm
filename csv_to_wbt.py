dosya_yolu = 'data/raw/high_detail_map.csv'
cikis_yolu = 'webots_icin_heights.txt'

with open(dosya_yolu, 'r') as f:
    # Tüm satırları okuyoruz
    satirlar = f.readlines()

satir_sayisi = len(satirlar)
# İlk satırı boşluklara göre ayırıp kaç tane sayı olduğuna (sütuna) bakıyoruz
sutun_sayisi = len(satirlar[0].split()) if satir_sayisi > 0 else 0

# Tüm satırları birleştirip tek bir uzun metin (string) yapıyoruz
# Satır sonlarındaki boşlukları veya enter karakterlerini temizleyip araya boşluk koyuyoruz
height_str = " ".join([satir.strip() for satir in satirlar])

with open(cikis_yolu, 'w') as out:
    out.write(height_str)

print("İşlem Başarılı! Webots için yükseklik verisi hazırlandı.")
print(f"xDimension (Sütun Sayısı): {sutun_sayisi}")
print(f"yDimension (Satır Sayısı): {satir_sayisi}")