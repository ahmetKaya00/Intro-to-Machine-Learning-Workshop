import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
notlar = np.random.randint(0,101,1000)

print("Ortalaması:", np.mean(notlar))
print("En Yüksek Not:", np.max(notlar))
print("En Düşük Not:", np.min(notlar))
print("Standart Sapma:", np.std(notlar))

gecenler = notlar[notlar >= 50]
kalanlar = notlar[notlar < 50]

print("Geçen Öğrencilerin Sayısı: ", len(gecenler))
print("Kalan Öğrencilerin Sayısı: ", len(kalanlar))

plt.figure(figsize=(10,5))
plt.hist(notlar,bins=10,edgecolor="black",alpha=0.7)
plt.xlabel("Not Araliklari")
plt.ylabel("Ogrenci Sayisi")
plt.title("Ogrenci Nor Dagilimi")
plt.grid(True)
plt.show()

sirali_notlar = np.sort(notlar)

dusuk_dilim = sirali_notlar[:100]
yuksek_dilim = sirali_notlar[-100:]

print("En Düşük %10 Ortalaması: ", np.mean(dusuk_dilim))
print("En Yuksek %10 Ortalaması: ", np.mean(yuksek_dilim))