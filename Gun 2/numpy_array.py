import numpy as np

liste = [1,2,3,4,5]
numpy_dizi = np.array(liste)
print("Türü:", type(numpy_dizi))

numpy_dizi2 = np.array([10,20,30,40,50])
print("NumPy Dizi:", numpy_dizi2)

dizi_aralik = np.arange(0,10,2)
print("Aralıklı NumPy Dizi:", dizi_aralik)

dizi_sifirlar = np.zeros(5)
print("Sıfırlı NumPy Dizi:", dizi_sifirlar)

dizi_birler = np.ones(5)
print("Birler NumPy Dizi:", dizi_birler)

dizi_linspace = np.linspace(0,20,5)
print("Linespace NumPy Dizi:", dizi_linspace)

dizi_random = np.random.rand(5)
print("Random NumPy Dizi:", dizi_random)

