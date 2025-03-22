from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


print("\nLojistik Regresyon İçin Veri Seti Oluşturuluyor...")

data_classification = {
    "Age":[22,25,47,52,46,56,55,60,62,61],
    "Income":[25000,32000,47000,52000,46000,58000,60000,62000,64000,61000],
    "Purchased":[0,0,1,1,1,1,1,1,1,1]
}

df_classification = pd.DataFrame(data_classification)
print("Lojistik Regresyon Veri Seti:")
print(df_classification.head())

print("\nLojistik Regresyon İçin Veri Eğitim ve Test Setine Arılıyor...")

x_cls = df_classification[["Age","Income"]]
y_cls = df_classification["Purchased"]

x_train_cls,x_test_cls,y_train_cls,y_test_cls = train_test_split(x_cls,y_cls,test_size=0.2,random_state=42)

print(f"Eğitim Seti Boyutu: {x_train_cls.shape}")
print(f"Test Seti Boyutu: {y_test_cls.shape}")

print("\nLojistik Regresyon Modeli Eğitiliyor...")

model_logistic = LogisticRegression()

model_logistic.fit(x_train_cls,y_train_cls)

print("\nLojistik Regresyon Modeli Eğitildi!")
print("Katsayılar (Coefficients):")
print(model_logistic.coef_)
print(f"Intercep (b0): {model_logistic.intercept_}")

print("\nLojistik Regresyon Modeli Test Verisi ile Tahmin Yapıyor...")
y_pred_logistic = model_logistic.predict(x_test_cls)

print("\n Gerçek vs Tahmin Edilen Değerler:")
for gerçek, tahmin in zip(y_test_cls,y_pred_logistic):
    print(f"Gerçek: {gerçek} -> Tahmin: {tahmin}")

print("\nLojistik Regresyon Modeli Modeli Performans Sonuçları...")
accuracy_logistic = accuracy_score(y_test_cls,y_pred_logistic)

print("\n Lojistik Regresyon Modeli Performans Sonuçları:")
print(f"Doğruluk Skoru: {accuracy_logistic:.4f}")

print("\nKNN Modeli Eğitiliyor...")

model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(x_train_cls,y_train_cls)

print("\n KNN Modeli Eğitildi")

print("\nKNN Modeli Test Verisi ile Tahmin Yapıyor...")

y_pred_knn = model_knn.predict(x_test_cls)

print("\n Gerçek vs Tahmin Edilen Değerler:")
for gerçek, tahmin in zip(y_test_cls,y_pred_knn):
    print(f"Gerçek: {gerçek} -> Tahmin: {tahmin}")

print("\nKNN Modeli Performansı Ölçülüyor...")

accuracy_knn = accuracy_score(y_test_cls,y_pred_knn)

print("\nKNN Modeli Performans Sonuçları:")
print(f"Doğruluk Skoru: {accuracy_knn:.4f}")

