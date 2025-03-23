import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv("Mall_Customers.csv")

X = data[['Annual Income (k$)','Spending Score (1-100)']]

print(X.head())

wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss,marker='o')
plt.title("Elbow Yonetimi")
plt.xlabel("Kume Sayisi")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

plt.figure(figsize=(8,6))
colors = ['red','green','blue','cyan','purple']

for i in range(5):
    plt.scatter(X[clusters == i]['Annual Income (k$)'],
                X[clusters == i]['Spending Score (1-100)'],
                c=colors[i], label=f"Kume {i}")

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow',label='Merkezler',marker='X')
plt.xlabel("Yillik Gelir (k$)")
plt.ylabel("Harcama Skoru")
plt.title("Musteri Segmentesyonu - KMeans")
plt.legend()
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8,6))

for i in range(5):
    plt.scatter(data[data['Cluster'] == i]['PCA1'],
                data[data['Cluster'] == i]['PCA2'],
                c=colors[i], label=f"Kume {i}")
plt.title("PCA ile Kume Gorsellestirme")
plt.xlabel("PCA Bileseni 1")
plt.ylabel("PCA Bileseni 2")
plt.legend()
plt.grid(True)
plt.show()