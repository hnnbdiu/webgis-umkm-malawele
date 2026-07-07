import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Memuat dan membersihkan dataset UMKM
df = pd.read_csv('dataset_umkm.csv', sep=',', engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip().str.lower()
df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
df = df.dropna(subset=['lat', 'lon'])

# Menyiapkan matriks fitur koordinat spasial
X = df[['lat', 'lon']].values

# 2. Menghitung WCSS untuk Elbow Method (Menguji K=2 hingga K=10)
wcss = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 3. Menghitung Silhouette Score spesifik untuk K=3 
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_3 = kmeans_3.fit_predict(X)
sil_score = silhouette_score(X, labels_3)

# Mencetak hasil Silhouette Score ke terminal
print("-" * 50)
print(f"HASIL EVALUASI ALGORITMA K-MEANS")
print(f"Total Data UMKM Diproses: {len(X)} titik")
print(f"Nilai Silhouette Score (K=3) : {sil_score:.4f}")
print("-" * 50)

# 4. Memvisualisasikan Grafik Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'bo-', marker='o', linewidth=2, markersize=8)
plt.title('Evaluasi Elbow Method untuk Data UMKM Malawele', fontsize=14)
plt.xlabel('Jumlah Klaster (K)', fontsize=12)
plt.ylabel('Nilai WCSS (Inertia)', fontsize=12)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()