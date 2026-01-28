import pandas as pd
import datetime as dt
import warnings
import time  # Süre ölçümü için
import numpy as np # Küme sayısı kontrolü için
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# --- Genel Ayarlar ---
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')
print("K-Means Script'i başlatıldı...")

# --- 1. VERİ YÜKLEME VE TEMİZLEME ---
try:
    df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    print("HATA: 'OnlineRetail.csv' dosyası bulunamadı.")
    exit()

df_cleaned = df.dropna(subset=['CustomerID'])
df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]
df_cleaned = df_cleaned[df_cleaned['UnitPrice'] > 0]
df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int)
print(f"Veri temizlendi. Analiz için kalan satır sayısı: {len(df_cleaned)}")

# --- 2. RFM METRİKLERİNİ HESAPLAMA ---
print("RFM metrikleri hesaplanıyor...")
df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'], format='%d-%m-%Y %H:%M')
snapshot_date = df_cleaned['InvoiceDate'].max() + dt.timedelta(days=1)
rfm_df = df_cleaned.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

# --- 3. VERİYİ ÖLÇEKLENDİRME (SCALING) ---
print("RFM verisi ölçeklendiriliyor (StandardScaler)...")
rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# --- 4. OPTİMAL KÜME SAYISINI (K) BULMA (ELBOW METODU) ---
print("Optimal K sayısı için Elbow Metodu çalıştırılıyor...")
inertia_list = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia_list.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_list, marker='o', linestyle='--')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Inertia (Hata)')
plt.title('K-Means için Optimal K (Elbow Metodu)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('kmeans_elbow_plot.png')
print("Elbow grafiği 'kmeans_elbow_plot.png' olarak kaydedildi. Lütfen grafiği inceleyin.")

# --- 5. K-MEANS MODELİNİ UYGULAMA ---
# Lütfen 'kmeans_elbow_plot.png' grafiğini inceleyip K değerini güncelleyin.
ideal_k = 4  # <--- BURAYI GRAFİĞE GÖRE GÜNCELLEYİN

print(f"\nK-Means modeli K={ideal_k} ile çalıştırılıyor...")
start_time = time.time()
kmeans_final = KMeans(n_clusters=ideal_k, init='k-means++', n_init=10, random_state=42)
kmeans_final.fit(rfm_scaled)
duration = time.time() - start_time
labels = kmeans_final.labels_
rfm_df['Cluster'] = labels

print("Kümeleme tamamlandı.")

# --- 6. K-MEANS PERFORMANS METRİKLERİ ---
print("\n--- K-Means Performans Metrikleri ---")
print(f"Model Çalışma Süresi: {duration:.2f} saniye")
print(f"Seçilen Küme Sayısı (K): {ideal_k}")

try:
    silhouette = silhouette_score(rfm_scaled, labels)
    davies_bouldin = davies_bouldin_score(rfm_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(rfm_scaled, labels)
    
    print(f"Siluet Skoru (İyisi +1'e yakın): {silhouette:.3f}")
    print(f"Davies-Bouldin Endeksi (İyisi 0'a yakın): {davies_bouldin:.3f}")
    print(f"Calinski-Harabasz Endeksi (İyisi yüksek): {calinski_harabasz:.3f}")

except ValueError as e:
    print(f"Metrik hesaplama hatası (Muhtemelen K=1 seçildi): {e}")

# --- 7. KÜME SONUÇLARINI ANALİZ ETME ---
print("\n--- Küme Analizi (Ortalama RFM Değerleri) ---")
cluster_summary = rfm_df.groupby('Cluster').agg(
    Recency_Avg=('Recency', 'mean'),
    Frequency_Avg=('Frequency', 'mean'),
    Monetary_Avg=('Monetary', 'mean'),
    Count=('CustomerID', 'count')
).sort_values('Count', ascending=False)
print(cluster_summary)

rfm_df.to_csv('customer_segments_kmeans.csv', index=False)
print("\nSonuçlar 'customer_segments_kmeans.csv' dosyasına kaydedildi.")
print("Script tamamlandı.")