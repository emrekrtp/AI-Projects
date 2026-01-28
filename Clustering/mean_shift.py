import pandas as pd
import datetime as dt
import warnings
import time  # Süre ölçümü için
import numpy as np # Küme sayısı kontrolü ve örnekleme için
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- Genel Ayarlar ---
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')
print("Mean Shift Script'i başlatıldı...")

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

# --- 4. MEAN SHIFT MODELİNİ UYGULAMA ---
print("\nMean Shift modeli için 'bandwidth' (bant genişliği) tahmin ediliyor...")
print("UYARI: Bu işlem 1-2 dakika sürebilir...")
try:
    # İşlemi hızlandırmak için 1000 rastgele örnek al
    n_samples_for_bandwidth = min(1000, len(rfm_scaled))
    indices = np.random.choice(rfm_scaled.shape[0], n_samples_for_bandwidth, replace=False)
    bandwidth_data = rfm_scaled[indices]
    bandwidth = estimate_bandwidth(bandwidth_data, quantile=0.2, n_jobs=-1)
    if bandwidth <= 0.01: bandwidth = 1.0 # Çok küçükse varsayılan ata
    print(f"Tahmin edilen optimal bandwidth: {bandwidth:.3f}")
except Exception:
    bandwidth = 1.0
    print(f"Bandwidth tahmini başarısız. Varsayılan (1.0) kullanılıyor.")

print("Mean Shift modeli çalıştırılıyor (Bu da zaman alabilir)...")
start_time = time.time()
model_ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
model_ms.fit(rfm_scaled)
duration = time.time() - start_time
labels = model_ms.labels_
rfm_df['Cluster'] = labels

print("Kümeleme tamamlandı.")

# --- 5. MEAN SHIFT PERFORMANS METRİKLERİ ---
print("\n--- Mean Shift Performans Metrikleri ---")
print(f"Model Çalışma Süresi: {duration:.2f} saniye")

# Bulunan küme sayısını kontrol et
n_clusters_found = len(np.unique(labels))
print(f"Bulunan Küme Sayısı: {n_clusters_found}")

if n_clusters_found < 2:
    print("Metrik hesaplanamıyor (1 veya daha az küme bulundu).")
else:
    silhouette = silhouette_score(rfm_scaled, labels)
    davies_bouldin = davies_bouldin_score(rfm_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(rfm_scaled, labels)
    
    print(f"Siluet Skoru (İyisi +1'e yakın): {silhouette:.3f}")
    print(f"Davies-Bouldin Endeksi (İyisi 0'a yakın): {davies_bouldin:.3f}")
    print(f"Calinski-Harabasz Endeksi (İyisi yüksek): {calinski_harabasz:.3f}")

# --- 6. KÜME SONUÇLARINI ANALİZ ETME ---
print("\n--- Küme Analizi (Ortalama RFM Değerleri) ---")
cluster_summary_ms = rfm_df.groupby('Cluster').agg(
    Recency_Avg=('Recency', 'mean'),
    Frequency_Avg=('Frequency', 'mean'),
    Monetary_Avg=('Monetary', 'mean'),
    Count=('CustomerID', 'count')
).sort_values('Count', ascending=False)
print(cluster_summary_ms)

rfm_df.to_csv('customer_segments_ms.csv', index=False)
print("\nSonuçlar 'customer_segments_ms.csv' dosyasına kaydedildi.")
print("Script tamamlandı.")