import pandas as pd
import datetime as dt
import warnings
import time  # Süre ölçümü için
import numpy as np # Küme sayısı kontrolü için
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- Genel Ayarlar ---
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')
print("Affinity Propagation Script'i başlatıldı...")

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

# --- 4. AFFINITY PROPAGATION MODELİNİ UYGULAMA ---
print("\nAffinity Propagation modeli çalıştırılıyor...")
print("UYARI: Bu işlem 5-10+ dakika sürebilir. Lütfen bekleyin...")

start_time = time.time()
model_ap = AffinityPropagation(damping=0.9, random_state=42)
model_ap.fit(rfm_scaled)
duration = time.time() - start_time
labels = model_ap.labels_
rfm_df['Cluster'] = labels

print("Kümeleme tamamlandı.")

# --- 5. AFFINITY PROPAGATION PERFORMANS METRİKLERİ ---
print("\n--- Affinity Propagation Performans Metrikleri ---")
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
cluster_summary_ap = rfm_df.groupby('Cluster').agg(
    Recency_Avg=('Recency', 'mean'),
    Frequency_Avg=('Frequency', 'mean'),
    Monetary_Avg=('Monetary', 'mean'),
    Count=('CustomerID', 'count')
).sort_values('Count', ascending=False)
print(cluster_summary_ap)

rfm_df.to_csv('customer_segments_ap.csv', index=False)
print("\nSonuçlar 'customer_segments_ap.csv' dosyasına kaydedildi.")
print("Script tamamlandı.")