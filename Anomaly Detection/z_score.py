import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

filename = 'bank_transactions_data_2.csv'
try:
    df = pd.read_csv(filename)
    print(f"{filename} başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Hata: {filename} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

df['Z_Score'] = stats.zscore(df['TransactionAmount'])

THRESHOLD = 3
df['Anormallik_ZScore'] = df['Z_Score'].apply(lambda x: 'Anormal' if np.abs(x) > THRESHOLD else 'Normal')

anomalies = df[df['Anormallik_ZScore'] == 'Anormal']
print(f"\nToplam İşlem Sayısı: {len(df)}")
print(f"Tespit Edilen Anormallik Sayısı (Z > {THRESHOLD}): {len(anomalies)}")
print("\nÖrnek Anormallikler:")
print(anomalies[['TransactionID', 'TransactionAmount', 'Z_Score']].head())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df.index, y='TransactionAmount', hue='Anormallik_ZScore', palette={'Normal': 'blue', 'Anormal': 'red'})
plt.title(f'Z-Score Yöntemi ile Anormallik Tespiti (Eşik: {THRESHOLD})')
plt.xlabel('İşlem İndeksi')
plt.ylabel('İşlem Tutarı')
plt.legend()
plt.show()