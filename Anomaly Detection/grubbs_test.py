import pandas as pd
import numpy as np
import scipy.stats as stats

def grubbs_test(data):
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    
    abs_deviation = np.abs(data - mean_val)
    max_deviation = np.max(abs_deviation)
    max_idx = np.argmax(abs_deviation)
    
    G_calculated = max_deviation / std_val
    
    # Kritik Değer (Alpha 0.05)
    t_dist = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_dist))
    G_critical = numerator / denominator
    
    return max_idx, G_calculated, G_critical

filename = 'bank_transactions_data_2.csv'
df = pd.read_csv(filename)

data_column = df['TransactionAmount'].values

print("Grubbs Testi Başlatılıyor (TransactionAmount)...")

idx, G_calc, G_crit = grubbs_test(data_column)

print(f"\nHesaplanan G Değeri: {G_calc:.4f}")
print(f"Kritik G Değeri: {G_crit:.4f}")

if G_calc > G_crit:
    print("\nSONUÇ: Bir aykırı değer (Outlier) tespit edildi!")
    print(f"Satır İndeksi: {idx}")
    print(f"Değer: {data_column[idx]}")
    print(f"Satır Detayı:\n{df.iloc[idx]}")
else:
    print("\nSONUÇ: İstatistiksel olarak anlamlı tek bir aykırı değer bulunamadı.")