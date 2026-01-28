import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Süre ölçümü için eklendi

# --- VERİ YÜKLEME VE AYIRMA ---
file_path = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_path)
y = df['House_Price']
X = df.drop('House_Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- BİTTİ ---

print("--- Çoklu Lineer Regresyon (Tüm Özellikler) ---")

# --- Eğitim Süresi Ölçümü ---
# time.time() YERİNE time.perf_counter() KULLANILDI
start_train = time.perf_counter()
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
train_time = time.perf_counter() - start_train
# --- BİTTİ ---

# --- Tahmin Süresi Ölçümü ---
# time.time() YERİNE time.perf_counter() KULLANILDI
start_pred = time.perf_counter()
y_pred_multi = multi_model.predict(X_test)
pred_time = time.perf_counter() - start_pred
# --- BİTTİ ---

r2_multi = r2_score(y_test, y_pred_multi)
mae_multi = mean_absolute_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print(f"\n--- Performans Metrikleri ---")
print(f"R-Kare (R2): {r2_multi:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae_multi:.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse_multi:.2f}")

print(f"\n--- Süre Metrikleri ---")
print(f"Eğitim Süresi: {train_time:.6f} saniye")
print(f"Tahmin Süresi: {pred_time:.6f} saniye")

print("\n--- Model Katsayıları (Özelliklerin Etkisi) ---")
coefficients = pd.DataFrame(multi_model.coef_, X.columns, columns=['Katsayı (Coefficient)'])
print(coefficients.sort_values('Katsayı (Coefficient)', ascending=False))

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_multi, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Mükemmel Tahmin (y=x)')
plt.title('Çoklu Lineer Regresyon: Gerçek Fiyatlar vs. Tahmin Edilen Fiyatlar')
plt.xlabel('Gerçek Ev Fiyatı (y_test)')
plt.ylabel('Tahmin Edilen Ev Fiyatı (y_pred)')
plt.legend()
plt.grid(True)
plt.savefig('multiple_linear_regression.png')
print("\nGrafik 'multiple_linear_regression.png' olarak kaydedildi.")