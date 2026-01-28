import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- VERİ YÜKLEME VE AYIRMA (Her dosyada gerekli) ---
file_path = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_path)
y = df['House_Price']
X = df.drop('House_Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- BİTTİ ---


print("--- Basit Lineer Regresyon ---")

# HATA BURADAYDI: 'Square Footage' -> 'Square_Footage' olarak düzeltildi
X_train_simple = X_train[['Square_Footage']]
X_test_simple = X_test[['Square_Footage']]

start_train = time.time()
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
train_time = time.time() - start_train

start_pred = time.time()
y_pred_simple = simple_model.predict(X_test_simple)
pred_time = time.time() - start_pred

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)

y_pred_simple = simple_model.predict(X_test_simple)

r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

print(f"Modelin Katsayısı (b1 - Eğim): {simple_model.coef_[0]:.2f}")
print(f"Modelin Kesişim Noktası (b0): {simple_model.intercept_:.2f}")
print(f"\n--- Test Seti Performans Metrikleri ---")
print(f"R-Kare (R2): {r2_simple:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae_simple:.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse_simple:.2f}")

print(f"\n--- Süre Metrikleri ---")
print(f"Eğitim Süresi: {train_time:.6f} saniye")
print(f"Tahmin Süresi: {pred_time:.6f} saniye")

# Grafiği oluşturma ve kaydetme
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_simple['Square_Footage'], y=y_test, alpha=0.6, label='Gerçek Değerler')
plt.plot(X_test_simple['Square_Footage'], y_pred_simple, color='red', linewidth=2, label='Regresyon Doğrusu (Tahmin)')
plt.title('Basit Lineer Regresyon: Metrekare vs. Ev Fiyatı (Test Seti)')
plt.xlabel('Metrekare (Square_Footage)')
plt.ylabel('Ev Fiyatı (House_Price)')
plt.legend()
plt.grid(True)
plt.savefig('simple_linear_regression.png')
print("Grafik 'simple_linear_regression.png' olarak kaydedildi.")
# plt.show() # Eğer grafiği ekranda görmek isterseniz bu satırı açabilirsiniz