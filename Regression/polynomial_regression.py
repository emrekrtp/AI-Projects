import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- VERİ YÜKLEME VE AYIRMA ---
file_path = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_path)
y = df['House_Price']
X = df.drop('House_Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- BİTTİ ---

print("--- Polinomsal Regresyon (Tüm Özellikler, 2. Derece) ---")


degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# --- Eğitim Süresi Ölçümü ---
print(f"{degree}. dereceden polinom özellikleri oluşturuluyor ve model eğitiliyor...")
start_train = time.time()
poly_model.fit(X_train, y_train)
train_time = time.time() - start_train
# --- BİTTİ ---

# --- Tahmin Süresi Ölçümü ---
print("Tahmin yapılıyor...")
start_pred = time.time()
y_pred_poly = poly_model.predict(X_test)
pred_time = time.time() - start_pred
# --- BİTTİ ---

print(f"{degree}. dereceden polinom özellikleri oluşturuluyor ve model eğitiliyor...")
poly_model.fit(X_train, y_train)

# 2. Test Seti Üzerinde Tahmin Yapma
print("Tahmin yapılıyor...")
y_pred_poly = poly_model.predict(X_test)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"\n--- Test Seti Performans Metrikleri ---")
print(f"R-Kare (R2): {r2_poly:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae_poly:.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse_poly:.2f}")

print(f"\n--- Süre Metrikleri ---")
print(f"Eğitim Süresi: {train_time:.6f} saniye")
print(f"Tahmin Süresi: {pred_time:.6f} saniye")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_poly, alpha=0.6)
# Mükemmel tahmin çizgisi (y=x doğrusu)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Mükemmel Tahmin (y=x)')
plt.title(f'Polinomsal Regresyon (Derece {degree}): Gerçek Fiyatlar vs. Tahminler')
plt.xlabel('Gerçek Ev Fiyatı (y_test)')
plt.ylabel('Tahmin Edilen Ev Fiyatı (y_pred)')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_regression.png')
print("\nGrafik 'polynomial_regression.png' olarak kaydedildi.")