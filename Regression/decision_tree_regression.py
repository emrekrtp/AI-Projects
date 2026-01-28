import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# Hata düzeltmesi: LinearRegression kaldırıldı, DecisionTreeRegressor eklendi
from sklearn.tree import DecisionTreeRegressor 
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

print("--- Karar Ağacı Regresyonu (Decision Tree) ---")

# --- HATA DÜZELTMESİ: Yanlış modele ait süre ölçümü kaldırıldı ---
# LinearRegression() ile ilgili kod blokları silindi.

# 1. Modeli Oluşturma ve Eğitme (Süre ölçümü ile birlikte)
print("Karar Ağacı modeli eğitiliyor...")
start_train = time.perf_counter()
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
train_time = time.perf_counter() - start_train
# --- BİTTİ ---

# 2. Test Seti Üzerinde Tahmin Yapma (Süre ölçümü ile birlikte)
print("Tahmin yapılıyor...")
start_pred = time.perf_counter()
y_pred_tree = tree_model.predict(X_test)
pred_time = time.perf_counter() - start_pred
# --- BİTTİ ---

# 3. Model Performansını Değerlendirme
r2_tree = r2_score(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))

print(f"\n--- Test Seti Performans Metrikleri ---")
print(f"R-Kare (R2): {r2_tree:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae_tree:.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse_tree:.2f}")

print(f"\n--- Süre Metrikleri ---")
# Süreleri daha hassas göstermek için .8f kullanıldı
print(f"Eğitim Süresi: {train_time:.8f} saniye")
print(f"Tahmin Süresi: {pred_time:.8f} saniye")

# 4. Görselleştirme: Gerçek Değerler vs. Tahmin Edilen Değerler
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_tree, alpha=0.6)
# Mükemmel tahmin çizgisi (y=x doğrusu)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Mükemmel Tahmin (y=x)')
plt.title('Karar Ağacı Regresyonu: Gerçek Fiyatlar vs. Tahmin Edilen Fiyatlar')
plt.xlabel('Gerçek Ev Fiyatı (y_test)')
plt.ylabel('Tahmin Edilen Ev Fiyatı (y_pred)')
plt.legend()
plt.grid(True)
plt.savefig('decision_tree_regression.png')
print("\nGrafik 'decision_tree_regression.png' olarak kaydedildi.")