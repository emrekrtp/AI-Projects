import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

file_path = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_path)

print("-- Veri Seti ilk 5 Satır --")
print(df.head())
print("\n--- Veri Seti Bilgisi (info) ---")
df.info()

y = df["House_Price"]
x = df.drop('House_Price', axis=1)

X_train, X_test, y_train, y_test, = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(f"\nVeri seti başarıyla ayrıldı:")
print(f"Toplam veri sayısı: {len(df)}")
print(f"Eğitim seti boyutu (X_train): {X_train.shape}")
print(f"Test seti boyutu (X_test): {X_test.shape}")