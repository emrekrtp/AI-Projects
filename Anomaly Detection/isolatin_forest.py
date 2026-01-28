import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

filename = 'bank_transactions_data_2.csv'
df = pd.read_csv(filename)

features = ['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'LoginAttempts']
X = df[features]

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

preds = iso_forest.fit_predict(X)

df['Anomaly_Score'] = iso_forest.decision_function(X) # Skor ne kadar düşükse o kadar anormal
df['Anormallik_IF'] = preds
df['Anormallik_Durumu'] = df['Anormallik_IF'].apply(lambda x: 'Anormal' if x == -1 else 'Normal')

anomalies = df[df['Anormallik_IF'] == -1]
print(f"Toplam Anormallik Sayısı: {len(anomalies)}")
print("\nIsolation Forest ile Bulunan Örnek Anormallikler:")
print(anomalies[features + ['Anomaly_Score']].head())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='TransactionDuration', y='TransactionAmount', hue='Anormallik_Durumu', palette={'Normal': 'blue', 'Anormal': 'red'})
plt.title('Isolation Forest: İşlem Süresi ve Tutarı İlişkisi')
plt.xlabel('İşlem Süresi (Duration)')
plt.ylabel('İşlem Tutarı (Amount)')
plt.show()