import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

filename = 'bank_transactions_data_2.csv'
df = pd.read_csv(filename)

features = ['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'LoginAttempts']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.8, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

df['Cluster'] = clusters
df['Anormallik_DBSCAN'] = df['Cluster'].apply(lambda x: 'Anormal' if x == -1 else 'Normal (Küme ' + str(x) + ')')

anomalies = df[df['Cluster'] == -1]
print(f"Toplam Anormallik (Gürültü) Sayısı: {len(anomalies)}")
print("\nDBSCAN ile Bulunan Örnek Anormallikler:")
print(anomalies[features].head())

plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x='AccountBalance', y='TransactionAmount', hue='Anormallik_DBSCAN', 
                palette=sns.color_palette("bright", df['Anormallik_DBSCAN'].nunique()))
plt.title('DBSCAN: Hesap Bakiyesi vs İşlem Tutarı Kümeleme')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()