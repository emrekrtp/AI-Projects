import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SINIF TANIMI: YAPAY ARI KOLONİSİ (ABC)
# ---------------------------------------------------------
class ArtificialBeeColony:
    def __init__(self, objective_func, bounds, colony_size=30, n_iter=100, limit=20):
        self.func = objective_func
        self.bounds = bounds  # [min, max] arama aralığı
        self.colony_size = colony_size
        self.n_iter = n_iter
        self.limit = limit    # Kaşif arı (scout) olma limiti
        
        # Koloniyi ikiye böl: Görevli (Employed) ve Gözcü (Onlooker)
        self.n_employed = colony_size // 2
        self.n_onlooker = colony_size // 2
        
        # Başlangıç: Rastgele yiyecek kaynakları (çözümler) üret
        self.foods = np.random.uniform(bounds[0], bounds[1], self.n_employed)
        self.trials = np.zeros(self.n_employed) # Her kaynağın kaç kez denendiği
        
        # İlk en iyiyi bul
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self):
        for it in range(self.n_iter):
            # A. GÖREVLİ ARILAR FAZI (Employed Bees Phase)
            for i in range(self.n_employed):
                self.update_location(i)

            # B. GÖZCÜ ARILAR FAZI (Onlooker Bees Phase)
            # Fitness hesapla (Minimizasyon problemi olduğu için değeri ters çeviriyoruz)
            # Amaç fonksiyonu negatif olduğu için (örn: -100), mutlak değer alıp olasılık hesaplıyoruz.
            fitness_vals = np.array([self.func(f) for f in self.foods])
            
            # Daha düşük değer (daha negatif) daha iyi fitness demektir.
            # Rulet tekerleği için ters orantı kuralım:
            # (Basitlik için: En kötüden farklarını alıp normalize edelim)
            worst = fitness_vals.max()
            diffs = worst - fitness_vals + 1e-5 # +epsilon sıfıra bölmeyi önler
            probs = diffs / np.sum(diffs)
            
            # Gözcüler olasılığa göre kaynak seçer
            for i in range(self.n_onlooker):
                selected_idx = np.random.choice(range(self.n_employed), p=probs)
                self.update_location(selected_idx)

            # C. KAŞİF ARILAR FAZI (Scout Bees Phase)
            # Limiti aşan kaynakları terk et ve yenisini bul
            for i in range(self.n_employed):
                if self.trials[i] > self.limit:
                    self.foods[i] = np.random.uniform(self.bounds[0], self.bounds[1])
                    self.trials[i] = 0
            
            # D. EN İYİYİ KAYDET
            current_vals = [self.func(f) for f in self.foods]
            min_val = min(current_vals)
            if min_val < self.best_fitness:
                self.best_fitness = min_val
                self.best_solution = self.foods[np.argmin(current_vals)]
                
        return self.best_solution, self.best_fitness

    def update_location(self, i):
        # Rastgele bir komşu (k) seç, kendisi olmasın
        k = np.random.randint(0, self.n_employed)
        while k == i: k = np.random.randint(0, self.n_employed)
            
        # Yeni pozisyon formülü: v_i = x_i + phi * (x_i - x_k)
        phi = np.random.uniform(-1, 1)
        new_solution = self.foods[i] + phi * (self.foods[i] - self.foods[k])
        
        # Sınır kontrolü (Fiyat negatif veya aşırı yüksek olmasın)
        new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
        
        # Greedy Selection: Yeni çözüm eskisinden iyiyse kabul et
        if self.func(new_solution) < self.func(self.foods[i]):
            self.foods[i] = new_solution
            self.trials[i] = 0
        else:
            self.trials[i] += 1

# ---------------------------------------------------------
# 2. VERİ HAZIRLIĞI VE MODELLEME
# ---------------------------------------------------------
# Veriyi oku
df = pd.read_csv('retail_price.csv')

# En popüler ürünü seç
top_product = df['product_id'].value_counts().idxmax()
product_df = df[df['product_id'] == top_product]

X_price = product_df['unit_price'].values
y_qty = product_df['qty'].values

# Talep Eğrisi (Polinom Regresyon 2. Derece)
# Qty = a*Price^2 + b*Price + c
coeffs = np.polyfit(X_price, y_qty, 2)
demand_model = np.poly1d(coeffs)

# Amaç Fonksiyonu (Objective Function)
# Geliri Maksimize etmek istiyoruz, Algoritmalar Minimize eder.
# Bu yüzden (-Revenue) döndürüyoruz.
def objective_function(price):
    pred_qty = demand_model(price)
    if pred_qty <= 0: return 1e10 # Satış yoksa çok yüksek ceza
    
    revenue = price * pred_qty
    return -revenue # Minimizasyon için negatif

# ---------------------------------------------------------
# 3. ÇALIŞTIRMA
# ---------------------------------------------------------
# Arama aralığı: Mevcut min fiyatın yarısı ile max fiyatın 1.5 katı arası
bounds = [X_price.min() * 0.5, X_price.max() * 1.5]

# ABC'yi Başlat
abc = ArtificialBeeColony(objective_function, bounds, colony_size=40, n_iter=100)
best_price, best_neg_revenue = abc.optimize()

print(f"--- SONUÇLAR ({top_product}) ---")
print(f"Mevcut Ortalama Fiyat: {X_price.mean():.2f}")
print(f"ABC Önerilen Fiyat   : {best_price:.2f}")
print(f"Beklenen Maks. Gelir : {-best_neg_revenue:.2f}")