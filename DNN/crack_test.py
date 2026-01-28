import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
import datetime

# --- 0. GPU YAPILANDIRMASI (RTX 3060 İÇİN) ---
# TensorFlow'un tüm VRAM'i baştan bloke etmesini engeller, ihtiyaca göre artırır.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Algılandı ve Yapılandırıldı: {len(gpus)} adet.")
        print(f"   Kullanılan Cihaz: {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ GPU Hatası: {e}")
else:
    print("⚠️ GPU Bulunamadı! CPU modunda çalışılacak.")

# --- 1. AYARLAR VE VERİ SETİ YOLU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = BASE_DIR 

# Klasör kontrolü
if not os.path.exists(os.path.join(DATASET_PATH, "Positive")) or not os.path.exists(os.path.join(DATASET_PATH, "Negative")):
    print("\n❌ HATA: 'Positive' ve 'Negative' klasörleri bu kodun çalıştığı dizinde bulunamadı!")
    print(f"   Aranan yer: {DATASET_PATH}")
    print("   Lütfen kod dosyasını ve klasörleri aynı yere koyduğundan emin ol.")
    exit()

print(f"📂 Veri seti yolu doğrulandı: {DATASET_PATH}")

IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 64  # RTX 3060 güçlü olduğu için batch size artırıldı (Hız artar)
EPOCHS = 10 

# Veri Hazırlığı
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 2. RAPORLAMA FONKSİYONU ---
def log_to_file(model_name, params, metrics):
    with open("rapor.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Model: {model_name} ---\n")
        f.write(f"Tarih: {datetime.datetime.now()}\n")
        f.write(f"Donanım: GPU (RTX 3060)\n")
        f.write(f"Parametreler: {params}\n")
        f.write(f"Test Accuracy: %{metrics['acc']*100:.2f}\n")
        f.write(f"Test Loss: {metrics['loss']:.4f}\n")
        f.write("-" * 30 + "\n\n")
    print(f">> {model_name} sonuçları rapor.txt dosyasına eklendi.")

# --- 3. MODEL OLUŞTURMA FONKSİYONU ---
def create_custom_model(scenario_config):
    model = Sequential()
    
    # Giriş Katmanı
    model.add(Conv2D(scenario_config['filters'][0], (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(2, 2))

    # Ara Katmanlar
    for i in range(1, len(scenario_config['filters'])):
        model.add(Conv2D(scenario_config['filters'][i], (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        
        if scenario_config.get('dropout'):
            model.add(Dropout(0.25))

    model.add(Flatten())
    
    # Dense Katmanları
    for units in scenario_config['dense_units']:
        model.add(Dense(units, activation='relu'))
        if scenario_config.get('dropout'):
            model.add(Dropout(0.5))

    # Çıkış Katmanı
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizasyon Algoritması
    if scenario_config['optimizer'] == 'adam':
        opt = Adam(learning_rate=0.001)
    elif scenario_config['optimizer'] == 'sgd':
        opt = SGD(learning_rate=0.01, momentum=0.9)
    elif scenario_config['optimizer'] == 'rmsprop':
        opt = RMSprop(learning_rate=0.001)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# --- 4. DENEY SENARYOLARI ---
scenarios = [
    {
        "name": "Senaryo 1 (Baz Model)",
        "filters": [32, 64, 128],
        "dense_units": [128],
        "optimizer": "adam",
        "dropout": False,
        "desc": "Standart 3 katmanlı yapı, Adam optimizasyonu."
    },
    {
        "name": "Senaryo 2 (Derin ve Dar - SGD)",
        "filters": [16, 32, 64, 128],
        "dense_units": [64, 32],
        "optimizer": "sgd",
        "dropout": False,
        "desc": "4 katmanlı, SGD optimizer."
    },
    {
        "name": "Senaryo 3 (Geniş ve RMSprop)",
        "filters": [64, 128, 256],
        "dense_units": [256],
        "optimizer": "rmsprop",
        "dropout": False,
        "desc": "Geniş ağ, RMSprop optimizasyonu."
    },
    {
        "name": "Senaryo 4 (Robust - Dropoutlu)",
        "filters": [32, 64, 128, 256],
        "dense_units": [512],
        "optimizer": "adam",
        "dropout": True,
        "desc": "4 katman + Dropout katmanları."
    }
]

# --- 5. EĞİTİM DÖNGÜSÜ ---
print("🚀 Test süreci GPU desteği ile başlıyor...")

# Rapor dosyasını sıfırla/başlık at
with open("rapor.txt", "w", encoding="utf-8") as f:
    f.write("YAPAY ZEKA MODEL KARŞILAŞTIRMA RAPORU\n")
    f.write(f"Donanım: NVIDIA RTX 3060\n")
    f.write("=====================================\n\n")

for scenario in scenarios:
    print(f"\n\n>>> EĞİTİM BAŞLIYOR: {scenario['name']}")
    
    # Her döngüde belleği temizlemek için session reset denebilir ama TF2'de garbage collection genelde yeterlidir.
    tf.keras.backend.clear_session() 
    
    model = create_custom_model(scenario)
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    final_loss = history.history['val_loss'][-1]
    final_acc = history.history['val_accuracy'][-1]
    metrics = {'loss': final_loss, 'acc': final_acc}
    
    log_to_file(scenario['name'], str(scenario), metrics)

print("\n\n✅ Tüm senaryolar tamamlandı. Sonuçlar 'rapor.txt' dosyasında.")