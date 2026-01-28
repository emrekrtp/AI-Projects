model_3 = Sequential([
    Rescaling(1./255, input_shape=(128, 128, 3)),

    # 1. Blok (64 Filtre - Geniş Başlangıç)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 2. Blok (128 Filtre)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 3. Blok (256 Filtre - Yüksek Kapasite)
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Sınıflandırma Kısmı (Geniş Dense)
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Derleme (RMSprop)
model_3.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])