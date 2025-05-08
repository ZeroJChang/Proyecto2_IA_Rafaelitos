import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter

#  Ruta del dataset
dataset_path = r'C:\IA\Segundo Proyecto\ALS'

#  Leer imágenes
labels = []
images = []

for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg'):
        letra = filename[0].upper()

        if 'A' <= letra <= 'Z':
            label_encoded = ord(letra) - ord('A')
        else:
            label_encoded = 26  # 'Nothing'

        img = cv2.imread(os.path.join(dataset_path, filename))
        img = cv2.resize(img, (224, 224))  # MobileNetV2 requiere 224x224
        images.append(img)
        labels.append(label_encoded)

#  Mostrar reporte de datos cargados
contador_labels = Counter(labels)
print("\n✅ Reporte de datos cargados:")
print(f"Total de imágenes: {len(images)}")
for etiqueta, cantidad in contador_labels.items():
    nombre = "Nothing" if etiqueta == 26 else chr(etiqueta + ord('A'))
    print(f"Clase '{nombre}': {cantidad} imágenes")

#  Preprocesar
X = np.array(images)
X = preprocess_input(X)  # Preprocesado especial para MobileNetV2
y = np.array(labels)
y = to_categorical(y, num_classes=27)

#  Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

#  Construir el modelo
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Congelar pesos de MobileNet

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(27, activation='softmax')  # 27 clases: A-Z + Nothing
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Entrenar
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stop]
)

#  Guardar modelo
if not os.path.exists('modelo'):
    os.makedirs('modelo')

model.save('modelo/modelo.h5')
print("✅ Modelo MobileNetV2 entrenado y guardado como modelo/modelo.h5")
