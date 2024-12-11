import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(dataset_path, model_output_path):
    """
    Entrena un modelo para clasificar hojas de papa como sanas o enfermas.
    """
    IMAGE_SIZE = 256
    BATCH_SIZE = 32

    # Preparación del dataset
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2  # 80% entrenamiento, 20% validación
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='binary'
    )

    # Crear el modelo
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    # Guardar el modelo entrenado
    model.save(model_output_path)
    print(f"Modelo guardado en {model_output_path}")

if __name__ == "__main__":
    dataset_path = "ruta/a/tu/dataset"
    model_output_path = "potato_model.h5"
    train_model(dataset_path, model_output_path)
