import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shutil

DATA_DIR = "training_data"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5  # Adjust as needed

def train_model():
    if not os.path.exists(DATA_DIR):
        print("❌ No training data found. Run image download first.")
        return False

    # Ensure models dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Use ImageDataGenerator for loading & preprocessing
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save labels for later use
    class_indices = train_gen.class_indices
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_indices), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    # Save as H5
    h5_path = os.path.join(MODEL_DIR, "model.h5")
    model.save(h5_path)

    print("✅ Model trained and saved:", h5_path)
    return True
