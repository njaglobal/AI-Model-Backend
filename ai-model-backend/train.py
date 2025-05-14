import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils.supabase import download_images
import shutil
import hashlib

DATA_DIR = "training_data"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")

def compute_dataset_hash(directory):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def train_model():
    # Ensure models dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sync only updated images from Supabase
    print("🔄 Syncing training images from Supabase...")
    updated = download_images()

    # Check if training data is present after sync
    if not os.path.exists(DATA_DIR) or not any(os.scandir(DATA_DIR)):
        print("❌ No training data found after sync.")
        return False

    # Compute current hash of dataset
    new_hash = compute_dataset_hash(DATA_DIR)
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash and not updated:
            print("✅ No new training data found. Skipping retraining.")
            return True

    # Data preprocessing
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

    # Save labels
    class_indices = train_gen.class_indices
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")

    # Build model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

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

    # Save dataset hash
    with open(HASH_PATH, "w") as f:
        f.write(new_hash)

    # Convert to TFLite
    tflite_path = os.path.join(MODEL_DIR, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("✅ Model trained and saved:", h5_path)
    print("✅ Model converted to .tflite:", tflite_path)
    return True
