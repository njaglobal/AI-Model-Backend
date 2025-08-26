# train.py (ResNet50 teacher + MobileNetV2 student)

import os
import random
import shutil
import hashlib
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from utils.supabase import download_images  # Must return List[str] of new local files

# ========= Reproducibility =========
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ========= Paths & Config =========
DATA_DIR = "training_data"
NEW_ONLY_DIR = "new_data_only"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = 100
HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, "teacher_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
PER_CLASS_CSV = os.path.join(MODEL_DIR, "per_class_metrics.csv")

# ======== Distillation =========
USE_DISTILLATION = True
TEMPERATURE = 2.0
ALPHA = 0.7  # Weight for GT loss; 1-ALPHA for distillation
DISTILL_LR = 1e-4
DISTILL_GRAD_CLIP_NORM = 5.0

# ======== Augmentation =========
AUG_SAVE_PREFIX = "aug_"
AUG_TARGET_CAP = 1000
AUG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

STRONG_AUG_PARAMS = dict(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.25,
    zoom_range=0.35,
    horizontal_flip=True,
    brightness_range=[0.4, 1.6],
    fill_mode="nearest"
)

MODERATE_AUG_PARAMS = dict(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)


# ========= Utilities =========
def compute_dataset_hash(directory: str) -> str:
    """Compute md5 hash of all files in directory."""
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _infer_class_from_path(p: str) -> str:
    """Infer class name from path structure."""
    parts = Path(p).parts
    root_name = Path(DATA_DIR).name
    if root_name in parts:
        i = parts.index(root_name)
        if i + 1 < len(parts):
            return parts[i + 1]
    return Path(p).parent.name


def prepare_new_only_dataset(new_files) -> str:
    """Build dataset with only new images, preserving class folders."""
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
    os.makedirs(NEW_ONLY_DIR, exist_ok=True)

    classes = set()
    for f in new_files:
        cls = _infer_class_from_path(f)
        classes.add(cls)
        dest_dir = Path(NEW_ONLY_DIR) / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest_dir / Path(f).name)

    print("🆕 New-only classes:", sorted(classes))
    return NEW_ONLY_DIR


def build_model(num_classes: int, backbone: str = "resnet50") -> tf.keras.Model:
    """Build a CNN model with specified backbone."""
    backbone = backbone.lower()
    if backbone == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif backbone == "mobilenetv2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_for_finetune(model: tf.keras.Model, fine_tune_at: int = FINE_TUNE_AT):
    """Unfreeze top layers for fine-tuning."""
    backbone = model.layers[0] if isinstance(model, models.Sequential) else model.get_layer(index=0)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= fine_tune_at) and not isinstance(layer, layers.BatchNormalization)
    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])


def balance_and_augment(train_dir: str, target_count: int = None, target_count_cap: int = AUG_TARGET_CAP, strong_when_below: int = 5):
    """Oversample minority classes by augmenting images."""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"{train_dir} not found")

    class_counts = {
        cls: len([p for p in (train_dir / cls).iterdir() if p.suffix.lower() in AUG_EXTS and not p.name.startswith(AUG_SAVE_PREFIX)])
        for cls in os.listdir(train_dir) if (train_dir / cls).is_dir()
    }
    if not class_counts:
        print("⚠️ No classes found")
        return

    max_count = max(class_counts.values())
    target_count = min(target_count or max_count, target_count_cap)
    print("📊 Before balancing:", class_counts, f"Target per class: {target_count}")

    for cls, seed_count in class_counts.items():
        cls_path = train_dir / cls
        total_existing = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
        if total_existing >= target_count:
            continue
        n_to_generate = target_count - total_existing
        aug_params = STRONG_AUG_PARAMS if seed_count <= strong_when_below else MODERATE_AUG_PARAMS
        aug = ImageDataGenerator(**aug_params)
        gen = aug.flow_from_directory(
            directory=str(train_dir),
            classes=[cls],
            target_size=IMG_SIZE,
            batch_size=1,
            save_to_dir=str(cls_path),
            save_prefix=AUG_SAVE_PREFIX,
            save_format="jpg",
            shuffle=True,
            seed=RANDOM_SEED
        )
        for _ in range(n_to_generate):
            try:
                next(gen)
            except Exception as e:
                print(f"⚠️ Augmentation error for {cls}: {e}")
                break

    print("✅ After balancing:", {cls: len(list((train_dir/cls).glob("*"))) for cls in class_counts})


# ========= Distiller =========
class Distiller(tf.keras.Model):
    """Knowledge distillation wrapper."""
    def __init__(self, student, teacher, temperature=3.0, alpha=0.5, teacher_indices=None):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_indices = tf.convert_to_tensor(teacher_indices, dtype=tf.int32) if teacher_indices else None

    def compile(self, optimizer, metrics=None, student_loss_fn=None, distillation_loss_fn=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        if not student_loss_fn or not distillation_loss_fn:
            raise ValueError("Provide student_loss_fn and distillation_loss_fn")
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def _align_teacher_logits(self, teacher_preds):
        if self.teacher_indices is None:
            return teacher_preds
        return tf.gather(teacher_preds, self.teacher_indices, axis=1)

    def train_step(self, data):
        x, y, sample_weight = self._unpack_data(data)
        teacher_logits = self._align_teacher_logits(tf.stop_gradient(self.teacher(x, training=False)))
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_logits, sample_weight)
            t = tf.cast(self.temperature, student_logits.dtype)
            distill_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / t, axis=1),
                tf.nn.softmax(student_logits / t, axis=1)
            ) * (t ** 2)
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, self.student.trainable_variables) if g is not None]
        if grads_vars:
            grads_to_apply, vars_to_apply = zip(*grads_vars)
            if DISTILL_GRAD_CLIP_NORM:
                grads_to_apply, _ = tf.clip_by_global_norm(list(grads_to_apply), DISTILL_GRAD_CLIP_NORM)
            self.optimizer.apply_gradients(zip(grads_to_apply, vars_to_apply))

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(student_logits, axis=1), tf.argmax(y, axis=1)), tf.float32))
        self.compiled_metrics.update_state(y, student_logits, sample_weight)
        return {"loss": total_loss, "student_loss": student_loss, "distillation_loss": distill_loss, "accuracy": acc}

    def test_step(self, data):
        x, y, sample_weight = self._unpack_data(data)
        preds = self.student(x, training=False)
        loss = self.student_loss_fn(y, preds, sample_weight)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))
        self.compiled_metrics.update_state(y, preds, sample_weight)
        return {"val_loss": loss, "val_accuracy": acc}

    def _unpack_data(self, data):
        if isinstance(data, (tuple, list)):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data[0], data[1], data[2]
            return data[0], data[1], None
        return data, None, None


# ========= Callback =========
class SaveBestStudent(Callback):
    """Save best student model during training."""
    def __init__(self, student: tf.keras.Model, best_path: str):
        super().__init__()
        self.student = student
        self.best_path = best_path
        self.best_val_acc = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy", -np.inf)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.student.save(self.best_path)
            print(f"💾 Saved improved student: val_acc={val_acc:.4f}")

# ========= Training =========
def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("🔄 Syncing training images from Supabase...")
    new_files = download_images()

    if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
        print("❌ No training data found.")
        return False

    # Compute dataset hash
    new_hash = compute_dataset_hash(DATA_DIR)
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash and not new_files:
            print("✅ No changes, skipping retrain.")
            return True

    # Choose dataset path
    dataset_path = prepare_new_only_dataset(new_files) if new_files else DATA_DIR
    print(f"📦 Using dataset at: {dataset_path}")

    # Balance and augment
    try:
        balance_and_augment(dataset_path)
    except Exception as e:
        print("⚠️ balance_and_augment() failed:", e)

    # Data generators
    datagen = ImageDataGenerator(
        validation_split=0.3,
        rescale=1./255,
        **MODERATE_AUG_PARAMS
    )
    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
    )

    # Save class labels
    class_indices = train_gen.class_indices
    with open(LABELS_PATH, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"🏷️ Labels saved -> {LABELS_PATH}")

    # Compute class weights
    all_labels = train_gen.classes
    class_weights = dict(enumerate(
        compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    ))

    history1, history2 = None, None
    model = None

    # ---------- Incremental Training with Distillation ----------
    if new_files and os.path.exists(TEACHER_MODEL_PATH) and USE_DISTILLATION:
        model = incremental_distillation(dataset_path, class_indices, class_weights, new_files)
    # ---------- Fallback incremental (no teacher) ----------
    elif new_files and os.path.exists(FINAL_MODEL_PATH):
        model = fallback_incremental(dataset_path, class_weights)
    # ---------- Cold start: train teacher ----------
    else:
        model = cold_start_teacher(DATA_DIR, class_weights)

    # ---------- Save final model & hash ----------
    model.save(FINAL_MODEL_PATH)
    with open(HASH_PATH, "w") as f:
        f.write(new_hash)
    print(f"💾 Saved student as FINAL_MODEL_PATH: {FINAL_MODEL_PATH}")

    # Export TFLite
    export_tflite(FINAL_MODEL_PATH)

    # Plot training curves
    plot_training(history1, history2)

    # Evaluate per-class metrics
    evaluate_model(model, val_gen)

    # Cleanup temp dataset
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
        print("🧹 Cleaned new-only dataset.")

    return True


# ========= Helper functions for modular training =========
def incremental_distillation(dataset_path, class_indices, class_weights, new_files):
    """Incremental training with distillation from teacher."""
    print("♻️ Incremental training with distillation...")
    teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
    teacher.trainable = False

    student = build_model(len(class_indices), backbone="mobilenetv2")
    unfreeze_for_finetune(student)

    # Sample old images
    old_sample_ratio = 0.2
    sampled_old_files = []
    for cls in os.listdir(DATA_DIR):
        cls_path = Path(DATA_DIR) / cls
        if not cls_path.is_dir(): continue
        files = [f for f in cls_path.iterdir() if f.suffix.lower() in AUG_EXTS]
        n_sample = max(1, int(len(files) * old_sample_ratio))
        sampled_old_files.extend(random.sample(files, min(n_sample, len(files))))

    combined_files = new_files + [str(f) for f in sampled_old_files]
    dataset_path = prepare_new_only_dataset(combined_files)
    balance_and_augment(dataset_path)

    # Generators
    datagen = ImageDataGenerator(validation_split=0.3, rescale=1./255, **MODERATE_AUG_PARAMS)
    train_gen = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED)
    val_gen = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED)

    # Align teacher indices
    student_labels = sorted(class_indices, key=lambda x: class_indices[x])
    teacher_labels_path = os.path.join(MODEL_DIR, "labels_full.txt")
    if os.path.exists(teacher_labels_path):
        with open(teacher_labels_path) as f:
            teacher_labels = [ln.strip() for ln in f if ln.strip()]
    else:
        teacher_labels = student_labels
    teacher_indices = [teacher_labels.index(lbl) for lbl in student_labels]

    distiller = Distiller(student, teacher, TEMPERATURE, ALPHA, teacher_indices)
    distiller.compile(
        optimizer=Adam(DISTILL_LR),
        metrics=["accuracy"],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=tf.keras.losses.KLDivergence()
    )
    distiller.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, class_weight=class_weights,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    return student


def fallback_incremental(dataset_path, class_weights):
    """Fallback incremental fine-tuning without teacher."""
    print("♻️ Fallback incremental: fine-tune existing model...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)
    unfreeze_for_finetune(model)
    datagen = ImageDataGenerator(validation_split=0.3, rescale=1./255, **MODERATE_AUG_PARAMS)
    train_gen = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED)
    val_gen = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED)
    model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, class_weight=class_weights,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    return model


def cold_start_teacher(data_dir, class_weights):
    """Train teacher from scratch (ResNet50) and optionally distill to student."""
    print("🚀 Cold start: training teacher...")
    datagen = ImageDataGenerator(validation_split=0.3, rescale=1./255, **MODERATE_AUG_PARAMS)
    train_gen = datagen.flow_from_directory(data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED)
    val_gen = datagen.flow_from_directory(data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED)

    class_indices = train_gen.class_indices
    labels_full_path = os.path.join(MODEL_DIR, "labels_full.txt")
    with open(labels_full_path, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")

    teacher = build_model(len(class_indices), backbone="resnet50")
    teacher.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                class_weight=dict(enumerate(compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes))),
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    unfreeze_for_finetune(teacher)
    teacher.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
                class_weight=dict(enumerate(compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes))),
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    teacher.save(TEACHER_MODEL_PATH)
    print("🎓 Teacher trained and saved")

    # Distill to student
    student = build_model(len(class_indices), backbone="mobilenetv2")
    distiller = Distiller(student, teacher, TEMPERATURE, ALPHA, list(range(len(class_indices))))
    distiller.compile(optimizer=Adam(DISTILL_LR), metrics=["accuracy"],
                      student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                      distillation_loss_fn=tf.keras.losses.KLDivergence())
    distiller.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
                  class_weight=dict(enumerate(compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes))),
                  callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    return student


def export_tflite(model_path):
    """Export Keras model to TFLite."""
    if not os.path.exists(model_path):
        print("⚠️ No model found for TFLite export")
        return
    model = tf.keras.models.load_model(model_path)
    tflite_path = os.path.join(MODEL_DIR, "final_model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"📱 Exported TFLite -> {tflite_path}")


def plot_training(history1, history2=None):
    """Plot training curves."""
    plt.figure(figsize=(10, 4))
    if history1:
        plt.plot(history1.history.get("accuracy", []), label="Train Acc")
        plt.plot(history1.history.get("val_accuracy", []), label="Val Acc")
    if history2:
        plt.plot(history2.history.get("accuracy", []), label="Train Acc (FT)")
        plt.plot(history2.history.get("val_accuracy", []), label="Val Acc (FT)")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def evaluate_model(model, val_gen):
    """Compute per-class metrics and save to CSV."""
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    labels = list(val_gen.class_indices.keys())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.show()

    with open(PER_CLASS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score"])
        for i, cls in enumerate(labels):
            writer.writerow([cls, precision[i], recall[i], f1[i]])
    print(f"📊 Per-class metrics saved -> {PER_CLASS_CSV}")



if __name__ == "__main__":
    ok = train_model()
    print("🎉 Training completed!" if ok else "❌ Training failed.")





# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from utils.supabase import download_images
# import shutil
# import hashlib

# DATA_DIR = "training_data"
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# EPOCHS = 5
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")

# def compute_dataset_hash(directory):
#     hash_md5 = hashlib.md5()
#     for root, _, files in os.walk(directory):
#         for name in sorted(files):
#             path = os.path.join(root, name)
#             with open(path, "rb") as f:
#                 while chunk := f.read(8192):
#                     hash_md5.update(chunk)
#     return hash_md5.hexdigest()

# def train_model():
#     # Ensure models dir exists
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     # Sync only updated images from Supabase
#     print("🔄 Syncing training images from Supabase...")
#     updated = download_images()

#     # Check if training data is present after sync
#     if not os.path.exists(DATA_DIR) or not any(os.scandir(DATA_DIR)):
#         print("❌ No training data found after sync.")
#         return False

#     # Compute current hash of dataset
#     new_hash = compute_dataset_hash(DATA_DIR)
#     if os.path.exists(HASH_PATH):
#         with open(HASH_PATH, "r") as f:
#             old_hash = f.read().strip()
#         if old_hash == new_hash and not updated:
#             print("✅ No new training data found. Skipping retraining.")
#             return True

#     # Data preprocessing
#     datagen = ImageDataGenerator(
#         validation_split=0.2,
#         rescale=1./255,
#         horizontal_flip=True,
#         zoom_range=0.2
#     )

#     train_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )

#     # Save labels
#     class_indices = train_gen.class_indices
#     labels_path = os.path.join(MODEL_DIR, "labels.txt")
#     with open(labels_path, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")

#     # Build model
#     base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(len(class_indices), activation='softmax')
#     ])

#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

#     # Save as H5
#     h5_path = os.path.join(MODEL_DIR, "model.h5")
#     model.save(h5_path)

#     # Save dataset hash
#     with open(HASH_PATH, "w") as f:
#         f.write(new_hash)

#     # Convert to TFLite
#     tflite_path = os.path.join(MODEL_DIR, "model.tflite")
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     with open(tflite_path, "wb") as f:
#         f.write(tflite_model)

#     print("✅ Model trained and saved:", h5_path)
#     print("✅ Model converted to .tflite:", tflite_path)
#     return True

# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from utils.supabase import download_images
# import shutil
# import hashlib

# DATA_DIR = "training_data"
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# EPOCHS = 5
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")

# def compute_dataset_hash(directory):
#     hash_md5 = hashlib.md5()
#     for root, _, files in os.walk(directory):
#         for name in sorted(files):
#             path = os.path.join(root, name)
#             with open(path, "rb") as f:
#                 while chunk := f.read(8192):
#                     hash_md5.update(chunk)
#     return hash_md5.hexdigest()

# def train_model():
#     # Ensure models dir exists
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     # Sync only updated images from Supabase
#     print("🔄 Syncing training images from Supabase...")
#     updated = download_images()

#     # Check if training data is present after sync
#     if not os.path.exists(DATA_DIR) or not any(os.scandir(DATA_DIR)):
#         print("❌ No training data found after sync.")
#         return False

#     # Compute current hash of dataset
#     new_hash = compute_dataset_hash(DATA_DIR)
#     if os.path.exists(HASH_PATH):
#         with open(HASH_PATH, "r") as f:
#             old_hash = f.read().strip()
#         if old_hash == new_hash and not updated:
#             print("✅ No new training data found. Skipping retraining.")
#             return True

#     # Data preprocessing
#     datagen = ImageDataGenerator(
#         validation_split=0.2,
#         rescale=1./255,
#         horizontal_flip=True,
#         zoom_range=0.2
#     )

#     train_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )

#     # Save labels
#     class_indices = train_gen.class_indices
#     labels_path = os.path.join(MODEL_DIR, "labels.txt")
#     with open(labels_path, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")

#     # Build model using ResNet50 instead of MobileNetV2
#     base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(len(class_indices), activation='softmax')
#     ])

#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

#     # Save as H5
#     h5_path = os.path.join(MODEL_DIR, "model.h5")
#     model.save(h5_path)

#     # Save dataset hash
#     with open(HASH_PATH, "w") as f:
#         f.write(new_hash)

#     # Convert to TFLite
#     tflite_path = os.path.join(MODEL_DIR, "model.tflite")
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     with open(tflite_path, "wb") as f:
#         f.write(bytes(tflite_model))

#     print("✅ Model trained and saved:", h5_path)
#     print("✅ Model converted to .tflite:", tflite_path)
#     return True


# if __name__ == "__main__":
#     success = train_model()
#     if success:
#         print("🎉 Training completed successfully!")
#     else:
#         print("❌ Training failed or was skipped.")


# //=================== Final Version ==========================

# import os
# import random
# import shutil
# import hashlib
# from pathlib import Path

# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam

# from utils.supabase import download_images  # must return List[str] of new local files

# # ========= Paths & Config =========
# # DATA_DIR = "training_data_backup" # for testing
# DATA_DIR = "training_data"
# INCREMENTAL_DIR = "incremental_data"   # temporary dataset (new + sampled old)
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# EPOCHS = 5               # Phase 1 epochs
# FINE_TUNE_EPOCHS = 3     # Phase 2 epochs
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
# LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")


# # ========= Utilities =========
# def compute_dataset_hash(directory: str) -> str:
#     """Compute md5 hash of all files to detect changes."""
#     hash_md5 = hashlib.md5()
#     for root, _, files in os.walk(directory):
#         for name in sorted(files):
#             path = os.path.join(root, name)
#             if not os.path.isfile(path):
#                 continue
#             with open(path, "rb") as f:
#                 while True:
#                     chunk = f.read(8192)
#                     if not chunk:
#                         break
#                     hash_md5.update(chunk)
#     return hash_md5.hexdigest()


# def prepare_incremental_dataset(new_files, all_files, ratio_old=0.2) -> str:
#     """
#     Create INCREMENTAL_DIR containing:
#       - ALL newly downloaded files (guaranteed)
#       - A random sample of old files per class (ratio_old)
#     """
#     # Reset temp dir
#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#     os.makedirs(INCREMENTAL_DIR, exist_ok=True)

#     # Bucket files by class name (parent folder)
#     def class_of(p: str) -> str:
#         return Path(p).parent.name

#     # Copy all NEW files
#     for f in new_files:
#         cls = class_of(f)
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy(f, dest_dir / Path(f).name)

#     # Build per-class lists
#     class_to_all = {}
#     for f in all_files:
#         cls = class_of(f)
#         class_to_all.setdefault(cls, []).append(f)

#     # Sample OLD files (exclude new ones)
#     new_set = set(Path(p).resolve() for p in new_files)
#     for cls, files in class_to_all.items():
#         old_files = [str(p) for p in map(Path, files) if p.resolve() not in new_set]
#         if not old_files:
#             continue
#         k = max(1, int(len(old_files) * ratio_old))
#         sampled = random.sample(old_files, min(len(old_files), k))
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         for f in sampled:
#             shutil.copy(f, dest_dir / Path(f).name)

#     return INCREMENTAL_DIR


# # ========= Training =========
# def train_model():
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     # 1) Sync images from Supabase
#     print("🔄 Syncing training images from Supabase...")
#     new_files = download_images()  # must be a list of local file paths

#     # Sanity: ensure dataset exists at all
#     if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
#         print("❌ No training data found after sync. Supabase bucket may be empty.")
#         return False

#     # 2) Hash check (on full dataset)
#     new_hash = compute_dataset_hash(DATA_DIR)
#     if os.path.exists(HASH_PATH):
#         with open(HASH_PATH, "r") as f:
#             old_hash = f.read().strip()
#         if old_hash == new_hash and not new_files:
#             print("✅ No new or changed training data. Skipping retraining.")
#             return True

#     # 3) Build dataset path
#     all_files = [str(p) for p in Path(DATA_DIR).rglob("*.*") if p.is_file()]
#     if new_files:
#         # Incremental: new + sample of old
#         dataset_path = prepare_incremental_dataset(new_files, all_files, ratio_old=0.2)
#         print(f"📦 Using incremental dataset at: {dataset_path}")
#     else:
#         # Fallback: changed hash but no explicit 'new_files' (rare)
#         dataset_path = DATA_DIR
#         print("ℹ️ No explicit new files but dataset changed; training on full DATA_DIR.")

#     # 4) Generators
#     datagen = ImageDataGenerator(
#         validation_split=0.2,
#         rescale=1. / 255,
#         horizontal_flip=True,
#         zoom_range=0.2
#     )
#     train_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training',
#         shuffle=True
#     )
#     val_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False
#     )

#     # ✅ Add here: sanity check counts
#     for cls, idx in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Save labels
    
#     class_indices = train_gen.class_indices
#     with open(LABELS_PATH, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")
#     print(f"🏷️  Labels saved -> {LABELS_PATH}")

#     # 5) Build Model
#     base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     # -------- Phase 1: feature extractor --------
#     base_model.trainable = False
#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(len(class_indices), activation='softmax')
#     ])
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     print("🚀 Phase 1: Training top layers...")
#     history1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

#     # -------- Phase 2: fine-tune last ResNet block --------
#     base_model.trainable = True
#     fine_tune_at = 140  # unfreeze from this layer onward (ResNet50 ~175 layers)
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False
#     model.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     print("🔧 Phase 2: Fine-tuning last ResNet layers...")
#     history2 = model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS)

#     # 6) Save model
#     h5_path = os.path.join(MODEL_DIR, "model.h5")
#     model.save(h5_path)
#     with open(HASH_PATH, "w") as f:
#         f.write(new_hash)
#     print("💾 Saved model:", h5_path)

#     # 7) Convert to TFLite
#     tflite_path = os.path.join(MODEL_DIR, "model.tflite")
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     with open(tflite_path, "wb") as f:
#         f.write(tflite_model)
#     print("📦 Saved TFLite:", tflite_path)

#     # 8) Plot training curves with fine-tune marker
#     acc = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
#     val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
#     loss = history1.history.get('loss', []) + history2.history.get('loss', [])
#     val_loss = history1.history.get('val_loss', []) + history2.history.get('val_loss', [])

#     epochs_total = len(acc)
#     epochs_range = range(1, epochs_total + 1)

#     plt.figure(figsize=(9, 6))
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.plot(epochs_range, loss, label='Training Loss', linestyle="--")
#     plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle="--")

#     # Vertical marker where fine-tuning starts
#     plt.axvline(x=EPOCHS, linestyle=":", label='Fine-tuning Start')

#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy / Loss")
#     plt.title("Training & Validation Performance")
#     plt.legend()
#     plt.grid(True)

#     chart_path = os.path.join(MODEL_DIR, "accuracy.png")
#     plt.savefig(chart_path)
#     plt.close()
#     print("📊 Accuracy chart saved:", chart_path)

#     # 9) Print final accuracies as percentages
#     if acc and val_acc:
#         final_train_acc = acc[-1] * 100
#         final_val_acc = val_acc[-1] * 100
#         print(f"🏆 Final Training Accuracy: {final_train_acc:.2f}%")
#         print(f"🏆 Final Validation Accuracy: {final_val_acc:.2f}%")

#     # 10) Clean up temporary incremental data
#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#         print("🧹 Cleaned up incremental training data.")

#     print("✅ Training pipeline complete.")
#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     if ok:
#         print("🎉 Training completed successfully!")
#     else:
#         print("❌ Training failed or was skipped.")


# import os
# import random
# import shutil
# import hashlib
# from pathlib import Path
# import numpy as np

# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from utils.supabase import download_images  # must return List[str] of new local files

# # ========= Paths & Config =========
# DATA_DIR = "training_data"
# INCREMENTAL_DIR = "incremental_data"   # temporary dataset (new + sampled old)
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 8         # smaller batch for tiny dataset
# EPOCHS = 20            # allow patience/early stopping
# FINE_TUNE_EPOCHS = 10
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
# LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")


# # ========= Utilities =========
# def compute_dataset_hash(directory: str) -> str:
#     """Compute md5 hash of all files to detect changes."""
#     hash_md5 = hashlib.md5()
#     for root, _, files in os.walk(directory):
#         for name in sorted(files):
#             path = os.path.join(root, name)
#             if not os.path.isfile(path):
#                 continue
#             with open(path, "rb") as f:
#                 for chunk in iter(lambda: f.read(8192), b""):
#                     hash_md5.update(chunk)
#     return hash_md5.hexdigest()


# def prepare_incremental_dataset(new_files, all_files, ratio_old=0.2) -> str:
#     """Create INCREMENTAL_DIR with new + sampled old files."""
#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#     os.makedirs(INCREMENTAL_DIR, exist_ok=True)

#     def class_of(p: str) -> str:
#         return Path(p).parent.name

#     # Copy all NEW files
#     for f in new_files:
#         cls = class_of(f)
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy(f, dest_dir / Path(f).name)

#     # Add sampled old
#     class_to_all = {}
#     for f in all_files:
#         cls = class_of(f)
#         class_to_all.setdefault(cls, []).append(f)

#     new_set = set(Path(p).resolve() for p in new_files)
#     for cls, files in class_to_all.items():
#         old_files = [str(p) for p in map(Path, files) if p.resolve() not in new_set]
#         if not old_files:
#             continue
#         k = max(1, int(len(old_files) * ratio_old))
#         sampled = random.sample(old_files, min(len(old_files), k))
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         for f in sampled:
#             shutil.copy(f, dest_dir / Path(f).name)

#     return INCREMENTAL_DIR


# # ========= Training =========
# def train_model():
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     print("🔄 Syncing training images from Supabase...")
#     new_files = download_images()

#     if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
#         print("❌ No training data found.")
#         return False

#     new_hash = compute_dataset_hash(DATA_DIR)
#     if os.path.exists(HASH_PATH):
#         with open(HASH_PATH, "r") as f:
#             old_hash = f.read().strip()
#         if old_hash == new_hash and not new_files:
#             print("✅ No changes, skipping retrain.")
#             return True

#     all_files = [str(p) for p in Path(DATA_DIR).rglob("*.*") if p.is_file()]
#     if new_files:
#         dataset_path = prepare_incremental_dataset(new_files, all_files, ratio_old=0.3)
#         print(f"📦 Using incremental dataset at: {dataset_path}")
#     else:
#         dataset_path = DATA_DIR
#         print("ℹ️ Training on full dataset.")

#     # 🔧 Stronger augmentation
#     datagen = ImageDataGenerator(
#         validation_split=0.25,
#         rescale=1. / 255,
#         rotation_range=25,
#         width_shift_range=0.15,
#         height_shift_range=0.15,
#         shear_range=0.15,
#         zoom_range=0.25,
#         horizontal_flip=True,
#         brightness_range=[0.7, 1.3],
#         fill_mode="nearest"
#     )
#     train_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training',
#         shuffle=True
#     )
#     val_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False
#     )

#     # ✅ Sanity: print class counts
#     for cls, idx in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Save labels
#     class_indices = train_gen.class_indices
#     with open(LABELS_PATH, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")
#     print(f"🏷️ Labels saved -> {LABELS_PATH}")

#     # ✅ Class weights (to fix imbalance)
#     all_labels = train_gen.classes
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.unique(all_labels),
#         y=all_labels
#     )
#     class_weights = dict(enumerate(weights))
#     print("⚖️ Class Weights:", class_weights)

#     # Build model
#     base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False
#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(len(class_indices), activation='softmax')
#     ])
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     # Callbacks
#     ckpt_path = os.path.join(MODEL_DIR, "best_model.h5")
#     callbacks = [
#         EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#         ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True)
#     ]

#     print("🚀 Phase 1: Training top layers...")
#     history1 = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=EPOCHS,
#         class_weight=class_weights,
#         callbacks=callbacks
#     )

#     # Fine-tune last layers
#     base_model.trainable = True
#     fine_tune_at = 140
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False
#     model.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     print("🔧 Phase 2: Fine-tuning ResNet...")
#     history2 = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=FINE_TUNE_EPOCHS,
#         class_weight=class_weights,
#         callbacks=callbacks
#     )

#     # Save final model
#     # h5_path = os.path.join(MODEL_DIR, "model.h5")
#     # model.save(h5_path)
#     # with open(HASH_PATH, "w") as f:
#     #     f.write(new_hash)
#     # print("💾 Saved model:", h5_path)

#     # # Export TFLite
#     # tflite_path = os.path.join(MODEL_DIR, "model.tflite")
#     # converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     # tflite_model = converter.convert()
#     # with open(tflite_path, "wb") as f:
#     #     f.write(tflite_model)
#     # print("📦 Saved TFLite:", tflite_path)

#     # Export TFLite from best_model.h5

#     # Save final model
#     h5_path = os.path.join(MODEL_DIR, "model.h5")
#     model.save(h5_path)
#     with open(HASH_PATH, "w") as f:
#         f.write(new_hash)
#     print("💾 Saved final Keras model:", h5_path)

#     # Export TFLite from BEST model
#     best_h5_path = os.path.join(MODEL_DIR, "best_model.h5")
#     if os.path.exists(best_h5_path):
#         best_model = tf.keras.models.load_model(best_h5_path)
#         tflite_path = os.path.join(MODEL_DIR, "best_model.tflite")
#         converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
#         # Optional: optimize for size / speed
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         # Optional: use float16 quantization
#         converter.target_spec.supported_types = [tf.float16]
#         tflite_model = converter.convert()
#         with open(tflite_path, "wb") as f:
#             f.write(tflite_model)
#         print("📦 Saved TFLite from best_model.h5 ->", tflite_path)
#     else:
#         print("⚠️ best_model.h5 not found, skipping TFLite export.")

#     # Plot history
#     acc = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
#     val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
#     loss = history1.history.get('loss', []) + history2.history.get('loss', [])
#     val_loss = history1.history.get('val_loss', []) + history2.history.get('val_loss', [])

#     plt.figure(figsize=(9, 6))
#     plt.plot(acc, label="Train Acc")
#     plt.plot(val_acc, label="Val Acc")
#     plt.plot(loss, "--", label="Train Loss")
#     plt.plot(val_loss, "--", label="Val Loss")
#     plt.axvline(x=len(history1.history['accuracy']), linestyle=":", label="Fine-tune start")
#     plt.legend()
#     plt.title("Training Performance")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy / Loss")
#     chart_path = os.path.join(MODEL_DIR, "accuracy.png")
#     plt.savefig(chart_path)
#     plt.close()
#     print("📊 Saved training chart:", chart_path)

#     if acc and val_acc:
#         print(f"🏆 Final Train Acc: {acc[-1]*100:.2f}%")
#         print(f"🏆 Final Val Acc: {val_acc[-1]*100:.2f}%")


#     val_gen.reset()
#     y_true = val_gen.classes
#     y_pred_probs = model.predict(val_gen, verbose=0)
#     y_pred = np.argmax(y_pred_probs, axis=1)

#    # Get all class names in the correct order
#     class_names = list(val_gen.class_indices.keys())

#     # Compute confusion matrix with all labels
#     cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

#     plt.figure(figsize=(8, 6))
#     disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca(), colorbar=False)
#     plt.title("Confusion Matrix on Validation Set")
#     cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
#     plt.savefig(cm_path)
#     plt.close()
#     print("📊 Confusion matrix saved ->", cm_path)

#     # Normalized (row-wise percentages)
#     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     print("\nNormalized Confusion Matrix:")
#     print(cm_norm)


#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#         print("🧹 Cleaned incremental dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")


# ========= Knowledge Distillation =========
# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher, temperature=3, alpha=0.5, teacher_indices=None):
#         """
#         teacher_indices: list/1D tensor of column indices to slice teacher logits
#                          to align with student's class order.
#         """
#         super().__init__()
#         self.teacher = teacher
#         self.student = student
#         self.temperature = temperature
#         self.alpha = alpha
#         self.teacher_indices = None
#         if teacher_indices is not None:
#             self.teacher_indices = tf.convert_to_tensor(teacher_indices, dtype=tf.int32)

#     def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
#         super().compile(optimizer=optimizer, metrics=metrics)
#         self.student_loss_fn = student_loss_fn
#         self.distillation_loss_fn = distillation_loss_fn

#     def _align_teacher_logits(self, teacher_preds):
#         if self.teacher_indices is None:
#             return teacher_preds
#         return tf.gather(teacher_preds, self.teacher_indices, axis=1)

#     def train_step(self, data):
#         # Accept (x, y) or (x, y, sample_weight)
#         if isinstance(data, (tuple, list)):
#             if len(data) == 2:
#                 x, y = data
#                 sample_weight = None
#             elif len(data) == 3:
#                 x, y, sample_weight = data
#             else:
#                 # dataset might be nested (x, y) inside a list
#                 x, y = data[0], data[1]
#                 sample_weight = None
#         else:
#             x, y = data
#             sample_weight = None

#         # Ensure teacher outputs are not trainable and aligned
#         teacher_preds = tf.stop_gradient(self.teacher(x, training=False))
#         teacher_preds = self._align_teacher_logits(teacher_preds)

#         with tf.GradientTape() as tape:
#             student_preds = self.student(x, training=True)

#             # Hard (ground-truth) loss (supports sample_weight)
#             student_loss = self.student_loss_fn(y, student_preds, sample_weight=sample_weight)

#             # Soft (distillation) loss — apply temperature scaling and scale by T^2 per Hinton
#             t = self.temperature
#             soft_teacher = tf.nn.softmax(teacher_preds / t, axis=1)
#             soft_student = tf.nn.softmax(student_preds / t, axis=1)
#             distill_loss = self.distillation_loss_fn(soft_teacher, soft_student) * (t ** 2)

#             # Weighted sum
#             loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

#         # Gradients & apply only to student
#         grads = tape.gradient(loss, self.student.trainable_variables)
#         # gradient clipping to stabilize
#         grads, _ = tf.clip_by_global_norm(grads, DISTILL_GRAD_CLIP_NORM)
#         self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

#         # Simple accuracy
#         acc = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(student_preds, axis=1), tf.argmax(y, axis=1)), tf.float32)
#         )

#         # Update compiled metrics if present
#         try:
#             self.compiled_metrics.update_state(y, student_preds, sample_weight=sample_weight)
#             metrics_result = {m.name: float(m.result()) for m in self.metrics}
#         except Exception:
#             metrics_result = {}

#         result = {
#             "loss": float(loss),
#             "student_loss": float(student_loss),
#             "distillation_loss": float(distill_loss),
#             "accuracy": float(acc),
#         }
#         result.update(metrics_result)
#         return result

#     def test_step(self, data):
#         # Accept (x, y) or (x, y, sample_weight)
#         if isinstance(data, (tuple, list)):
#             if len(data) == 2:
#                 x, y = data
#                 sample_weight = None
#             elif len(data) == 3:
#                 x, y, sample_weight = data
#             else:
#                 x, y = data[0], data[1]
#                 sample_weight = None
#         else:
#             x, y = data
#             sample_weight = None

#         preds = self.student(x, training=False)
#         student_loss = self.student_loss_fn(y, preds, sample_weight=sample_weight)
#         acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))

#         # Update metrics
#         try:
#             self.compiled_metrics.update_state(y, preds, sample_weight=sample_weight)
#             metrics_result = {m.name: float(m.result()) for m in self.metrics}
#         except Exception:
#             metrics_result = {}

#         result = {"val_loss": float(student_loss), "val_accuracy": float(acc)}
#         result.update(metrics_result)
#         return result



# # train.py (updated)
# import os
# import random
# import shutil
# import hashlib
# from pathlib import Path
# import csv
# import numpy as np
# import math
# import time

# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, Callback

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
# from utils.supabase import download_images  # must return List[str] of new local files

# # ========= Reproducibility =========
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

# # ========= Paths & Config =========
# DATA_DIR = "training_data"
# NEW_ONLY_DIR = "new_data_only"   # temporary dataset with ONLY new images
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 8
# EPOCHS = 20
# FINE_TUNE_EPOCHS = 10
# FINE_TUNE_AT = 140
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
# LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
# BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
# FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
# PER_CLASS_CSV = os.path.join(MODEL_DIR, "per_class_metrics.csv")

# # ======== Distillation knobs =========
# USE_DISTILLATION = True
# TEMPERATURE = 2.0
# ALPHA = 0.7                 # weight for GT loss; (1-ALPHA) for distillation
# INIT_STUDENT_FROM_TEACHER = True  # if output dims match; otherwise fresh student
# DISTILL_LR = 1e-4
# DISTILL_GRAD_CLIP_NORM = 5.0

# # ======== Augmentation balancing knobs ========
# AUG_SAVE_PREFIX = "aug_"
# AUG_TARGET_CAP = 1000  # hard cap per class to avoid runaway disk growth
# AUG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# # Strong vs moderate augment params (used when generating saved augmented files)
# STRONG_AUG_PARAMS = dict(
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     shear_range=0.25,
#     zoom_range=0.35,
#     horizontal_flip=True,
#     brightness_range=[0.4, 1.6],
#     fill_mode="nearest"
# )

# MODERATE_AUG_PARAMS = dict(
#     rotation_range=25,
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     shear_range=0.15,
#     zoom_range=0.25,
#     horizontal_flip=True,
#     brightness_range=[0.7, 1.3],
#     fill_mode="nearest"
# )


# # --- NEW helper: robust balance_and_augment ---
# def balance_and_augment(train_dir: str, target_count: int = None, target_count_cap: int = AUG_TARGET_CAP,
#                         strong_when_below: int = 5):
#     """
#     Oversample minority classes in `train_dir` by generating augmented images and saving them
#     into the class folders. Uses STRONG_AUG for classes with seed count <= strong_when_below.
#     target_count defaults to max class size (seed originals).
#     """
#     train_dir = Path(train_dir)
#     if not train_dir.exists():
#         raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

#     # Count current seed samples per class (ignore previously generated AUG_SAVE_PREFIX files)
#     class_counts = {}
#     for cls in sorted(os.listdir(train_dir)):
#         cls_path = train_dir / cls
#         if not cls_path.is_dir():
#             continue
#         original_files = [p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS and not p.name.startswith(AUG_SAVE_PREFIX)]
#         class_counts[cls] = len(original_files)

#     if not class_counts:
#         print("⚠️ No class directories found in", train_dir)
#         return

#     max_count = max(class_counts.values())
#     if target_count is None:
#         target_count = max_count

#     # Cap target_count so we don't generate huge numbers accidentally
#     target_count = min(target_count, target_count_cap)

#     print("📊 Class distribution before balancing (seed originals):", class_counts)
#     print(f"🎯 Target per class (capped): {target_count}")

#     for cls, seed_count in class_counts.items():
#         cls_path = train_dir / cls
#         if not cls_path.exists():
#             continue

#         # Count total existing images including previous augmentations
#         total_existing = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
#         if total_existing >= target_count:
#             print(f"ℹ️ Class '{cls}' already has {total_existing} images (>= target {target_count}) - skipping.")
#             continue

#         n_to_generate = target_count - total_existing
#         print(f"🔄 Augmenting '{cls}': need {n_to_generate} more images (seed originals: {seed_count}, existing total: {total_existing})")

#         # Choose augmentation strength
#         aug_params = STRONG_AUG_PARAMS if seed_count <= strong_when_below else MODERATE_AUG_PARAMS
#         aug = ImageDataGenerator(**aug_params)

#         gen = aug.flow_from_directory(
#             directory=str(train_dir),
#             classes=[cls],
#             target_size=(IMG_SIZE[0], IMG_SIZE[1]),
#             batch_size=1,
#             save_to_dir=str(cls_path),
#             save_prefix=AUG_SAVE_PREFIX,
#             save_format="jpg",
#             shuffle=True,
#             seed=RANDOM_SEED
#         )

#         # Generate required augmented images
#         for _ in range(n_to_generate):
#             try:
#                 next(gen)
#             except Exception as e:
#                 print(f"⚠️ Error during augmentation for {cls}: {e}")
#                 break

#     # Final distribution check (count everything including augmented files)
#     balanced_counts = {}
#     for cls in sorted(os.listdir(train_dir)):
#         cls_path = train_dir / cls
#         if not cls_path.is_dir():
#             continue
#         balanced_counts[cls] = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
#     print("✅ Class distribution after balancing:", balanced_counts)


# # ========= Utilities =========
# def compute_dataset_hash(directory: str) -> str:
#     """Compute md5 hash of all files to detect changes."""
#     hash_md5 = hashlib.md5()
#     for root, _, files in os.walk(directory):
#         for name in sorted(files):
#             path = os.path.join(root, name)
#             if not os.path.isfile(path):
#                 continue
#             with open(path, "rb") as f:
#                 for chunk in iter(lambda: f.read(8192), b""):
#                     hash_md5.update(chunk)
#     return hash_md5.hexdigest()


# def _infer_class_from_path(p: str) -> str:
#     """
#     Infer class from a path under training_data/<class>/... or fallback to parent name.
#     """
#     parts = Path(p).parts
#     root_name = Path(DATA_DIR).name
#     if root_name in parts:
#         i = parts.index(root_name)
#         if i + 1 < len(parts):
#             return parts[i + 1]
#     return Path(p).parent.name


# def prepare_new_only_dataset(new_files) -> str:
#     """
#     Build NEW_ONLY_DIR with ONLY the new images, preserving class subfolders.
#     """
#     if os.path.exists(NEW_ONLY_DIR):
#         shutil.rmtree(NEW_ONLY_DIR)
#     os.makedirs(NEW_ONLY_DIR, exist_ok=True)

#     classes = set()
#     for f in new_files:
#         cls = _infer_class_from_path(f)
#         classes.add(cls)
#         dest_dir = Path(NEW_ONLY_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy(f, dest_dir / Path(f).name)

#     print("🆕 New-only classes:", sorted(classes))
#     return NEW_ONLY_DIR


# def build_new_model(num_classes: int) -> tf.keras.Model:
#     base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
#     base_model.trainable = False
#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# def unfreeze_for_finetune(model: tf.keras.Model, fine_tune_at: int = FINE_TUNE_AT):
#     """Unfreeze last layers of the base model for fine-tuning while keeping early layers frozen."""
#     backbone = model.layers[0] if isinstance(model, models.Sequential) else model.get_layer(index=0)
#     if not isinstance(backbone, tf.keras.Model):
#         return
#     backbone.trainable = True
#     for i, layer in enumerate(backbone.layers):
#         layer.trainable = (i >= fine_tune_at) and not isinstance(layer, tf.keras.layers.BatchNormalization)
#     model.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])


# def try_build_student_from_teacher(teacher: tf.keras.Model, num_classes: int) -> tf.keras.Model:
#     """
#     If teacher's last Dense matches `num_classes`, clone weights; otherwise build fresh.
#     """
#     try:
#         last = teacher.layers[-1]
#         units = getattr(last, "units", None)
#         if units == num_classes:
#             student = tf.keras.models.clone_model(teacher)
#             student.set_weights(teacher.get_weights())
#             student.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#             return student
#     except Exception:
#         pass
#     return build_new_model(num_classes)



# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher, temperature=3.0, alpha=0.5, teacher_indices=None):
#         """
#         Distillation wrapper for training a student model with guidance from a teacher.

#         Args:
#             student: trainable Keras Model (student)
#             teacher: frozen Keras Model (teacher)
#             temperature: float, softening factor for distillation
#             alpha: weight for ground-truth (hard) loss (0..1). Distill weight = 1-alpha
#             teacher_indices: optional list/1D tensor of column indices to align teacher logits
#         """
#         super().__init__()
#         self.teacher = teacher
#         self.student = student
#         self.temperature = float(temperature)
#         self.alpha = float(alpha)
#         self.teacher_indices = None
#         if teacher_indices is not None:
#             self.teacher_indices = tf.convert_to_tensor(teacher_indices, dtype=tf.int32)

#     def compile(self, optimizer, metrics=None, student_loss_fn=None, distillation_loss_fn=None):
#         super().compile(optimizer=optimizer, metrics=metrics)
#         if student_loss_fn is None or distillation_loss_fn is None:
#             raise ValueError("student_loss_fn and distillation_loss_fn must be provided to compile().")
#         self.student_loss_fn = student_loss_fn
#         self.distillation_loss_fn = distillation_loss_fn

#     def _align_teacher_logits(self, teacher_preds):
#         if self.teacher_indices is None:
#             return teacher_preds
#         # teacher_preds shape: (batch, teacher_num_classes)
#         # gather along axis=1 to select columns that correspond to student class order
#         return tf.gather(teacher_preds, self.teacher_indices, axis=1)

#     def train_step(self, data):
#         # Accept (x, y) or (x, y, sample_weight) or nested list/tuple
#         if isinstance(data, (tuple, list)):
#             if len(data) == 2:
#                 x, y = data
#                 sample_weight = None
#             elif len(data) == 3:
#                 x, y, sample_weight = data
#             else:
#                 # nested dataset element: try first two elements
#                 x, y = data[0], data[1]
#                 sample_weight = None
#         else:
#             x, y = data
#             sample_weight = None

#         # Teacher forward (frozen)
#         teacher_logits = tf.stop_gradient(self.teacher(x, training=False))
#         teacher_logits = self._align_teacher_logits(teacher_logits)

#         with tf.GradientTape() as tape:
#             student_logits = self.student(x, training=True)

#             # Hard (ground-truth) loss. Allow sample_weight if provided.
#             student_loss = self.student_loss_fn(y, student_logits, sample_weight=sample_weight)

#             # Soft (distillation) loss. Use temperature scaling.
#             t = tf.cast(self.temperature, student_logits.dtype)
#             soft_teacher = tf.nn.softmax(teacher_logits / t, axis=1)
#             soft_student = tf.nn.softmax(student_logits / t, axis=1)
#             distill_loss = self.distillation_loss_fn(soft_teacher, soft_student)

#             # Multiply distillation term by T^2 (Hinton)
#             distill_loss = distill_loss * (t ** 2)

#             total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

#         # Gradients & apply only to student's trainable variables
#         trainable_vars = self.student.trainable_variables
#         grads = tape.gradient(total_loss, trainable_vars)

#         # Filter out None grads (very important)
#         grads_vars = [(g, v) for (g, v) in zip(grads, trainable_vars) if g is not None]

#         if grads_vars:
#             grads_to_apply, vars_to_apply = zip(*grads_vars)
#             # optional clipping
#             if DISTILL_GRAD_CLIP_NORM is not None:
#                 grads_to_apply, _ = tf.clip_by_global_norm(list(grads_to_apply), DISTILL_GRAD_CLIP_NORM)
#             self.optimizer.apply_gradients(zip(grads_to_apply, vars_to_apply))

#         # Simple accuracy on hard labels
#         acc = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(student_logits, axis=1), tf.argmax(y, axis=1)), tf.float32)
#         )

#         # Update compiled metrics (if any)
#         try:
#             # compiled_metrics expects (y_true, y_pred)
#             self.compiled_metrics.update_state(y, student_logits, sample_weight=sample_weight)
#             metrics_result = {m.name: m.result() for m in self.metrics}
#         except Exception:
#             metrics_result = {}

#         # Return metric values as tensors (Keras handles them)
#         result = {
#             "loss": total_loss,
#             "student_loss": student_loss,
#             "distillation_loss": distill_loss,
#             "accuracy": acc,
#         }
#         result.update(metrics_result)
#         return result

#     def test_step(self, data):
#         # Accept (x, y) or (x, y, sample_weight)
#         if isinstance(data, (tuple, list)):
#             if len(data) == 2:
#                 x, y = data
#                 sample_weight = None
#             elif len(data) == 3:
#                 x, y, sample_weight = data
#             else:
#                 x, y = data[0], data[1]
#                 sample_weight = None
#         else:
#             x, y = data
#             sample_weight = None

#         preds = self.student(x, training=False)
#         student_loss = self.student_loss_fn(y, preds, sample_weight=sample_weight)
#         acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))

#         # Update compiled metrics
#         try:
#             self.compiled_metrics.update_state(y, preds, sample_weight=sample_weight)
#             metrics_result = {m.name: m.result() for m in self.metrics}
#         except Exception:
#             metrics_result = {}

#         result = {"val_loss": student_loss, "val_accuracy": acc}
#         result.update(metrics_result)
#         return result

# class SaveBestStudent(Callback):
#     """Save the student (not Distiller) to BEST_MODEL_PATH based on best val_accuracy."""
#     def __init__(self, student: tf.keras.Model, best_path: str):
#         super().__init__()
#         self.student = student
#         self.best_path = best_path
#         self.best_val_acc = -np.inf

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_acc = logs.get("val_accuracy")
#         if val_acc is not None and val_acc > self.best_val_acc:
#             self.best_val_acc = float(val_acc)
#             self.student.save(self.best_path)
#             print(f"\n💾 Saved improved student to {self.best_path} (val_acc={self.best_val_acc:.4f})")


# # ========= Training =========
# def train_model():
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     print("🔄 Syncing training images from Supabase...")
#     new_files = download_images()  # should download any new files and return their local paths

#     if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
#         print("❌ No training data found.")
#         return False

#     # Dataset change detection (on full DATA_DIR)
#     new_hash = compute_dataset_hash(DATA_DIR)
#     if os.path.exists(HASH_PATH):
#         with open(HASH_PATH, "r") as f:
#             old_hash = f.read().strip()
#         if old_hash == new_hash and not new_files:
#             print("✅ No changes, skipping retrain.")
#             return True

#     # Choose dataset path
#     if new_files:
#         dataset_path = prepare_new_only_dataset(new_files)  # 🆕 ONLY new data
#         print(f"📦 Using new-only dataset at: {dataset_path}")
#     else:
#         dataset_path = DATA_DIR
#         print("ℹ️ No new files → Training on full dataset.")

#     # Balance the dataset we will use (new-only or full)
#     try:
#         # Use stronger augmentation for classes with very few seeds
#         balance_and_augment(dataset_path)
#     except Exception as e:
#         print("⚠️ balance_and_augment() failed:", e)
#         # proceed anyway

#     # Data generators (train-time augmentation = moderate augment)
#     val_split = 0.30 if new_files else (0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20)
#     datagen = ImageDataGenerator(
#         validation_split=val_split,
#         rescale=1. / 255,
#         rotation_range=MODERATE_AUG_PARAMS["rotation_range"],
#         width_shift_range=MODERATE_AUG_PARAMS["width_shift_range"],
#         height_shift_range=MODERATE_AUG_PARAMS["height_shift_range"],
#         shear_range=MODERATE_AUG_PARAMS["shear_range"],
#         zoom_range=MODERATE_AUG_PARAMS["zoom_range"],
#         horizontal_flip=MODERATE_AUG_PARAMS["horizontal_flip"],
#         brightness_range=MODERATE_AUG_PARAMS["brightness_range"],
#         fill_mode=MODERATE_AUG_PARAMS["fill_mode"]
#     )
#     train_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training',
#         shuffle=True,
#         seed=RANDOM_SEED
#     )
#     val_gen = datagen.flow_from_directory(
#         dataset_path,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False,
#         seed=RANDOM_SEED
#     )

#     # Class counts (folder-level)
#     for cls, _ in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Labels: write current subset order (overwrite labels.txt to match this model)
#     class_indices = train_gen.class_indices
#     with open(LABELS_PATH, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")
#     print(f"🏷️ Labels (subset) saved -> {LABELS_PATH}")

#     # Class weights (subset)
#     all_labels = train_gen.classes
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.unique(all_labels),
#         y=all_labels
#     )
#     class_weights = dict(enumerate(weights))
#     print("⚖️ Class Weights:", class_weights)

#     history1 = None
#     history2 = None

#     if new_files and os.path.exists(BEST_MODEL_PATH) and USE_DISTILLATION:
#         # ---------- Incremental update with distillation on new-only ----------
#         print("♻️ Loading teacher model for distillation...")
#         teacher = tf.keras.models.load_model(BEST_MODEL_PATH)
#         teacher.trainable = False

#         # Build student sized to the CURRENT SUBSET of classes
#         num_subset_classes = len(class_indices)
#         if INIT_STUDENT_FROM_TEACHER:
#             student = try_build_student_from_teacher(teacher, num_classes=num_subset_classes)
#         else:
#             student = build_new_model(num_classes=num_subset_classes)

#         # Light unfreeze of student tail for adaptation (BNs stay frozen)
#         try:
#             unfreeze_for_finetune(student, fine_tune_at=FINE_TUNE_AT)
#         except Exception:
#             pass

#         # Build mapping from student class order to teacher class order by class name
#         student_labels = [lbl for lbl, _ in sorted(class_indices.items(), key=lambda x: x[1])]

#         # Try to read teacher's full labels saved during a full training run
#         teacher_labels_path_guess = os.path.join(MODEL_DIR, "labels_full.txt")
#         if os.path.exists(teacher_labels_path_guess):
#             with open(teacher_labels_path_guess, "r") as f:
#                 teacher_labels = [ln.strip() for ln in f if ln.strip()]
#         else:
#             # Fallback: assume teacher and student share class names (common in many setups)
#             teacher_labels = student_labels

#         # Map each student label to its index in teacher labels
#         teacher_indices = []
#         for lbl in student_labels:
#             if lbl not in teacher_labels:
#                 raise ValueError(f"Student label '{lbl}' not found in teacher label list. "
#                                  "Please maintain a labels_full.txt with the teacher's label order.")
#             teacher_indices.append(teacher_labels.index(lbl))

#         distiller = Distiller(
#             student=student,
#             teacher=teacher,
#             temperature=TEMPERATURE,
#             alpha=ALPHA,
#             teacher_indices=teacher_indices
#         )
#         distiller.compile(
#             optimizer=Adam(learning_rate=DISTILL_LR),
#             metrics=["accuracy"],
#             student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
#             distillation_loss_fn=tf.keras.losses.KLDivergence(),
#         )

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
#             SaveBestStudent(student, BEST_MODEL_PATH),
#         ]

#         print("🔧 Incremental training (new-only) with knowledge distillation...")

#         # Check number of classes in new-only dataset
#         num_classes_in_train = len(train_gen.class_indices)
#         if num_classes_in_train < 2:
#             print("⚠️ Only one class in dataset — skipping distillation/fine-tune.")
#             model = student  # assign student as the model, even if not trained
#         else:
#             history2 = distiller.fit(
#                 train_gen,
#                 validation_data=val_gen,
#                 epochs=FINE_TUNE_EPOCHS,
#                 class_weight=class_weights,
#                 callbacks=callbacks
#             )
#             model = student

#     elif new_files and os.path.exists(BEST_MODEL_PATH):
#         # ---------- Fallback incremental: simple fine-tune on new-only ----------
#         print("♻️ Loading previous best model for incremental fine-tuning (no distillation)...")
#         # Build a student for subset classes
#         model = build_new_model(num_classes=len(class_indices))
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
#         history2 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         # # --- After training, check a single validation batch predictions for debug ---
#         # try:
#         #     # Map indices to class names
#         #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

#         #     val_gen.reset()
#         #     x_batch, y_batch = next(val_gen)
#         #     preds = model.predict(x_batch)
#         #     pred_classes = np.argmax(preds, axis=1)
#         #     true_classes = np.argmax(y_batch, axis=1)

#         #     for i in range(len(pred_classes)):
#         #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
#         # except Exception as e:
#         #     print("⚠️ Could not run single-batch debug predictions:", e)

#         model.save(BEST_MODEL_PATH)

#     else:
#         # ---------- Cold start on full data ----------
#         print("🚀 Building a new model and training from scratch on full dataset...")
#         # For cold-start we should balance the full dataset (dataset_path == DATA_DIR)
#         try:
#             balance_and_augment(DATA_DIR)
#         except Exception as e:
#             print("⚠️ balance_and_augment(DATA_DIR) failed:", e)

#         # Rebuild generators for full dataset (ensures augmentation we just added is picked up)
#         dataset_path = DATA_DIR
#         datagen_full = ImageDataGenerator(
#             validation_split=0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20,
#             rescale=1./255,
#             rotation_range=MODERATE_AUG_PARAMS["rotation_range"],
#             width_shift_range=MODERATE_AUG_PARAMS["width_shift_range"],
#             height_shift_range=MODERATE_AUG_PARAMS["height_shift_range"],
#             shear_range=MODERATE_AUG_PARAMS["shear_range"],
#             zoom_range=MODERATE_AUG_PARAMS["zoom_range"],
#             horizontal_flip=MODERATE_AUG_PARAMS["horizontal_flip"],
#             brightness_range=MODERATE_AUG_PARAMS["brightness_range"],
#             fill_mode=MODERATE_AUG_PARAMS["fill_mode"]
#         )
#         train_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
#         )
#         val_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
#         )

#         # Save full labels order for future teacher alignment
#         full_class_indices = train_gen.class_indices
#         labels_full_path = os.path.join(MODEL_DIR, "labels_full.txt")
#         with open(labels_full_path, "w") as f:
#             for label, index in sorted(full_class_indices.items(), key=lambda x: x[1]):
#                 f.write(f"{label}\n")
#         print(f"🏷️ Full labels saved -> {labels_full_path}")

#         model = build_new_model(num_classes=len(full_class_indices))
#         callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

#         print("🚀 Phase 1: Training top layers (frozen backbone)...")
#         history1 = model.fit(
#             train_gen, validation_data=val_gen, epochs=EPOCHS,
#             class_weight=dict(enumerate(
#                 compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
#             )),
#             callbacks=callbacks
#         )

#         # # --- After training, check predictions (small debug batch) ---
#         # try:
#         #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
#         #     val_gen.reset()
#         #     x_batch, y_batch = next(val_gen)
#         #     preds = model.predict(x_batch)
#         #     pred_classes = np.argmax(preds, axis=1)
#         #     true_classes = np.argmax(y_batch, axis=1)
#         #     for i in range(len(pred_classes)):
#         #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
#         # except Exception as e:
#         #     print("⚠️ Could not run single-batch debug predictions:", e)

#         print("🔧 Phase 2: Fine-tuning ResNet tail...")
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         history2 = model.fit(
#             train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
#             class_weight=dict(enumerate(
#                 compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
#             )),
#             callbacks=callbacks
#         )

#         # # --- After training, check predictions (small debug batch) ---
#         # try:
#         #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
#         #     val_gen.reset()
#         #     x_batch, y_batch = next(val_gen)
#         #     preds = model.predict(x_batch)
#         #     pred_classes = np.argmax(preds, axis=1)
#         #     true_classes = np.argmax(y_batch, axis=1)
#         #     for i in range(len(pred_classes)):
#         #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
#         # except Exception as e:
#         #     print("⚠️ Could not run single-batch debug predictions:", e)

#         model.save(BEST_MODEL_PATH)

#     # Save final model snapshot
#     model.save(FINAL_MODEL_PATH)
#     with open(HASH_PATH, "w") as f:
#         f.write(new_hash)
#     print("💾 Saved final Keras model:", FINAL_MODEL_PATH)

#     # Export TFLite from BEST model (with optimization)
#     if os.path.exists(BEST_MODEL_PATH):
#         best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
#         tflite_path = os.path.join(MODEL_DIR, "best_model.tflite")
#         converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter.target_spec.supported_types = [tf.float16]
#         tflite_model = converter.convert()
#         with open(tflite_path, "wb") as f:
#             f.write(tflite_model)
#         print("📦 Saved TFLite from best_model.h5 ->", tflite_path)
#     else:
#         print("⚠️ best_model.h5 not found, skipping TFLite export.")

#     # Plot training curves
#     acc, val_acc, loss, val_loss = [], [], [], []
#     if history1 is not None:
#         acc += history1.history.get('accuracy', [])
#         val_acc += history1.history.get('val_accuracy', [])
#         loss += history1.history.get('loss', [])
#         val_loss += history1.history.get('val_loss', [])
#     if history2 is not None:
#         acc += history2.history.get('accuracy', [])
#         val_acc += history2.history.get('val_accuracy', [])
#         loss += history2.history.get('loss', [])
#         val_loss += history2.history.get('val_loss', [])

#     if acc or val_acc:
#         plt.figure(figsize=(9, 6))
#         if acc: plt.plot(acc, label="Train Acc")
#         if val_acc: plt.plot(val_acc, label="Val Acc")
#         if loss: plt.plot(loss, "--", label="Train Loss")
#         if val_loss: plt.plot(val_loss, "--", label="Val Loss")
#         if history1 is not None:
#             plt.axvline(x=len(history1.history.get('accuracy', [])), linestyle=":", label="Fine-tune start")
#         plt.legend(); plt.title("Training Performance"); plt.xlabel("Epoch"); plt.ylabel("Accuracy / Loss")
#         chart_path = os.path.join(MODEL_DIR, "accuracy.png")
#         plt.savefig(chart_path); plt.close()
#         print("📊 Saved training chart:", chart_path)
#         if acc: print(f"🏆 Final Train Acc: {acc[-1]*100:.2f}%")
#         if val_acc: print(f"🏆 Final Val Acc: {val_acc[-1]*100:.2f}%")

#     # Confusion Matrix on validation split + per-class metrics
#     try:
#         val_gen.reset()
#         steps = len(val_gen) if len(val_gen) > 0 else 1
#         y_true = val_gen.classes
#         # Use predict with explicit steps to ensure we get predictions for all validation samples
#         y_pred_probs = model.predict(val_gen, steps=steps, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)

#         # Confusion matrix (counts)
#         cm = confusion_matrix(y_true, y_pred)
#         labels = list(val_gen.class_indices.keys())

#         # Safe normalization: avoid divide-by-zero for rows with sum 0
#         row_sums = cm.sum(axis=1, keepdims=True).astype(float)
#         row_sums[row_sums == 0] = 1.0
#         cm_norm = cm.astype(float) / row_sums

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#         plt.figure(figsize=(8, 6))
#         disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
#         plt.title("Confusion Matrix (counts)")
#         plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
#         plt.close()

#         # Normalized image
#         plt.figure(figsize=(8, 6))
#         disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
#         disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca())
#         plt.title("Confusion Matrix (normalized by true class)")
#         plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_norm.png"))
#         plt.close()

#         # Per-class precision/recall/f1/accuracy
#         precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
#         per_class_accuracy = []
#         for i in range(len(labels)):
#             total = cm[i].sum()
#             correct = cm[i, i]
#             acc_val = (correct / total) if total > 0 else 0.0
#             per_class_accuracy.append(acc_val)

#         # Print and save per-class metrics
#         print("\nPer-class metrics:")
#         header = ["class", "precision", "recall", "f1", "support", "accuracy"]
#         for i, lab in enumerate(labels):
#             print(f"{lab:15s}  prec={precision[i]:.3f}  rec={recall[i]:.3f}  f1={f1[i]:.3f}  sup={int(support[i])}  acc={per_class_accuracy[i]:.3f}")

#         # Save CSV
#         with open(PER_CLASS_CSV, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(header)
#             for i, lab in enumerate(labels):
#                 writer.writerow([lab, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", int(support[i]), f"{per_class_accuracy[i]:.4f}"])
#         print(f"📊 Per-class metrics saved -> {PER_CLASS_CSV}")

#         np.set_printoptions(precision=3, suppress=True)
#         print("\nNormalized Confusion Matrix:\n", cm_norm)
#     except Exception as e:
#         print("⚠️ Could not compute confusion matrix / per-class metrics:", e)

#     # Cleanup temp dataset
#     if os.path.exists(NEW_ONLY_DIR):
#         shutil.rmtree(NEW_ONLY_DIR)
#         print("🧹 Cleaned new-only dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")


# train.py (ResNet50 teacher + MobileNetV2 student for predictions)

import os
import random
import shutil
import hashlib
from pathlib import Path
import csv
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from utils.supabase import download_images  # must return List[str] of new local files

# ========= Reproducibility =========
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ========= Paths & Config =========
DATA_DIR = "training_data"
NEW_ONLY_DIR = "new_data_only"   # temporary dataset with ONLY new images
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = 100
HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, "teacher_model.h5")
FINAL_MODEL_PATH   = os.path.join(MODEL_DIR, "final_model.h5")
PER_CLASS_CSV = os.path.join(MODEL_DIR, "per_class_metrics.csv")

# ======== Distillation knobs =========
USE_DISTILLATION = True
TEMPERATURE = 2.0
ALPHA = 0.7                 # weight for GT loss; (1-ALPHA) for distillation
INIT_STUDENT_FROM_TEACHER = True  # if output dims match; otherwise fresh student
DISTILL_LR = 1e-4
DISTILL_GRAD_CLIP_NORM = 5.0

# ======== Augmentation balancing knobs ========
AUG_SAVE_PREFIX = "aug_"
AUG_TARGET_CAP = 1000  # hard cap per class to avoid runaway disk growth
AUG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Strong vs moderate augment params (used when generating saved augmented files)
STRONG_AUG_PARAMS = dict(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.25,
    zoom_range=0.35,
    horizontal_flip=True,
    brightness_range=[0.4, 1.6],
    fill_mode="nearest"
)

MODERATE_AUG_PARAMS = dict(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)


# --- NEW helper: robust balance_and_augment ---
def balance_and_augment(train_dir: str, target_count: int = None, target_count_cap: int = AUG_TARGET_CAP,
                        strong_when_below: int = 5):
    """
    Oversample minority classes in `train_dir` by generating augmented images and saving them
    into the class folders. Uses STRONG_AUG for classes with seed count <= strong_when_below.
    target_count defaults to max class size (seed originals).
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

    # Count current seed samples per class (ignore previously generated AUG_SAVE_PREFIX files)
    class_counts = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = train_dir / cls
        if not cls_path.is_dir():
            continue
        original_files = [p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS and not p.name.startswith(AUG_SAVE_PREFIX)]
        class_counts[cls] = len(original_files)

    if not class_counts:
        print("⚠️ No class directories found in", train_dir)
        return

    max_count = max(class_counts.values())
    if target_count is None:
        target_count = max_count

    # Cap target_count so we don't generate huge numbers accidentally
    target_count = min(target_count, target_count_cap)

    print("📊 Class distribution before balancing (seed originals):", class_counts)
    print(f"🎯 Target per class (capped): {target_count}")

    for cls, seed_count in class_counts.items():
        cls_path = train_dir / cls
        if not cls_path.exists():
            continue

        # Count total existing images including previous augmentations
        total_existing = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
        if total_existing >= target_count:
            print(f"ℹ️ Class '{cls}' already has {total_existing} images (>= target {target_count}) - skipping.")
            continue

        n_to_generate = target_count - total_existing
        print(f"🔄 Augmenting '{cls}': need {n_to_generate} more images (seed originals: {seed_count}, existing total: {total_existing})")

        # Choose augmentation strength
        aug_params = STRONG_AUG_PARAMS if seed_count <= strong_when_below else MODERATE_AUG_PARAMS
        aug = ImageDataGenerator(**aug_params)

        gen = aug.flow_from_directory(
            directory=str(train_dir),
            classes=[cls],
            target_size=(IMG_SIZE[0], IMG_SIZE[1]),
            batch_size=1,
            save_to_dir=str(cls_path),
            save_prefix=AUG_SAVE_PREFIX,
            save_format="jpg",
            shuffle=True,
            seed=RANDOM_SEED
        )

        # Generate required augmented images
        for _ in range(n_to_generate):
            try:
                next(gen)
            except Exception as e:
                print(f"⚠️ Error during augmentation for {cls}: {e}")
                break

    # Final distribution check (count everything including augmented files)
    balanced_counts = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = train_dir / cls
        if not cls_path.is_dir():
            continue
        balanced_counts[cls] = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
    print("✅ Class distribution after balancing:", balanced_counts)


# ========= Utilities =========
def compute_dataset_hash(directory: str) -> str:
    """Compute md5 hash of all files to detect changes."""
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _infer_class_from_path(p: str) -> str:
    """
    Infer class from a path under training_data/<class>/... or fallback to parent name.
    """
    parts = Path(p).parts
    root_name = Path(DATA_DIR).name
    if root_name in parts:
        i = parts.index(root_name)
        if i + 1 < len(parts):
            return parts[i + 1]
    return Path(p).parent.name


def prepare_new_only_dataset(new_files) -> str:
    """
    Build NEW_ONLY_DIR with ONLY the new images, preserving class subfolders.
    """
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
    os.makedirs(NEW_ONLY_DIR, exist_ok=True)

    classes = set()
    for f in new_files:
        cls = _infer_class_from_path(f)
        classes.add(cls)
        dest_dir = Path(NEW_ONLY_DIR) / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest_dir / Path(f).name)

    print("🆕 New-only classes:", sorted(classes))
    return NEW_ONLY_DIR


def build_new_model(num_classes: int,  backbone: str = "resnet50") -> tf.keras.Model:
    if backbone.lower() == "resnet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif backbone.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def unfreeze_for_finetune(model: tf.keras.Model, fine_tune_at: int = FINE_TUNE_AT):
    """Unfreeze last layers of the base model for fine-tuning while keeping early layers frozen."""
    backbone = model.layers[0] if isinstance(model, models.Sequential) else model.get_layer(index=0)
    if not isinstance(backbone, tf.keras.Model):
        return
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= fine_tune_at) and not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.5, teacher_indices=None):
        """
        Distillation wrapper for training a student model with guidance from a teacher.

        Args:
            student: trainable Keras Model (student)
            teacher: frozen Keras Model (teacher)
            temperature: float, softening factor for distillation
            alpha: weight for ground-truth (hard) loss (0..1). Distill weight = 1-alpha
            teacher_indices: optional list/1D tensor of column indices to align teacher logits
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_indices = None
        if teacher_indices is not None:
            self.teacher_indices = tf.convert_to_tensor(teacher_indices, dtype=tf.int32)

    def compile(self, optimizer, metrics=None, student_loss_fn=None, distillation_loss_fn=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        if student_loss_fn is None or distillation_loss_fn is None:
            raise ValueError("student_loss_fn and distillation_loss_fn must be provided to compile().")
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def _align_teacher_logits(self, teacher_preds):
        if self.teacher_indices is None:
            return teacher_preds
        # teacher_preds shape: (batch, teacher_num_classes)
        # gather along axis=1 to select columns that correspond to student class order
        return tf.gather(teacher_preds, self.teacher_indices, axis=1)

    def train_step(self, data):
        # Accept (x, y) or (x, y, sample_weight) or nested list/tuple
        if isinstance(data, (tuple, list)):
            if len(data) == 2:
                x, y = data
                sample_weight = None
            elif len(data) == 3:
                x, y, sample_weight = data
            else:
                # nested dataset element: try first two elements
                x, y = data[0], data[1]
                sample_weight = None
        else:
            x, y = data
            sample_weight = None

        # Teacher forward (frozen)
        teacher_logits = tf.stop_gradient(self.teacher(x, training=False))
        teacher_logits = self._align_teacher_logits(teacher_logits)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            # Hard (ground-truth) loss. Allow sample_weight if provided.
            student_loss = self.student_loss_fn(y, student_logits, sample_weight=sample_weight)

            # Soft (distillation) loss. Use temperature scaling.
            t = tf.cast(self.temperature, student_logits.dtype)
            soft_teacher = tf.nn.softmax(teacher_logits / t, axis=1)
            soft_student = tf.nn.softmax(student_logits / t, axis=1)
            distill_loss = self.distillation_loss_fn(soft_teacher, soft_student)

            # Multiply distillation term by T^2 (Hinton)
            distill_loss = distill_loss * (t ** 2)

            total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

        # Gradients & apply only to student's trainable variables
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)

        # Filter out None grads (very important)
        grads_vars = [(g, v) for (g, v) in zip(grads, trainable_vars) if g is not None]

        if grads_vars:
            grads_to_apply, vars_to_apply = zip(*grads_vars)
            # optional clipping
            if DISTILL_GRAD_CLIP_NORM is not None:
                grads_to_apply, _ = tf.clip_by_global_norm(list(grads_to_apply), DISTILL_GRAD_CLIP_NORM)
            self.optimizer.apply_gradients(zip(grads_to_apply, vars_to_apply))

        # Simple accuracy on hard labels
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(student_logits, axis=1), tf.argmax(y, axis=1)), tf.float32)
        )

        # Update compiled metrics (if any)
        try:
            # compiled_metrics expects (y_true, y_pred)
            self.compiled_metrics.update_state(y, student_logits, sample_weight=sample_weight)
            metrics_result = {m.name: m.result() for m in self.metrics}
        except Exception:
            metrics_result = {}

        # Return metric values as tensors (Keras handles them)
        result = {
            "loss": total_loss,
            "student_loss": student_loss,
            "distillation_loss": distill_loss,
            "accuracy": acc,
        }
        result.update(metrics_result)
        return result

    def test_step(self, data):
        # Accept (x, y) or (x, y, sample_weight)
        if isinstance(data, (tuple, list)):
            if len(data) == 2:
                x, y = data
                sample_weight = None
            elif len(data) == 3:
                x, y, sample_weight = data
            else:
                x, y = data[0], data[1]
                sample_weight = None
        else:
            x, y = data
            sample_weight = None

        preds = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, preds, sample_weight=sample_weight)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))

        # Update compiled metrics
        try:
            self.compiled_metrics.update_state(y, preds, sample_weight=sample_weight)
            metrics_result = {m.name: m.result() for m in self.metrics}
        except Exception:
            metrics_result = {}

        result = {"val_loss": student_loss, "val_accuracy": acc}
        result.update(metrics_result)
        return result

class SaveBestStudent(Callback):
    """Save the student (not Distiller) to BEST_MODEL_PATH based on best val_accuracy."""
    def __init__(self, student: tf.keras.Model, best_path: str):
        super().__init__()
        self.student = student
        self.best_path = best_path
        self.best_val_acc = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = float(val_acc)
            self.student.save(self.best_path)
            print(f"\n💾 Saved improved student to {self.best_path} (val_acc={self.best_val_acc:.4f})")


# ========= Training =========
def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("🔄 Syncing training images from Supabase...")
    new_files = download_images()

    if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
        print("❌ No training data found.")
        return False

    new_hash = compute_dataset_hash(DATA_DIR)
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash and not new_files:
            print("✅ No changes, skipping retrain.")
            return True

    # choose dataset path
    if new_files:
        dataset_path = prepare_new_only_dataset(new_files)
        print(f"📦 Using new-only dataset at: {dataset_path}")
    else:
        dataset_path = DATA_DIR
        print("ℹ️ No new files → Training on full dataset.")

    # balance
    try:
        balance_and_augment(dataset_path)
    except Exception as e:
        print("⚠️ balance_and_augment() failed:", e)

    # data generators
    val_split = 0.30 if new_files else (0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20)
    datagen = ImageDataGenerator(
        validation_split=val_split,
        rescale=1. / 255,
        rotation_range=MODERATE_AUG_PARAMS["rotation_range"],
        width_shift_range=MODERATE_AUG_PARAMS["width_shift_range"],
        height_shift_range=MODERATE_AUG_PARAMS["height_shift_range"],
        shear_range=MODERATE_AUG_PARAMS["shear_range"],
        zoom_range=MODERATE_AUG_PARAMS["zoom_range"],
        horizontal_flip=MODERATE_AUG_PARAMS["horizontal_flip"],
        brightness_range=MODERATE_AUG_PARAMS["brightness_range"],
        fill_mode=MODERATE_AUG_PARAMS["fill_mode"]
    )
    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
    )

    # labels
    class_indices = train_gen.class_indices
    with open(LABELS_PATH, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"🏷️ Labels (subset) saved -> {LABELS_PATH}")

    all_labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights = dict(enumerate(weights))

    history1, history2 = None, None
    model = None

    # ---------- INCREMENTAL (teacher exists) ----------
    if new_files and os.path.exists(TEACHER_MODEL_PATH) and USE_DISTILLATION:
        print("♻️ Incremental training with distillation (mixing old + new)...")
        
        teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
        teacher.trainable = False

        # Build student model
        student = build_new_model(num_classes=len(class_indices), backbone="mobilenetv2")
        unfreeze_for_finetune(student, fine_tune_at=FINE_TUNE_AT)

        # ---------------- Mix old + new ----------------
        # Sample a small fraction of old images from DATA_DIR
        old_sample_ratio = 0.2  # 20% of old data
        sampled_old_files = []

        for cls in os.listdir(DATA_DIR):
            cls_path = Path(DATA_DIR) / cls
            if not cls_path.is_dir(): continue
            files = [f for f in cls_path.iterdir() if f.suffix.lower() in AUG_EXTS]
            n_sample = max(1, int(len(files) * old_sample_ratio))
            sampled_old_files.extend(random.sample(files, min(n_sample, len(files))))

        # Combine with new files
        combined_files = new_files + [str(f) for f in sampled_old_files]

        # Prepare temp dataset with only combined files
        dataset_path = prepare_new_only_dataset(combined_files)
        balance_and_augment(dataset_path)  # optional augmentation

        # Data generators
        datagen = ImageDataGenerator(
            validation_split=0.3,
            rescale=1./255,
            rotation_range=MODERATE_AUG_PARAMS["rotation_range"],
            width_shift_range=MODERATE_AUG_PARAMS["width_shift_range"],
            height_shift_range=MODERATE_AUG_PARAMS["height_shift_range"],
            shear_range=MODERATE_AUG_PARAMS["shear_range"],
            zoom_range=MODERATE_AUG_PARAMS["zoom_range"],
            horizontal_flip=MODERATE_AUG_PARAMS["horizontal_flip"],
            brightness_range=MODERATE_AUG_PARAMS["brightness_range"],
            fill_mode=MODERATE_AUG_PARAMS["fill_mode"]
        )
        train_gen = datagen.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
        )
        val_gen = datagen.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
        )

        # ---------------- Distillation ----------------
        student_labels = [lbl for lbl, _ in sorted(class_indices.items(), key=lambda x: x[1])]
        teacher_labels_path = os.path.join(MODEL_DIR, "labels_full.txt")
        if os.path.exists(teacher_labels_path):
            with open(teacher_labels_path, "r") as f:
                teacher_labels = [ln.strip() for ln in f if ln.strip()]
        else:
            teacher_labels = student_labels
        teacher_indices = [teacher_labels.index(lbl) for lbl in student_labels]

        distiller = Distiller(student=student, teacher=teacher,
                            temperature=TEMPERATURE, alpha=ALPHA,
                            teacher_indices=teacher_indices)
        distiller.compile(
            optimizer=Adam(learning_rate=DISTILL_LR),
            metrics=["accuracy"],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence()
        )

        history2 = distiller.fit(
            train_gen, validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )
        model = student

        # Cleanup temp dataset
        if os.path.exists(NEW_ONLY_DIR):
            shutil.rmtree(NEW_ONLY_DIR)

    # ---------- FALLBACK incremental ----------
    elif new_files and os.path.exists(FINAL_MODEL_PATH):
        # ---------------- Fallback incremental (no teacher) ----------------
        print("♻️ Fallback incremental: fine-tune student only...")
        model = tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)
        unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)

        # Optional: balance/augment new files for better training
        # balance_and_augment(dataset_path)

        history2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )

        model.save(FINAL_MODEL_PATH)
        print(f"✅ Fine-tuned student saved to {FINAL_MODEL_PATH}")

    else:
        # ---------------- Cold start (train teacher) ----------------
        print("🚀 Cold start: training teacher (ResNet50)...")

        # Do NOT augment full dataset — teacher sees real images only
        datagen_full = ImageDataGenerator(
            validation_split=0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20,
            rescale=1./255,
            rotation_range=MODERATE_AUG_PARAMS["rotation_range"],
            width_shift_range=MODERATE_AUG_PARAMS["width_shift_range"],
            height_shift_range=MODERATE_AUG_PARAMS["height_shift_range"],
            shear_range=MODERATE_AUG_PARAMS["shear_range"],
            zoom_range=MODERATE_AUG_PARAMS["zoom_range"],
            horizontal_flip=MODERATE_AUG_PARAMS["horizontal_flip"],
            brightness_range=MODERATE_AUG_PARAMS["brightness_range"],
            fill_mode=MODERATE_AUG_PARAMS["fill_mode"]
        )
        train_gen = datagen_full.flow_from_directory(
            DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
        )
        val_gen = datagen_full.flow_from_directory(
            DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
        )

        # Save full labels
        full_class_indices = train_gen.class_indices
        labels_full_path = os.path.join(MODEL_DIR, "labels_full.txt")
        with open(labels_full_path, "w") as f:
            for label, index in sorted(full_class_indices.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")

        # Train teacher
        teacher = build_new_model(num_classes=len(full_class_indices), backbone="resnet50")
        history1 = teacher.fit(
            train_gen, validation_data=val_gen,
            epochs=EPOCHS,
            class_weight=dict(enumerate(
                compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
            )),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )

        # Optional fine-tune teacher
        unfreeze_for_finetune(teacher, fine_tune_at=FINE_TUNE_AT)
        history2 = teacher.fit(
            train_gen, validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=dict(enumerate(
                compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
            )),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )
        teacher.save(TEACHER_MODEL_PATH)

        # Distill immediately into student
        print("🎓 Distilling into student (MobileNetV2)...")
        student = build_new_model(num_classes=len(full_class_indices), backbone="mobilenetv2")
        distiller = Distiller(student=student, teacher=teacher,
                            temperature=TEMPERATURE, alpha=ALPHA,
                            teacher_indices=list(range(len(full_class_indices))))
        distiller.compile(
            optimizer=Adam(learning_rate=DISTILL_LR),
            metrics=["accuracy"],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence()
        )
        distiller.fit(
            train_gen, validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )
        model = student

    # ---------- SAVE ----------
    model.save(FINAL_MODEL_PATH)
    with open(HASH_PATH, "w") as f:
        f.write(new_hash)
    print("💾 Saved student as FINAL_MODEL_PATH:", FINAL_MODEL_PATH)

    # Export TFLite from BEST model (with optimization)
    if os.path.exists(FINAL_MODEL_PATH):
        best_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
        tflite_path = os.path.join(MODEL_DIR, "final_model.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"📦 Saved TFLite -> {tflite_path}")
    else:
        print("⚠️ No final model found for TFLite export.")

    # Plot training curves
    acc, val_acc, loss, val_loss = [], [], [], []
    if history1 is not None:
        acc += history1.history.get('accuracy', [])
        val_acc += history1.history.get('val_accuracy', [])
        loss += history1.history.get('loss', [])
        val_loss += history1.history.get('val_loss', [])
    if history2 is not None:
        acc += history2.history.get('accuracy', [])
        val_acc += history2.history.get('val_accuracy', [])
        loss += history2.history.get('loss', [])
        val_loss += history2.history.get('val_loss', [])

    if acc or val_acc:
        plt.figure(figsize=(9, 6))
        if acc: plt.plot(acc, label="Train Acc")
        if val_acc: plt.plot(val_acc, label="Val Acc")
        if loss: plt.plot(loss, "--", label="Train Loss")
        if val_loss: plt.plot(val_loss, "--", label="Val Loss")
        if history1 is not None:
            plt.axvline(x=len(history1.history.get('accuracy', [])), linestyle=":", label="Fine-tune start")
        plt.legend(); plt.title("Training Performance"); plt.xlabel("Epoch"); plt.ylabel("Accuracy / Loss")
        chart_path = os.path.join(MODEL_DIR, "accuracy.png")
        plt.savefig(chart_path); plt.close()
        print("📊 Saved training chart:", chart_path)
        if acc: print(f"🏆 Final Train Acc: {acc[-1]*100:.2f}%")
        if val_acc: print(f"🏆 Final Val Acc: {val_acc[-1]*100:.2f}%")

    # Confusion Matrix on validation split + per-class metrics
    try:
        val_gen.reset()
        steps = len(val_gen) if len(val_gen) > 0 else 1
        y_true = val_gen.classes
        # Use predict with explicit steps to ensure we get predictions for all validation samples
        y_pred_probs = model.predict(val_gen, steps=steps, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Confusion matrix (counts)
        cm = confusion_matrix(y_true, y_pred)
        labels = list(val_gen.class_indices.keys())

        # Safe normalization: avoid divide-by-zero for rows with sum 0
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm.astype(float) / row_sums

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
        plt.title("Confusion Matrix (counts)")
        plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
        plt.close()

        # Normalized image
        plt.figure(figsize=(8, 6))
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
        disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca())
        plt.title("Confusion Matrix (normalized by true class)")
        plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_norm.png"))
        plt.close()

        # Per-class precision/recall/f1/accuracy
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
        per_class_accuracy = []
        for i in range(len(labels)):
            total = cm[i].sum()
            correct = cm[i, i]
            acc_val = (correct / total) if total > 0 else 0.0
            per_class_accuracy.append(acc_val)

        # Print and save per-class metrics
        print("\nPer-class metrics:")
        header = ["class", "precision", "recall", "f1", "support", "accuracy"]
        for i, lab in enumerate(labels):
            print(f"{lab:15s}  prec={precision[i]:.3f}  rec={recall[i]:.3f}  f1={f1[i]:.3f}  sup={int(support[i])}  acc={per_class_accuracy[i]:.3f}")

        # Save CSV
        with open(PER_CLASS_CSV, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i, lab in enumerate(labels):
                writer.writerow([lab, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", int(support[i]), f"{per_class_accuracy[i]:.4f}"])
        print(f"📊 Per-class metrics saved -> {PER_CLASS_CSV}")

        np.set_printoptions(precision=3, suppress=True)
        print("\nNormalized Confusion Matrix:\n", cm_norm)
    except Exception as e:
        print("⚠️ Could not compute confusion matrix / per-class metrics:", e)

    # Cleanup temp dataset
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
        print("🧹 Cleaned new-only dataset.")

    return True


if __name__ == "__main__":
    ok = train_model()
    print("🎉 Training completed!" if ok else "❌ Training failed.")