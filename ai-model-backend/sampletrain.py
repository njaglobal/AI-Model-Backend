# train.py (updated)
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
from tensorflow.keras.applications import ResNet50
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
FINE_TUNE_AT = 140
HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
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


def build_new_model(num_classes: int) -> tf.keras.Model:
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
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


def try_build_student_from_teacher(teacher: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    """
    If teacher's last Dense matches `num_classes`, clone weights; otherwise build fresh.
    """
    try:
        last = teacher.layers[-1]
        units = getattr(last, "units", None)
        if units == num_classes:
            student = tf.keras.models.clone_model(teacher)
            student.set_weights(teacher.get_weights())
            student.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
            return student
    except Exception:
        pass
    return build_new_model(num_classes)



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
    new_files = download_images()  # should download any new files and return their local paths

    if not os.path.exists(DATA_DIR) or not any(Path(DATA_DIR).rglob("*.*")):
        print("❌ No training data found.")
        return False

    # Dataset change detection (on full DATA_DIR)
    new_hash = compute_dataset_hash(DATA_DIR)
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash and not new_files:
            print("✅ No changes, skipping retrain.")
            return True

    # Choose dataset path
    if new_files:
        dataset_path = prepare_new_only_dataset(new_files)  # 🆕 ONLY new data
        print(f"📦 Using new-only dataset at: {dataset_path}")
    else:
        dataset_path = DATA_DIR
        print("ℹ️ No new files → Training on full dataset.")

    # Balance the dataset we will use (new-only or full)
    try:
        # Use stronger augmentation for classes with very few seeds
        balance_and_augment(dataset_path)
    except Exception as e:
        print("⚠️ balance_and_augment() failed:", e)
        # proceed anyway

    # Data generators (train-time augmentation = moderate augment)
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
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=RANDOM_SEED
    )

    # Class counts (folder-level)
    for cls, _ in train_gen.class_indices.items():
        count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
        print(f"Class '{cls}' -> {count} images")

    # Labels: write current subset order (overwrite labels.txt to match this model)
    class_indices = train_gen.class_indices
    with open(LABELS_PATH, "w") as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"🏷️ Labels (subset) saved -> {LABELS_PATH}")

    # Class weights (subset)
    all_labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = dict(enumerate(weights))
    print("⚖️ Class Weights:", class_weights)

    history1 = None
    history2 = None

    if new_files and os.path.exists(BEST_MODEL_PATH) and USE_DISTILLATION:
        # ---------- Incremental update with distillation on new-only ----------
        print("♻️ Loading teacher model for distillation...")
        teacher = tf.keras.models.load_model(BEST_MODEL_PATH)
        teacher.trainable = False

        # Build student sized to the CURRENT SUBSET of classes
        num_subset_classes = len(class_indices)
        if INIT_STUDENT_FROM_TEACHER:
            student = try_build_student_from_teacher(teacher, num_classes=num_subset_classes)
        else:
            student = build_new_model(num_classes=num_subset_classes)

        # Light unfreeze of student tail for adaptation (BNs stay frozen)
        try:
            unfreeze_for_finetune(student, fine_tune_at=FINE_TUNE_AT)
        except Exception:
            pass

        # Build mapping from student class order to teacher class order by class name
        student_labels = [lbl for lbl, _ in sorted(class_indices.items(), key=lambda x: x[1])]

        # Try to read teacher's full labels saved during a full training run
        teacher_labels_path_guess = os.path.join(MODEL_DIR, "labels_full.txt")
        if os.path.exists(teacher_labels_path_guess):
            with open(teacher_labels_path_guess, "r") as f:
                teacher_labels = [ln.strip() for ln in f if ln.strip()]
        else:
            # Fallback: assume teacher and student share class names (common in many setups)
            teacher_labels = student_labels

        # Map each student label to its index in teacher labels
        teacher_indices = []
        for lbl in student_labels:
            if lbl not in teacher_labels:
                raise ValueError(f"Student label '{lbl}' not found in teacher label list. "
                                 "Please maintain a labels_full.txt with the teacher's label order.")
            teacher_indices.append(teacher_labels.index(lbl))

        distiller = Distiller(
            student=student,
            teacher=teacher,
            temperature=TEMPERATURE,
            alpha=ALPHA,
            teacher_indices=teacher_indices
        )
        distiller.compile(
            optimizer=Adam(learning_rate=DISTILL_LR),
            metrics=["accuracy"],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
            SaveBestStudent(student, BEST_MODEL_PATH),
        ]

        print("🔧 Incremental training (new-only) with knowledge distillation...")

        # Check number of classes in new-only dataset
        num_classes_in_train = len(train_gen.class_indices)
        if num_classes_in_train < 2:
            print("⚠️ Only one class in dataset — skipping distillation/fine-tune.")
            model = student  # assign student as the model, even if not trained
        else:
            history2 = distiller.fit(
                train_gen,
                validation_data=val_gen,
                epochs=FINE_TUNE_EPOCHS,
                class_weight=class_weights,
                callbacks=callbacks
            )
            model = student

    elif new_files and os.path.exists(BEST_MODEL_PATH):
        # ---------- Fallback incremental: simple fine-tune on new-only ----------
        print("♻️ Loading previous best model for incremental fine-tuning (no distillation)...")
        # Build a student for subset classes
        model = build_new_model(num_classes=len(class_indices))
        unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
        callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # # --- After training, check a single validation batch predictions for debug ---
        # try:
        #     # Map indices to class names
        #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

        #     val_gen.reset()
        #     x_batch, y_batch = next(val_gen)
        #     preds = model.predict(x_batch)
        #     pred_classes = np.argmax(preds, axis=1)
        #     true_classes = np.argmax(y_batch, axis=1)

        #     for i in range(len(pred_classes)):
        #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
        # except Exception as e:
        #     print("⚠️ Could not run single-batch debug predictions:", e)

        model.save(BEST_MODEL_PATH)

    else:
        # ---------- Cold start on full data ----------
        print("🚀 Building a new model and training from scratch on full dataset...")
        # For cold-start we should balance the full dataset (dataset_path == DATA_DIR)
        try:
            balance_and_augment(DATA_DIR)
        except Exception as e:
            print("⚠️ balance_and_augment(DATA_DIR) failed:", e)

        # Rebuild generators for full dataset (ensures augmentation we just added is picked up)
        dataset_path = DATA_DIR
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
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='training', shuffle=True, seed=RANDOM_SEED
        )
        val_gen = datagen_full.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='validation', shuffle=False, seed=RANDOM_SEED
        )

        # Save full labels order for future teacher alignment
        full_class_indices = train_gen.class_indices
        labels_full_path = os.path.join(MODEL_DIR, "labels_full.txt")
        with open(labels_full_path, "w") as f:
            for label, index in sorted(full_class_indices.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")
        print(f"🏷️ Full labels saved -> {labels_full_path}")

        model = build_new_model(num_classes=len(full_class_indices))
        callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

        print("🚀 Phase 1: Training top layers (frozen backbone)...")
        history1 = model.fit(
            train_gen, validation_data=val_gen, epochs=EPOCHS,
            class_weight=dict(enumerate(
                compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
            )),
            callbacks=callbacks
        )

        # # --- After training, check predictions (small debug batch) ---
        # try:
        #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
        #     val_gen.reset()
        #     x_batch, y_batch = next(val_gen)
        #     preds = model.predict(x_batch)
        #     pred_classes = np.argmax(preds, axis=1)
        #     true_classes = np.argmax(y_batch, axis=1)
        #     for i in range(len(pred_classes)):
        #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
        # except Exception as e:
        #     print("⚠️ Could not run single-batch debug predictions:", e)

        print("🔧 Phase 2: Fine-tuning ResNet tail...")
        unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
        history2 = model.fit(
            train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
            class_weight=dict(enumerate(
                compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
            )),
            callbacks=callbacks
        )

        # # --- After training, check predictions (small debug batch) ---
        # try:
        #     idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
        #     val_gen.reset()
        #     x_batch, y_batch = next(val_gen)
        #     preds = model.predict(x_batch)
        #     pred_classes = np.argmax(preds, axis=1)
        #     true_classes = np.argmax(y_batch, axis=1)
        #     for i in range(len(pred_classes)):
        #         print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")
        # except Exception as e:
        #     print("⚠️ Could not run single-batch debug predictions:", e)

        model.save(BEST_MODEL_PATH)

    # Save final model snapshot
    model.save(FINAL_MODEL_PATH)
    with open(HASH_PATH, "w") as f:
        f.write(new_hash)
    print("💾 Saved final Keras model:", FINAL_MODEL_PATH)

    # Export TFLite from BEST model (with optimization)
    if os.path.exists(BEST_MODEL_PATH):
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
        tflite_path = os.path.join(MODEL_DIR, "best_model.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("📦 Saved TFLite from best_model.h5 ->", tflite_path)
    else:
        print("⚠️ best_model.h5 not found, skipping TFLite export.")

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