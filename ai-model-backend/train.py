# train.py (ResNet50 teacher + MobileNetV2 student)

import os
import gzip
import json
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
from utils.supabase import download_images, METADATA_FILE, supabase

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
BEST_PATH = os.path.join(MODEL_DIR, "best_student.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, "teacher_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
PER_CLASS_CSV = os.path.join(MODEL_DIR, "per_class_metrics.csv")
BUCKET_NAME = "models"

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


# ========= Supabase Helpers =========
def download_models():
   
    for model_file in ["dataset.hash", "labels_full.txt" ]:
        # Use .gz for .h5 files
        remote_file = f"{model_file}.gz" if model_file.endswith(".h5") else model_file
        local_path = os.path.join(MODEL_DIR, remote_file)
        try:
            content = supabase.storage.from_(BUCKET_NAME).download(f"models/{remote_file}")
            with open(local_path, "wb") as f:
                f.write(content)
            
            # Decompress if .gz
            if local_path.endswith(".gz"):
                decompressed_path = local_path[:-3]  # remove .gz
                with gzip.open(local_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(local_path)  # remove compressed file
                print(f"✅ Downloaded and decompressed {model_file}")
            else:
                print(f"✅ Downloaded {model_file}")
        except Exception as e:
            print(f"⚠️ {model_file} not found in Supabase! Creating new file {model_file}")

# def download_images_from_metadata():
#     """
#     Syncs images from Supabase for incremental or cold-start training.
#     Returns:
#         new_files: list of newly downloaded files (for incremental)
#         metadata_exists: True if metadata.json exists in Supabase
#     """
#     print("🔍 Checking Supabase for metadata.json...")

#     # Ensure main data directory exists
#     os.makedirs(DATA_DIR, exist_ok=True)

#     metadata_exists = False
#     new_files = []

#     # Attempt to download _downloaded_metadata.json
#     try:
#         content = supabase.storage.from_(BUCKET_NAME).download("_downloaded_metadata.json")
#         metadata = json.loads(content)
#         metadata_exists = True
#         print(f"✅ Found metadata.json with {len(metadata)} entries")
#     except Exception:
#         print("⚠️ metadata.json not found in Supabase. Proceeding with cold start.")
#         metadata = {}

#     # If metadata exists, download only new/updated images
#     if metadata_exists:
#         for key, size in metadata.items():
#             folder, name = key.split("/", 1)
#             dest_dir = os.path.join(DATA_DIR, folder)
#             os.makedirs(dest_dir, exist_ok=True)
#             local_path = os.path.join(dest_dir, name)

#             # Skip unchanged files
#             if os.path.exists(local_path) and os.path.getsize(local_path) == size:
#                 continue

#             try:
#                 file_content = supabase.storage.from_(BUCKET_NAME).download(key)
#                 with open(local_path, "wb") as f:
#                     f.write(file_content)
#                 new_files.append(local_path)
#             except Exception as e:
#                 print(f"❌ Failed to download {key}: {e}")

#         print(f"✅ Downloaded {len(new_files)} new/updated files based on metadata")

#     # If metadata does not exist, download all images (cold start)
#     else:
#        new_files = download_images()

#     return new_files, metadata_exists

# def upload_compressed_h5(local_path, remote_path):
#     compressed_path = local_path + ".gz"
#     with open(local_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

#     safe_upload(BUCKET_NAME, remote_path + ".gz", compressed_path)
#     print(f"✅ Uploaded compressed {remote_path}.gz")

# def download_and_decompress_h5(remote_path, local_path):
#     compressed_path = local_path + ".gz"
#     # Download from Supabase
#     content = supabase.storage.from_(BUCKET_NAME).download(remote_path + ".gz")
#     with open(compressed_path, "wb") as f:
#         f.write(content)
#     # Decompress
#     with gzip.open(compressed_path, "rb") as f_in, open(local_path, "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)
#     print(f"✅ Decompressed {local_path} ready for training")


def log_model_metadata(version: int, model_path:str, accuracy: float, class_metrics: dict):
    """Insert model metadata into Supabase table 'models'."""
    try:
        supabase.table("models").insert({
        "version": version,
        "model_path": model_path,
        "accuracy": accuracy,
        "per_class_metrics": class_metrics
        }).execute()
        print(f"📝 Logged metadata for {version} (acc={accuracy:.4f})")
    except Exception as e:
        print(f"❌ Failed to log metadata: {e}")


def safe_upload(bucket, path, file_path, content_type="application/octet-stream"):
    """Remove file if exists, then upload fresh."""
    try:
        supabase.storage.from_(bucket).remove([path])
    except Exception:
        pass  # Ignore if missing
    with open(file_path, "rb") as f:
        return supabase.storage.from_(bucket).upload(
            path, f, {"content-type": content_type}
        )

def upload_model_and_labels(student_acc, class_metrics):
    """
    Upload labels, dataset.hash, TFLite model, and compressed H5 models to Supabase with versioning and cleanup.
    """
    next_version = None
    versioned_name = None

    try:

        # ---------------- Upload JSON Metadata ----------------
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
            safe_upload(BUCKET_NAME, "_downloaded_metadata.json", METADATA_FILE, content_type="application/json")
            print(f"🗂️ Updated _downloaded_metadata.json in Supabase with {len(metadata)} entries")
        # ---------------- Upload labels ----------------
        for labels_file in ["labels_full.txt"]:
            path = os.path.join(MODEL_DIR, labels_file)
            if os.path.exists(path):
                safe_upload(BUCKET_NAME, f"models/{labels_file}", path, content_type="text/plain")
                print(f"✅ Uploaded {labels_file}")

        # ---------------- Upload dataset.hash ----------------
        if os.path.exists(HASH_PATH):
            safe_upload(BUCKET_NAME, "models/dataset.hash", HASH_PATH, content_type="text/plain")
            print("✅ Uploaded dataset.hash")

        # ---------------- Upload compressed H5 models ----------------
        # for model_file in ["teacher_model.h5", "final_model.h5"]:
        #     path = os.path.join(MODEL_DIR, model_file)
        #     if os.path.exists(path):
        #         compressed_path = path + ".gz"
        #         # Compress the .h5 file
        #         with open(path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
        #             shutil.copyfileobj(f_in, f_out)
        #         # Upload compressed file
        #         safe_upload(BUCKET_NAME, f"models/{model_file}.gz", compressed_path, content_type="application/gzip")
        #         print(f"✅ Uploaded {model_file} as {model_file}.gz")
        #         os.remove(compressed_path)

        # ---------------- Upload TFLite model with versioning ----------------
        tflite_path = os.path.join(MODEL_DIR, "final_model.tflite")
        if os.path.exists(tflite_path):
            existing = supabase.storage.from_(BUCKET_NAME).list("models")
            versions = [f["name"] for f in existing if f["name"].startswith("final_model_v")]
            next_version = len(versions) + 1

            versioned_name = f"final_model_v{next_version}.tflite"
            latest_name = "final_model_latest.tflite"

            safe_upload(BUCKET_NAME, f"models/{versioned_name}", tflite_path)
            print(f"✅ Uploaded {versioned_name}")

            safe_upload(BUCKET_NAME, f"models/{latest_name}", tflite_path)
            print(f"✅ Uploaded {latest_name} (overwritten)")

            # Keep only last 5 versions
            if len(versions) >= 5:
                versions_sorted = sorted(
                    versions,
                    key=lambda x: int(x.replace("final_model_v", "").replace(".tflite", ""))
                )
                to_delete = versions_sorted[:-5]
                for old in to_delete:
                    supabase.storage.from_(BUCKET_NAME).remove([f"models/{old}"])
                    print(f"🗑️ Removed old version {old}")

        # ---------------- Log metadata in Supabase ----------------
        if next_version is not None and versioned_name is not None:
            log_model_metadata(next_version, versioned_name, float(student_acc), class_metrics)
            print(f"📊 Logged metadata for v{next_version}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")

    return next_version

#  def upload_model_and_labels(student_acc, class_metrics):
#     """Upload TFLite, H5 models, labels, and dataset.hash to Supabase Storage with versioning and cleanup."""
#     next_version = None
#     try:
#         # 📂 Upload labels
#         for labels_file in ["labels_full.txt", "labels.txt"]:
#             path = os.path.join(MODEL_DIR, labels_file)
#             if os.path.exists(path):
#                 safe_upload(
#                     BUCKET_NAME,
#                     f"models/{labels_file}",
#                     path,
#                     content_type="text/plain"
#                 )
#                 print(f"✅ Uploaded {labels_file}")

#         # 📂 Upload dataset.hash
#         if os.path.exists(HASH_PATH):
#             safe_upload(BUCKET_NAME, "models/dataset.hash", HASH_PATH, content_type="text/plain")
#             print("✅ Uploaded dataset.hash")

       
#         # 📂 Upload TFLite model with versioning
#         tflite_path = os.path.join(MODEL_DIR, "final_model.tflite")
#         existing = supabase.storage.from_(BUCKET_NAME).list("models")
#         versions = [f["name"] for f in existing if f["name"].startswith("final_model_v")]
#         next_version = len(versions) + 1

#         versioned_name = f"final_model_v{next_version}.tflite"
#         latest_name = "final_model_latest.tflite"

#         safe_upload(BUCKET_NAME, f"models/{versioned_name}", tflite_path)
#         print(f"✅ Uploaded {versioned_name}")

#         safe_upload(BUCKET_NAME, f"models/{latest_name}", tflite_path)
#         print(f"✅ Uploaded {latest_name} (overwritten)")

#         # 🧹 Cleanup (keep last 5 versions)
#         if len(versions) >= 5:
#             versions_sorted = sorted(
#                 versions,
#                 key=lambda x: int(x.replace("final_model_v", "").replace(".tflite", ""))
#             )
#             to_delete = versions_sorted[:-5]
#             for old in to_delete:
#                 supabase.storage.from_(BUCKET_NAME).remove([f"models/{old}"])
#                 print(f"🗑️ Removed old version {old}")

#         # 📝 Log in DB
#         log_model_metadata(next_version, versioned_name, float(student_acc), class_metrics)
#         print(f"📊 Logged metadata for v{next_version}")

#        # 📂 Upload compressed teacher_model.h5 and final_model.h5
#         for model_file in ["teacher_model.h5", "final_model.h5"]:
#             path = os.path.join(MODEL_DIR, model_file)
#             if os.path.exists(path):
#                 compressed_path = path + ".gz"
#                 # Compress the .h5 file
#                 with open(path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#                 # Upload the compressed file
#                 safe_upload(BUCKET_NAME, f"models/{model_file}.gz", compressed_path, content_type="application/gzip")
#                 print(f"✅ Uploaded {model_file} as {model_file}.gz")
#                 # Optionally, remove the compressed file after upload
#                 os.remove(compressed_path)

#     except Exception as e:
#         print(f"❌ Upload failed: {e}")

#     return next_version

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
        return {"loss": loss, "accuracy": acc}

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

    download_models()

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
    if os.path.exists(BEST_PATH):
        export_tflite(BEST_PATH)
    else:
        export_tflite(FINAL_MODEL_PATH)

    # Plot training curves
    # plot_training(history1, history2)

    # Evaluate per-class metrics
    student_acc, metrics = evaluate_model(model, val_gen)


    # Upload + log to Supabase
    upload_model_and_labels(student_acc, metrics)

    # ---------------- Update metadata.json in Supabase ----------------
    # update_metadata_json()

   
    # Cleanup temporary datasets and models
    clean_directories()

    return True



def clean_directories():
    # Completely remove NEW_ONLY_DIR
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
        print(f"🧹 Cleaned directory: {NEW_ONLY_DIR}")

    # Remove only specific files in MODEL_DIR
    # if os.path.exists(MODEL_DIR):
    #     for file_name in os.listdir(MODEL_DIR):
    #         file_path = os.path.join(MODEL_DIR, file_name)
    #         if (
    #             file_name in ["dataset.hash", "labels_full.txt", "labels.txt"]
    #             or file_name.endswith(".tflite")
    #         ):
    #             try:
    #                 os.remove(file_path)
    #                 print(f"🧹 Removed file: {file_path}")
    #             except Exception as e:
    #                 print(f"❌ Failed to remove {file_path}: {e}")

    # if os.path.exists(DATA_DIR):
    #     for file_name in os.listdir(DATA_DIR):
    #         file_path = os.path.join(DATA_DIR, file_name)
    #         if file_name.endswith(".json"):
    #             try:
    #                 os.remove(file_path)
    #                 print(f"🧹 Removed file: {file_path}")
    #             except Exception as e:
    #                 print(f"❌ Failed to remove {file_path}: {e}")

    return True



# ========= Update Supabase metadata =========
# def update_metadata_json():
#     """Save current dataset snapshot to Supabase as _downloaded_metadata.json."""
#     metadata = {}
#     for root, _, files in os.walk(DATA_DIR):
#         for f in files:
#             if f.lower().endswith(AUG_EXTS):
#                 path = os.path.join(root, f)
#                 rel_path = os.path.relpath(path, DATA_DIR)
#                 metadata[rel_path] = os.path.getsize(path)

#     local_path = METADATA_FILE
    
#     safe_upload(BUCKET_NAME, local_path, local_path, "application/json")
#     print(f"🗂️ Updated _downloaded_metadata.json in Supabase with {len(metadata)} entries")

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
                  callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True) , 
                    SaveBestStudent(
                            student=student,
                            best_path=BEST_PATH
                        )
                  ])
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
                  
                  callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True), 
                    SaveBestStudent(student=student,best_path=BEST_PATH)
                  ])
    
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

# ========= Evaluation =========
def evaluate_model(model, val_gen):
    # """Compute per-class metrics and save to CSV."""
    # y_true = val_gen.classes
    # y_pred = np.argmax(model.predict(val_gen), axis=1)
    # labels = list(val_gen.class_indices.keys())
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    # cm = confusion_matrix(y_true, y_pred)
    # ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    # plt.show()

    # with open(PER_CLASS_CSV, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["class", "precision", "recall", "f1-score"])
    #     for i, cls in enumerate(labels):
    #         writer.writerow([cls, precision[i], recall[i], f1[i]])
    # print(f"📊 Per-class metrics saved -> {PER_CLASS_CSV}")
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


    # Compute global accuracy
    acc = np.mean(y_true == y_pred)
    # Convert metrics to dict for Supabase
    metrics_dict = {cls: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i, cls in enumerate(labels)}


    return acc, metrics_dict



if __name__ == "__main__":
    ok = train_model()
    print("🎉 Training completed!" if ok else "❌ Training failed.")
