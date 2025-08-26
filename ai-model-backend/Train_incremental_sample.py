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
# BATCH_SIZE = 8          # smaller batch for tiny dataset
# EPOCHS = 20             # allow patience/early stopping
# FINE_TUNE_EPOCHS = 10
# FINE_TUNE_AT = 140      # freeze up to this layer index when unfreezing
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
# LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
# BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
# FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")


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

#     DATASET_ROOTS = ["training_data"]

#     def class_of(p: str) -> str:
#         parts = Path(p).parts
#         for root in DATASET_ROOTS:
#             if root in parts:
#                 idx = parts.index(root)
#                 return parts[idx + 1]  # the folder after "training_data"
#         raise ValueError(f"Could not determine class from path: {p}")

#     # Copy all NEW files
#     for f in new_files:
#         cls = class_of(f)
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy(f, dest_dir / Path(f).name)

#     # Add sampled OLD files per class (avoid catastrophic forgetting)
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
#     """Unfreeze last layers of the base model for fine-tuning while keeping early layers frozen.
#     Handles Sequential with ResNet50 as first layer."""
#     # Locate the backbone (first layer if Sequential)
#     backbone = model.layers[0] if isinstance(model, models.Sequential) else model.get_layer(index=0)
#     if not isinstance(backbone, tf.keras.Model):
#         return

#     backbone.trainable = True
#     for i, layer in enumerate(backbone.layers):
#         layer.trainable = (i >= fine_tune_at) and not isinstance(layer, tf.keras.layers.BatchNormalization)

#     model.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])


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

#     # Determine dataset to use (incremental or full)
#     all_files = [str(p) for p in Path(DATA_DIR).rglob("*.*") if p.is_file()]
#     if new_files:
#         dataset_path = prepare_incremental_dataset(new_files, all_files, ratio_old=0.3)
#         print(f"📦 Using incremental dataset at: {dataset_path}")
#     else:
#         dataset_path = DATA_DIR
#         print("ℹ️ Training on full dataset.")

#     # Data generators with strong augmentation (helps small datasets)
#     datagen = ImageDataGenerator(
#         validation_split=0.30 if len(all_files) < 600 else 0.20,
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

#     # ✅ Print class counts for visibility
#     for cls, _ in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Save labels in fixed index order
#     class_indices = train_gen.class_indices
#     with open(LABELS_PATH, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")
#     print(f"🏷️ Labels saved -> {LABELS_PATH}")

#     # ✅ Class weights to mitigate imbalance
#     all_labels = train_gen.classes
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.unique(all_labels),
#         y=all_labels
#     )
#     class_weights = dict(enumerate(weights))
#     print("⚖️ Class Weights:", class_weights)

#     # ========== Build / Load Model depending on scenario ==========
#     model = None

#     if new_files and os.path.exists(BEST_MODEL_PATH):
#         # Incremental update → load previous best and fine-tune only
#         print("♻️ Loading previous best model for incremental fine-tuning...")
#         model = tf.keras.models.load_model(BEST_MODEL_PATH)
#         # Recompile with small LR and unfreeze tail
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#             ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True)
#         ]
#         print("🔧 Incremental fine-tuning...")
#         history1 = None
#         history2 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )
#     else:
#         # Cold-start or no previous model → two-phase training
#         print("🚀 Building a new model and training from scratch...")
#         model = build_new_model(num_classes=len(class_indices))

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#             ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True)
#         ]

#         print("🚀 Phase 1: Training top layers (frozen backbone)...")
#         history1 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         print("🔧 Phase 2: Fine-tuning ResNet tail...")
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         history2 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#     # Save final model snapshot (in addition to BEST_MODEL_PATH via checkpoint)
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

#     # Plot training curves (handle both incremental and full training)
#     acc = []
#     val_acc = []
#     loss = []
#     val_loss = []

#     if 'history1' in locals() and history1 is not None:
#         acc += history1.history.get('accuracy', [])
#         val_acc += history1.history.get('val_accuracy', [])
#         loss += history1.history.get('loss', [])
#         val_loss += history1.history.get('val_loss', [])

#     if 'history2' in locals() and history2 is not None:
#         acc += history2.history.get('accuracy', [])
#         val_acc += history2.history.get('val_accuracy', [])
#         loss += history2.history.get('loss', [])
#         val_loss += history2.history.get('val_loss', [])

#     if acc or val_acc:
#         plt.figure(figsize=(9, 6))
#         if acc:
#             plt.plot(acc, label="Train Acc")
#         if val_acc:
#             plt.plot(val_acc, label="Val Acc")
#         if loss:
#             plt.plot(loss, "--", label="Train Loss")
#         if val_loss:
#             plt.plot(val_loss, "--", label="Val Loss")
#         if 'history1' in locals() and history1 is not None:
#             plt.axvline(x=len(history1.history.get('accuracy', [])), linestyle=":", label="Fine-tune start")
#         plt.legend()
#         plt.title("Training Performance")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy / Loss")
#         chart_path = os.path.join(MODEL_DIR, "accuracy.png")
#         plt.savefig(chart_path)
#         plt.close()
#         print("📊 Saved training chart:", chart_path)

#         print(f"🏆 Final Train Acc: {acc[-1]*100:.2f}%")
#         print(f"🏆 Final Val Acc: {val_acc[-1]*100:.2f}%")

#     # Confusion Matrix on validation split
#     try:
#         val_gen.reset()
#         y_true = val_gen.classes
#         y_pred_probs = model.predict(val_gen, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)

#         cm = confusion_matrix(y_true, y_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_gen.class_indices.keys()))
#         plt.figure(figsize=(8, 6))
#         disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
#         plt.title("Confusion Matrix on Validation Set")
#         cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
#         plt.savefig(cm_path)
#         plt.close()
#         print("📊 Confusion matrix saved ->", cm_path)

#         cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
#         np.set_printoptions(precision=3, suppress=True)
#         print("\nNormalized Confusion Matrix:\n", cm_norm)
#     except Exception as e:
#         print("⚠️ Could not compute confusion matrix:", e)

#     # Cleanup incremental dataset
#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#         print("🧹 Cleaned incremental dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")

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
# from tensorflow.keras.callbacks import EarlyStopping, Callback

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from utils.supabase import download_images  # must return List[str] of new local files

# # ========= Paths & Config =========
# DATA_DIR = "training_data"
# INCREMENTAL_DIR = "incremental_data"   # temporary dataset (new + sampled old)
# MODEL_DIR = "models"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 8          # smaller batch for tiny dataset
# EPOCHS = 20             # allow patience/early stopping
# FINE_TUNE_EPOCHS = 10
# FINE_TUNE_AT = 140      # freeze up to this layer index when unfreezing
# HASH_PATH = os.path.join(MODEL_DIR, "dataset.hash")
# LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
# BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
# FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# # ======== Distillation knobs ========
# USE_DISTILLATION = True          # use teacher-student for incremental updates
# TEMPERATURE = 2.0                # softening factor for teacher/student logits
# ALPHA = 0.7                      # weight for supervised (GT) loss vs distillation
# INIT_STUDENT_FROM_TEACHER = True # start student from teacher weights when compatible

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

#     # Derive class from path relative to DATA_DIR to avoid 'training_data' becoming a class
#     DATASET_ROOTS = [Path(DATA_DIR).name]

#     def class_of(p: str) -> str:
#         parts = Path(p).parts
#         for root in DATASET_ROOTS:
#             if root in parts:
#                 idx = parts.index(root)
#                 if idx + 1 < len(parts):
#                     return parts[idx + 1]  # folder after the dataset root
#         # fallback: immediate parent (keeps backward compatibility)
#         return Path(p).parent.name

#     # Copy all NEW files
#     for f in new_files:
#         cls = class_of(f)
#         dest_dir = Path(INCREMENTAL_DIR) / cls
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy(f, dest_dir / Path(f).name)

#     # Add sampled OLD files per class (avoid catastrophic forgetting)
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
#         # keep BN frozen for stability
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
#             # compile with a standard optimizer; Distiller handles the loss
#             student.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#             return student
#     except Exception:
#         pass
#     return build_new_model(num_classes)


# # ========= Knowledge Distillation =========
# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher, temperature=2.0, alpha=0.5):
#         super().__init__()
#         self.teacher = teacher
#         self.student = student
#         self.temperature = temperature
#         self.alpha = alpha

#     def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
#         super().compile(optimizer=optimizer, metrics=metrics)
#         self.student_loss_fn = student_loss_fn
#         self.distillation_loss_fn = distillation_loss_fn

#     def train_step(self, data):
#         x, y = data
#         # No grad through teacher
#         teacher_preds = tf.stop_gradient(self.teacher(x, training=False))

#         with tf.GradientTape() as tape:
#             student_preds = self.student(x, training=True)

#             # supervised loss on GT
#             student_loss = self.student_loss_fn(y, student_preds)

#             # distillation loss on softened distributions
#             t = self.temperature
#             distill_loss = self.distillation_loss_fn(
#                 tf.nn.softmax(teacher_preds / t, axis=1),
#                 tf.nn.softmax(student_preds / t, axis=1)
#             )

#             loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

#         grads = tape.gradient(loss, self.student.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

#         acc = tf.reduce_mean(
#             tf.cast(
#                 tf.equal(tf.argmax(student_preds, axis=1), tf.argmax(y, axis=1)),
#                 tf.float32
#             )
#         )
#         return {
#             "loss": loss,
#             "student_loss": student_loss,
#             "distillation_loss": distill_loss,
#             "accuracy": acc,
#         }

#     def test_step(self, data):
#         x, y = data
#         preds = self.student(x, training=False)
#         student_loss = self.student_loss_fn(y, preds)
#         acc = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32)
#         )
#         return {"val_loss": student_loss, "val_accuracy": acc}


# class SaveBestStudent(Callback):
#     """
#     Track val_accuracy and save the student (not the Distiller wrapper) to BEST_MODEL_PATH.
#     """
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

#     # Determine dataset to use (incremental or full)
#     all_files = [str(p) for p in Path(DATA_DIR).rglob("*.*") if p.is_file()]
#     if new_files:
#         dataset_path = prepare_incremental_dataset(new_files, all_files, ratio_old=0.3)
#         print(f"📦 Using incremental dataset at: {dataset_path}")
#     else:
#         dataset_path = DATA_DIR
#         print("ℹ️ Training on full dataset.")

#     # Data generators with strong augmentation (helps small datasets)
#     datagen = ImageDataGenerator(
#         validation_split=0.30 if len(all_files) < 600 else 0.20,
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

#     # ✅ Print class counts for visibility
#     for cls, _ in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Save labels in fixed index order
#     class_indices = train_gen.class_indices
#     with open(LABELS_PATH, "w") as f:
#         for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
#             f.write(f"{label}\n")
#     print(f"🏷️ Labels saved -> {LABELS_PATH}")

#     # ✅ Class weights to mitigate imbalance
#     all_labels = train_gen.classes
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.unique(all_labels),
#         y=all_labels
#     )
#     class_weights = dict(enumerate(weights))
#     print("⚖️ Class Weights:", class_weights)

#     # ========== Build / Load Model depending on scenario ==========
#     history1 = None
#     history2 = None

#     if new_files and os.path.exists(BEST_MODEL_PATH) and USE_DISTILLATION:
#         # Incremental update → Teacher-Student Distillation
#         print("♻️ Loading teacher model for distillation...")
#         teacher = tf.keras.models.load_model(BEST_MODEL_PATH)
#         teacher.trainable = False

#         print("👩‍🏫 Creating student model...")
#         if INIT_STUDENT_FROM_TEACHER:
#             student = try_build_student_from_teacher(teacher, num_classes=len(class_indices))
#         else:
#             student = build_new_model(num_classes=len(class_indices))

#         # optional: lightly unfreeze tail for better adaptation
#         try:
#             unfreeze_for_finetune(student, fine_tune_at=FINE_TUNE_AT)
#         except Exception:
#             pass

#         distiller = Distiller(student=student, teacher=teacher, temperature=TEMPERATURE, alpha=ALPHA)
#         distiller.compile(
#             optimizer=Adam(learning_rate=1e-4),
#             metrics=["accuracy"],
#             student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
#             distillation_loss_fn=tf.keras.losses.KLDivergence(),
#         )

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
#             SaveBestStudent(student, BEST_MODEL_PATH),
#         ]

#         print("🔧 Incremental training with knowledge distillation...")
#         history2 = distiller.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         model = student  # final working model is the student

#     elif new_files and os.path.exists(BEST_MODEL_PATH):
#         # Fallback: simple fine-tune if distillation disabled
#         print("♻️ Loading previous best model for incremental fine-tuning...")
#         model = tf.keras.models.load_model(BEST_MODEL_PATH)
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#         ]

#         print("🔧 Incremental fine-tuning...")
#         history2 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         # Save best snapshot from this run
#         model.save(BEST_MODEL_PATH)

#     else:
#         # Cold-start or no previous model → two-phase training
#         print("🚀 Building a new model and training from scratch...")
#         model = build_new_model(num_classes=len(class_indices))

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#         ]

#         print("🚀 Phase 1: Training top layers (frozen backbone)...")
#         history1 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         print("🔧 Phase 2: Fine-tuning ResNet tail...")
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         history2 = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         # Save as current best for future incremental steps
#         model.save(BEST_MODEL_PATH)

#     # Save final model snapshot (in addition to BEST_MODEL_PATH via callback/logic above)
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

#     # Plot training curves (handle both incremental and full training)
#     acc = []
#     val_acc = []
#     loss = []
#     val_loss = []

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
#         if acc:
#             plt.plot(acc, label="Train Acc")
#         if val_acc:
#             plt.plot(val_acc, label="Val Acc")
#         if loss:
#             plt.plot(loss, "--", label="Train Loss")
#         if val_loss:
#             plt.plot(val_loss, "--", label="Val Loss")
#         if history1 is not None:
#             plt.axvline(x=len(history1.history.get('accuracy', [])), linestyle=":", label="Fine-tune start")
#         plt.legend()
#         plt.title("Training Performance")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy / Loss")
#         chart_path = os.path.join(MODEL_DIR, "accuracy.png")
#         plt.savefig(chart_path)
#         plt.close()
#         print("📊 Saved training chart:", chart_path)

#         if acc:
#             print(f"🏆 Final Train Acc: {acc[-1]*100:.2f}%")
#         if val_acc:
#             print(f"🏆 Final Val Acc: {val_acc[-1]*100:.2f}%")

#     # Confusion Matrix on validation split
#     try:
#         val_gen.reset()
#         y_true = val_gen.classes
#         y_pred_probs = model.predict(val_gen, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)

#         cm = confusion_matrix(y_true, y_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_gen.class_indices.keys()))
#         plt.figure(figsize=(8, 6))
#         disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
#         plt.title("Confusion Matrix on Validation Set")
#         cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
#         plt.savefig(cm_path)
#         plt.close()
#         print("📊 Confusion matrix saved ->", cm_path)

#         cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
#         np.set_printoptions(precision=3, suppress=True)
#         print("\nNormalized Confusion Matrix:\n", cm_norm)
#     except Exception as e:
#         print("⚠️ Could not compute confusion matrix:", e)

#     # Cleanup incremental dataset
#     if os.path.exists(INCREMENTAL_DIR):
#         shutil.rmtree(INCREMENTAL_DIR)
#         print("🧹 Cleaned incremental dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")


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
# from tensorflow.keras.callbacks import EarlyStopping, Callback

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from utils.supabase import download_images  # must return List[str] of new local files

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

# # ======== Distillation knobs ========
# USE_DISTILLATION = True
# TEMPERATURE = 2.0
# ALPHA = 0.7                 # weight for GT loss; (1-ALPHA) for distillation
# INIT_STUDENT_FROM_TEACHER = True  # if output dims match; otherwise fresh student


# # --- New helper ---
# def balance_and_augment(train_dir, target_count=None):
#     """
#     Oversample minority classes (fire, none-accident) with stronger augmentation.
#     Creates synthetic images until each class has ~target_count samples.
#     If target_count=None, it will balance to the max class count.
#     """

#     strong_aug = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=40,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.2,
#         zoom_range=0.3,
#         horizontal_flip=True,
#         brightness_range=[0.5, 1.5],
#         fill_mode="nearest"
#     )

#     # Count current samples per class
#     class_counts = {}
#     for cls in os.listdir(train_dir):
#         cls_path = Path(train_dir) / cls
#         if cls_path.is_dir():
#             class_counts[cls] = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png")))

#     max_count = max(class_counts.values())
#     if target_count is None:
#         target_count = max_count

#     print("📊 Class distribution before balancing:", class_counts)

#     # Oversample minority classes
#     for cls, count in class_counts.items():
#         if count >= target_count:
#             continue

#         cls_path = Path(train_dir) / cls
#         n_to_generate = target_count - count
#         print(f"🔄 Augmenting {cls}: need {n_to_generate} more images")

#         gen = strong_aug.flow_from_directory(
#             directory=str(train_dir),
#             classes=[cls],
#             target_size=(224,224),
#             batch_size=1,
#             save_to_dir=str(cls_path),
#             save_prefix="aug",
#             save_format="jpg"
#         )

#         for _ in range(n_to_generate):
#             gen.next()  # generate and save new image

#     # Final distribution check
#     balanced_counts = {}
#     for cls in os.listdir(train_dir):
#         cls_path = Path(train_dir) / cls
#         if cls_path.is_dir():
#             balanced_counts[cls] = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png")))
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
#     Infer class from a path under training_data/<class>/... or any structure.
#     Prefer class as the directory immediately under DATA_DIR; fallback to parent.
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


# # ========= Knowledge Distillation =========
# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher, temperature=2.0, alpha=0.5, teacher_indices=None):
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
#         x, y = data
#         teacher_preds = tf.stop_gradient(self.teacher(x, training=False))
#         teacher_preds = self._align_teacher_logits(teacher_preds)

#         with tf.GradientTape() as tape:
#             student_preds = self.student(x, training=True)

#             student_loss = self.student_loss_fn(y, student_preds)

#             t = self.temperature
#             distill_loss = self.distillation_loss_fn(
#                 tf.nn.softmax(teacher_preds / t, axis=1),
#                 tf.nn.softmax(student_preds / t, axis=1)
#             )

#             loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

#         grads = tape.gradient(loss, self.student.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

#         acc = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(student_preds, axis=1), tf.argmax(y, axis=1)), tf.float32)
#         )
#         return {
#             "loss": loss,
#             "student_loss": student_loss,
#             "distillation_loss": distill_loss,
#             "accuracy": acc,
#         }

#     def test_step(self, data):
#         x, y = data
#         preds = self.student(x, training=False)
#         student_loss = self.student_loss_fn(y, preds)
#         acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))
#         return {"val_loss": student_loss, "val_accuracy": acc}


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

#     # Data generators
#     # With tiny new-only sets, use a larger validation split (0.3)
#     # Before training starts, balance dataset
#     balance_and_augment(TRAIN_DIR)

#     datagen = ImageDataGenerator(
#         validation_split=0.30 if new_files else (0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20),
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

#     # Class counts
#     for cls, _ in train_gen.class_indices.items():
#         count = sum(1 for _ in Path(dataset_path, cls).rglob("*.*"))
#         print(f"Class '{cls}' -> {count} images")

#     # Labels: keep existing file for reference (full label space), but
#     # write current subset order as well (overwriting is fine for deployment tied to this model)
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

#         # Align teacher logits to student's class order if needed
#         teacher_label_list = None
#         if os.path.exists(LABELS_PATH):
#             # We saved subset above; for alignment, we need the teacher's *original* labels if you keep them elsewhere.
#             # If you keep a "labels_full.txt", load it here. Otherwise, we assume teacher and student share class names.
#             pass

#         # Build mapping from student class order to teacher class order by class name
#         # Get student class order (subset)
#         student_labels = [lbl for lbl, _ in sorted(class_indices.items(), key=lambda x: x[1])]

#         # Try to infer teacher's class names from an older labels file if you keep one like "labels_full.txt".
#         # If not available, we assume the teacher's last layer order == previous LABELS_PATH at the time that model was saved.
#         teacher_labels_path_guess = os.path.join(MODEL_DIR, "labels_full.txt")
#         if os.path.exists(teacher_labels_path_guess):
#             with open(teacher_labels_path_guess, "r") as f:
#                 teacher_labels = [ln.strip() for ln in f if ln.strip()]
#         else:
#             # Fallback: assume same as current subset order; in many setups, teacher and student share the same label names.
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
#             optimizer=Adam(learning_rate=1e-4),
#             metrics=["accuracy"],
#             student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
#             distillation_loss_fn=tf.keras.losses.KLDivergence(),
#         )

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
#             SaveBestStudent(student, BEST_MODEL_PATH),
#         ]

#         print("🔧 Incremental training (new-only) with knowledge distillation...")
#         history2 = distiller.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         model = student

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
#         model.save(BEST_MODEL_PATH)

#     else:
#         # ---------- Cold start on full data ----------
#         print("🚀 Building a new model and training from scratch on full dataset...")
#         # Before training starts, balance dataset
#         balance_and_augment(TRAIN_DIR)
#         # Rebuild generators for full dataset
#         dataset_path = DATA_DIR
#         datagen_full = ImageDataGenerator(
#             validation_split=0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20,
#             rescale=1./255,
#             rotation_range=25,
#             width_shift_range=0.15,
#             height_shift_range=0.15,
#             shear_range=0.15,
#             zoom_range=0.25,
#             horizontal_flip=True,
#             brightness_range=[0.7, 1.3],
#             fill_mode="nearest"
#         )
#         train_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='training', shuffle=True
#         )
#         val_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='validation', shuffle=False
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

#         print("🔧 Phase 2: Fine-tuning ResNet tail...")
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         history2 = model.fit(
#             train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
#             class_weight=dict(enumerate(
#                 compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
#             )),
#             callbacks=callbacks
#         )

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

#     # Confusion Matrix on validation split
#     try:
#         val_gen.reset()
#         y_true = val_gen.classes
#         y_pred_probs = model.predict(val_gen, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)

#         cm = confusion_matrix(y_true, y_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_gen.class_indices.keys()))
#         plt.figure(figsize=(8, 6))
#         disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
#         plt.title("Confusion Matrix on Validation Set")
#         cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
#         plt.savefig(cm_path); plt.close()
#         print("📊 Confusion matrix saved ->", cm_path)

#         cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
#         np.set_printoptions(precision=3, suppress=True)
#         print("\nNormalized Confusion Matrix:\n", cm_norm)
#     except Exception as e:
#         print("⚠️ Could not compute confusion matrix:", e)

#     # Cleanup temp dataset
#     if os.path.exists(NEW_ONLY_DIR):
#         shutil.rmtree(NEW_ONLY_DIR)
#         print("🧹 Cleaned new-only dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")




# train.py (updated)
# import os
# import random
# import shutil
# import hashlib
# from pathlib import Path
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
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from utils.supabase import download_images  # must return List[str] of new local files

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

# # ======== Distillation knobs ========
# USE_DISTILLATION = True
# TEMPERATURE = 2.0
# ALPHA = 0.7                 # weight for GT loss; (1-ALPHA) for distillation
# INIT_STUDENT_FROM_TEACHER = True  # if output dims match; otherwise fresh student

# # ======== Augmentation balancing knobs ========
# AUG_SAVE_PREFIX = "aug_"
# AUG_TARGET_CAP = 1000  # hard cap per class to avoid runaway disk growth
# AUG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# # --- NEW helper: robust balance_and_augment ---
# def balance_and_augment(train_dir: str, target_count: int = None, target_count_cap: int = AUG_TARGET_CAP):
#     """
#     Oversample minority classes in `train_dir` by generating augmented images and saving them
#     into the class folders. Does NOT apply rescale (so saved images remain normal uint8).
#     - train_dir: path to dataset root containing class subfolders
#     - target_count: desired sample count per class (if None -> use max class size)
#     - target_count_cap: absolute hard cap per class to avoid runaway disk growth
#     """
#     train_dir = Path(train_dir)
#     if not train_dir.exists():
#         raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

#     # Strong augmentation (no rescale here; we're saving files to disk)
#     strong_aug = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.2,
#         zoom_range=0.3,
#         horizontal_flip=True,
#         brightness_range=[0.5, 1.5],
#         fill_mode="nearest"
#     )

#     # Count current samples per class (ignore files starting with AUG_SAVE_PREFIX when counting seeds)
#     class_counts = {}
#     for cls in sorted(os.listdir(train_dir)):
#         cls_path = train_dir / cls
#         if not cls_path.is_dir():
#             continue
#         # count original images (not previously generated aug_ files) as seeds for augmentation
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

#         # Count total existing images (including previously generated ones) so we know how many to add
#         total_existing = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
#         if total_existing >= target_count:
#             print(f"ℹ️ Class '{cls}' already has {total_existing} images (>= target {target_count}) - skipping.")
#             continue

#         n_to_generate = target_count - total_existing
#         # Build a generator that will save augmented images into the class folder.
#         # We use classes=[cls] & directory=train_dir so flow_from_directory will focus on this single class.
#         print(f"🔄 Augmenting '{cls}': need {n_to_generate} more images (seeds: {seed_count}, existing total: {total_existing})")

#         gen = strong_aug.flow_from_directory(
#             directory=str(train_dir),
#             classes=[cls],
#             target_size=(IMG_SIZE[0], IMG_SIZE[1]),
#             batch_size=1,
#             save_to_dir=str(cls_path),
#             save_prefix=AUG_SAVE_PREFIX,
#             save_format="jpg",
#             shuffle=True
#         )

#         # Generate required augmented images
#         for _ in range(n_to_generate):
#             try:
#                 next(gen)  # generate and save new image
#             except Exception as e:
#                 print(f"⚠️ Error during augmentation for {cls}: {e}")
#                 break  # will save an image as aug_<...>.jpg in the class folder

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


# # ========= Knowledge Distillation =========
# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher, temperature=2.0, alpha=0.5, teacher_indices=None):
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
#         x, y = data
#         teacher_preds = tf.stop_gradient(self.teacher(x, training=False))
#         teacher_preds = self._align_teacher_logits(teacher_preds)

#         with tf.GradientTape() as tape:
#             student_preds = self.student(x, training=True)

#             student_loss = self.student_loss_fn(y, student_preds)

#             t = self.temperature
#             distill_loss = self.distillation_loss_fn(
#                 tf.nn.softmax(teacher_preds / t, axis=1),
#                 tf.nn.softmax(student_preds / t, axis=1)
#             )

#             loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

#         grads = tape.gradient(loss, self.student.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

#         acc = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(student_preds, axis=1), tf.argmax(y, axis=1)), tf.float32)
#         )
#         return {
#             "loss": loss,
#             "student_loss": student_loss,
#             "distillation_loss": distill_loss,
#             "accuracy": acc,
#         }

#     def test_step(self, data):
#         x, y = data
#         preds = self.student(x, training=False)
#         student_loss = self.student_loss_fn(y, preds)
#         acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1)), tf.float32))
#         return {"val_loss": student_loss, "val_accuracy": acc}


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
#         balance_and_augment(dataset_path)
#     except Exception as e:
#         print("⚠️ balance_and_augment() failed:", e)
#         # proceed anyway

#     # Data generators
#     datagen = ImageDataGenerator(
#         validation_split=0.30 if new_files else (0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20),
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

#     # Class counts
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
#             optimizer=Adam(learning_rate=1e-4),
#             metrics=["accuracy"],
#             student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
#             distillation_loss_fn=tf.keras.losses.KLDivergence(),
#         )

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
#             SaveBestStudent(student, BEST_MODEL_PATH),
#         ]

#         print("🔧 Incremental training (new-only) with knowledge distillation...")
#         history2 = distiller.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=FINE_TUNE_EPOCHS,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#         model = student

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

#         # --- After training, check predictions ---
#         # Map indices to class names
#         idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

#         # Get a batch of validation data
#         x_batch, y_batch = next(val_gen)
#         preds = model.predict(x_batch)
#         pred_classes = np.argmax(preds, axis=1)
#         true_classes = np.argmax(y_batch, axis=1)

#         for i in range(len(pred_classes)):
#             print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

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
#             rotation_range=25,
#             width_shift_range=0.15,
#             height_shift_range=0.15,
#             shear_range=0.15,
#             zoom_range=0.25,
#             horizontal_flip=True,
#             brightness_range=[0.7, 1.3],
#             fill_mode="nearest"
#         )
#         train_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='training', shuffle=True
#         )
#         val_gen = datagen_full.flow_from_directory(
#             dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
#             class_mode='categorical', subset='validation', shuffle=False
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

#         # --- After training, check predictions ---
#         # Map indices to class names
#         idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

#         # Get a batch of validation data
#         x_batch, y_batch = next(val_gen)
#         preds = model.predict(x_batch)
#         pred_classes = np.argmax(preds, axis=1)
#         true_classes = np.argmax(y_batch, axis=1)

#         for i in range(len(pred_classes)):
#             print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

#         print("🔧 Phase 2: Fine-tuning ResNet tail...")
#         unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
#         history2 = model.fit(
#             train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
#             class_weight=dict(enumerate(
#                 compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
#             )),
#             callbacks=callbacks
#         )

#         # --- After training, check predictions ---
#         # Map indices to class names
#         idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

#         # Get a batch of validation data
#         x_batch, y_batch = next(val_gen)
#         preds = model.predict(x_batch)
#         pred_classes = np.argmax(preds, axis=1)
#         true_classes = np.argmax(y_batch, axis=1)

#         for i in range(len(pred_classes)):
#             print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

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

#     # Confusion Matrix on validation split
#     try:
#         val_gen.reset()
#         y_true = val_gen.classes
#         y_pred_probs = model.predict(val_gen, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)

#        # Confusion matrix (robust normalization)
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

#         # Also save normalized matrix image
#         plt.figure(figsize=(8, 6))
#         disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
#         disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca())
#         plt.title("Confusion Matrix (normalized by true class)")
#         plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_norm.png"))
#         plt.close()

#         np.set_printoptions(precision=3, suppress=True)
#         print("\nNormalized Confusion Matrix:\n", cm_norm)
#     except Exception as e:
#         print("⚠️ Could not compute confusion matrix:", e)

#     # Cleanup temp dataset
#     if os.path.exists(NEW_ONLY_DIR):
#         shutil.rmtree(NEW_ONLY_DIR)
#         print("🧹 Cleaned new-only dataset.")

#     return True


# if __name__ == "__main__":
#     ok = train_model()
#     print("🎉 Training completed!" if ok else "❌ Training failed.")


# train.py (updated)
import os
import random
import shutil
import hashlib
from pathlib import Path
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.supabase import download_images  # must return List[str] of new local files

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

# ======== Distillation knobs =========
USE_DISTILLATION = True
TEMPERATURE = 2.0
ALPHA = 0.7                 # weight for GT loss; (1-ALPHA) for distillation
INIT_STUDENT_FROM_TEACHER = True  # if output dims match; otherwise fresh student

# ======== Augmentation balancing knobs ========
AUG_SAVE_PREFIX = "aug_"
AUG_TARGET_CAP = 1000  # hard cap per class to avoid runaway disk growth
AUG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# --- NEW helper: robust balance_and_augment ---
def balance_and_augment(train_dir: str, target_count: int = None, target_count_cap: int = AUG_TARGET_CAP):
    """
    Oversample minority classes in `train_dir` by generating augmented images and saving them
    into the class folders. Does NOT apply rescale (so saved images remain normal uint8).
    - train_dir: path to dataset root containing class subfolders
    - target_count: desired sample count per class (if None -> use max class size)
    - target_count_cap: absolute hard cap per class to avoid runaway disk growth
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

    # Strong augmentation (no rescale here; we're saving files to disk)
    strong_aug = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode="nearest"
    )

    # Count current samples per class (ignore files starting with AUG_SAVE_PREFIX when counting seeds)
    class_counts = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = train_dir / cls
        if not cls_path.is_dir():
            continue
        # count original images (not previously generated aug_ files) as seeds for augmentation
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

        # Count total existing images (including previously generated ones) so we know how many to add
        total_existing = len([p for p in cls_path.iterdir() if p.suffix.lower() in AUG_EXTS])
        if total_existing >= target_count:
            print(f"ℹ️ Class '{cls}' already has {total_existing} images (>= target {target_count}) - skipping.")
            continue

        n_to_generate = target_count - total_existing
        # Build a generator that will save augmented images into the class folder.
        # We use classes=[cls] & directory=train_dir so flow_from_directory will focus on this single class.
        print(f"🔄 Augmenting '{cls}': need {n_to_generate} more images (seeds: {seed_count}, existing total: {total_existing})")

        gen = strong_aug.flow_from_directory(
            directory=str(train_dir),
            classes=[cls],
            target_size=(IMG_SIZE[0], IMG_SIZE[1]),
            batch_size=1,
            save_to_dir=str(cls_path),
            save_prefix=AUG_SAVE_PREFIX,
            save_format="jpg",
            shuffle=True
        )

        # Generate required augmented images
        for _ in range(n_to_generate):
            try:
                next(gen)  # generate and save new image
            except Exception as e:
                print(f"⚠️ Error during augmentation for {cls}: {e}")
                break  # will save an image as aug_<...>.jpg in the class folder

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


# ========= Knowledge Distillation =========
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=2.0, alpha=0.5, teacher_indices=None):
        """
        teacher_indices: list/1D tensor of column indices to slice teacher logits
                         to align with student's class order.
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_indices = None
        if teacher_indices is not None:
            self.teacher_indices = tf.convert_to_tensor(teacher_indices, dtype=tf.int32)

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def _align_teacher_logits(self, teacher_preds):
        if self.teacher_indices is None:
            return teacher_preds
        return tf.gather(teacher_preds, self.teacher_indices, axis=1)

    def train_step(self, data):
        # Handle (x, y) and (x, y, sample_weight) cases
        if isinstance(data, tuple) or isinstance(data, list):
            if len(data) == 2:
                x, y = data
                sample_weight = None
            elif len(data) == 3:
                x, y, sample_weight = data
            else:
                # Keras sometimes passes Dataset elements in a nested form; try unpacking first element
                x, y = data[0], data[1]
                sample_weight = None
        else:
            # Fallback: assume data yields exactly two
            x, y = data
            sample_weight = None

        # Teacher predictions (frozen)
        teacher_preds = tf.stop_gradient(self.teacher(x, training=False))
        teacher_preds = self._align_teacher_logits(teacher_preds)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            # student (hard) loss supports sample_weight
            student_loss = self.student_loss_fn(y, student_preds, sample_weight=sample_weight)

            # distillation (soft) loss - do not pass sample_weight here (soft labels applied per-sample is uncommon)
            t = self.temperature
            distill_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_preds / t, axis=1),
                tf.nn.softmax(student_preds / t, axis=1)
            )

            loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

        # Compute gradients and apply them to student only
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # Accuracy (plain argmax) — keep this simple and consistent with your other code
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(student_preds, axis=1), tf.argmax(y, axis=1)), tf.float32)
        )

        # Update compiled metrics (if any)
        try:
            self.compiled_metrics.update_state(y, student_preds, sample_weight=sample_weight)
            metrics_result = {m.name: m.result() for m in self.metrics}
        except Exception:
            metrics_result = {}

        result = {
            "loss": loss,
            "student_loss": student_loss,
            "distillation_loss": distill_loss,
            "accuracy": acc,
        }
        result.update(metrics_result)
        return result

    def test_step(self, data):
        # Handle (x, y) and (x, y, sample_weight) cases
        if isinstance(data, tuple) or isinstance(data, list):
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

        # Update metrics
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
        balance_and_augment(dataset_path)
    except Exception as e:
        print("⚠️ balance_and_augment() failed:", e)
        # proceed anyway

    # Data generators
    datagen = ImageDataGenerator(
        validation_split=0.30 if new_files else (0.30 if len(list(Path(DATA_DIR).rglob('*.*'))) < 600 else 0.20),
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest"
    )
    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Class counts
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
            optimizer=Adam(learning_rate=1e-4),
            metrics=["accuracy"],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False),
            SaveBestStudent(student, BEST_MODEL_PATH),
        ]

        print("🔧 Incremental training (new-only) with knowledge distillation...")
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

        # --- After training, check predictions ---
        # Map indices to class names
        idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

        # Get a batch of validation data
        x_batch, y_batch = next(val_gen)
        preds = model.predict(x_batch)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(y_batch, axis=1)

        for i in range(len(pred_classes)):
            print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

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
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode="nearest"
        )
        train_gen = datagen_full.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='training', shuffle=True
        )
        val_gen = datagen_full.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', subset='validation', shuffle=False
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

        # --- After training, check predictions ---
        # Map indices to class names
        idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

        # Get a batch of validation data
        x_batch, y_batch = next(val_gen)
        preds = model.predict(x_batch)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(y_batch, axis=1)

        for i in range(len(pred_classes)):
            print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

        print("🔧 Phase 2: Fine-tuning ResNet tail...")
        unfreeze_for_finetune(model, fine_tune_at=FINE_TUNE_AT)
        history2 = model.fit(
            train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS,
            class_weight=dict(enumerate(
                compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
            )),
            callbacks=callbacks
        )

        # --- After training, check predictions ---
        # Map indices to class names
        idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

        # Get a batch of validation data
        x_batch, y_batch = next(val_gen)
        preds = model.predict(x_batch)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(y_batch, axis=1)

        for i in range(len(pred_classes)):
            print(f"True: {idx_to_class[true_classes[i]]}, Pred: {idx_to_class[pred_classes[i]]}")

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

    # Confusion Matrix on validation split
    try:
        val_gen.reset()
        y_true = val_gen.classes
        y_pred_probs = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

       # Confusion matrix (robust normalization)
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

        # Also save normalized matrix image
        plt.figure(figsize=(8, 6))
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
        disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca())
        plt.title("Confusion Matrix (normalized by true class)")
        plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_norm.png"))
        plt.close()

        np.set_printoptions(precision=3, suppress=True)
        print("\nNormalized Confusion Matrix:\n", cm_norm)
    except Exception as e:
        print("⚠️ Could not compute confusion matrix:", e)

    # Cleanup temp dataset
    if os.path.exists(NEW_ONLY_DIR):
        shutil.rmtree(NEW_ONLY_DIR)
        print("🧹 Cleaned new-only dataset.")

    return True


if __name__ == "__main__":
    ok = train_model()
    print("🎉 Training completed!" if ok else "❌ Training failed.")
