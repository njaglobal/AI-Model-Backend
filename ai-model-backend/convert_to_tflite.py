import tensorflow as tf
import os

MODEL_H5_PATH = "models/model.h5"
TFLITE_MODEL_PATH = "models/model.tflite"

def convert_model():
    if not os.path.exists(MODEL_H5_PATH):
        print("❌ model.h5 not found.")
        return False

    model = tf.keras.models.load_model(MODEL_H5_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: smaller model

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"✅ model.tflite saved at {TFLITE_MODEL_PATH}")
    return True

if __name__ == "__main__":
    convert_model()
