import tensorflow as tf
import numpy as np
from PIL import Image
import io

MODEL_PATH = "models/model.tflite"
LABELS_PATH = "models/labels.txt"
IMG_SIZE = (224, 224)

# Load labels
with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0  # normalize
    return np.expand_dims(img, axis=0)  # add batch dim

def classify_image(image_bytes: bytes) -> str:
    try:
        input_data = preprocess(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = int(np.argmax(output))
        confidence = float(np.max(output))
        label = class_labels[predicted_idx]
        return f"{label} ({confidence:.2f})"
    except Exception as e:
        return f"error: {str(e)}"
