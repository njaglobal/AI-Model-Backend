# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import cv2
# from exif import Image as ExifImage

# MODEL_PATH = "models/model.tflite"
# LABELS_PATH = "models/labels.txt"
# IMG_SIZE = (224, 224)

# # Load class labels
# with open(LABELS_PATH, "r") as f:
#     class_labels = [line.strip() for line in f.readlines()]

# # Load TFLite model
# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# def preprocess(image_bytes: bytes) -> np.ndarray:
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize(IMG_SIZE)
#     img = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(img, axis=0)

# # def is_likely_fake_photo(image_bytes: bytes) -> bool:
# #     is_fake = False

# #     # EXIF check
# #     try:
# #         exif = ExifImage(io.BytesIO(image_bytes))
# #         camera_make = getattr(exif, "make", "").lower()
# #         camera_model = getattr(exif, "model", "").lower()
# #         if any(k in camera_make for k in ["apple", "samsung", "xiaomi", "oppo", "vivo", "huawei", "google"]) or \
# #            any(k in camera_model for k in ["iphone", "galaxy", "pixel", "redmi", "android"]):
# #             is_fake = True
# #     except Exception as e:
# #         print("EXIF error or missing:", str(e))

# #     # Blurriness check
# #     try:
# #         pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
# #         np_img = np.array(pil_img)
# #         gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
# #         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
# #         if laplacian_var < 50:
# #             is_fake = True
# #     except Exception as e:
# #         print("Blurriness check failed:", str(e))

# #     return is_fake

# def is_likely_fake_photo(image_bytes: bytes) -> bool:
#     exif_checked = False
#     blur_checked = False
#     is_fake = False

#     # EXIF check
#     try:
#         exif = ExifImage(io.BytesIO(image_bytes))
#         camera_make = getattr(exif, "make", "").lower()
#         camera_model = getattr(exif, "model", "").lower()

#         if any(k in camera_make for k in ["apple", "samsung", "xiaomi", "oppo", "vivo", "huawei", "google"]) or \
#            any(k in camera_model for k in ["iphone", "galaxy", "pixel", "redmi", "android"]):
#             is_fake = True
#         exif_checked = True
#     except Exception as e:
#         print("EXIF check failed:", e)

#     # Blurriness check
#     try:
#         pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         np_img = np.array(pil_img)
#         gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
#         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

#         if laplacian_var < 50:
#             is_fake = True
#         blur_checked = True
#     except Exception as e:
#         print("Blurriness check failed:", e)

#     # Fallback decision: if neither check succeeded, treat as suspicious
#     if not exif_checked and not blur_checked:
#         print("⚠️ Both checks failed. Defaulting to fake.")
#         return True

#     return is_fake

# def is_ambiguous(output: np.ndarray, threshold: float = 0.15) -> bool:
#     # True if top 2 predictions are too close
#     sorted_probs = np.sort(output[0])
#     return (sorted_probs[-1] - sorted_probs[-2]) < threshold

# def classify_image(image_bytes: bytes) -> dict:
#     try:
#         input_data = preprocess(image_bytes)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output = interpreter.get_tensor(output_details[0]['index'])

#         predicted_idx = int(np.argmax(output))
#         confidence = float(np.max(output))
#         label = class_labels[predicted_idx]

#         print(f"Predicted label: {label}")
#         print(f"Confidence: {confidence:.4f}")

#         # Reject anything not labeled fire or road
#         if label not in ["fire", "road"]:
#             return {
#                 "label": None,
#                 "confidence": round(confidence, 4),
#                 "status": "invalid",
#                 "action": "reject",
#                 "reason": "Prediction is not a valid fire or road incident."
#             }

#         # Reject fake/photos of photos
#         if is_likely_fake_photo(image_bytes):
#             return {
#                 "label": label,
#                 "confidence": round(confidence, 4),
#                 "status": "invalid",
#                 "action": "reject",
#                 "reason": "Detected as a photo of a photo or fake"
#             }

#         # Reject if ambiguous even with high confidence
#         if is_ambiguous(output, threshold=0.15):
#             return {
#                 "label": None,
#                 "confidence": round(confidence, 4),
#                 "status": "invalid",
#                 "action": "reject",
#                 "reason": "Prediction is ambiguous — no clear label"
#             }

#         # Reject low confidence
#         if confidence < 0.75:
#             return {
#                 "label": label,
#                 "confidence": round(confidence, 4),
#                 "status": "invalid",
#                 "action": "reject",
#                 "reason": "Low confidence"
#             }

#         # Accept valid
#         return {
#             "label": label,
#             "confidence": round(confidence, 4),
#             "status": "valid",
#             "action": "accept"
#         }

#     except Exception as e:
#         return {
#             "error": str(e),
#             "label": None,
#             "confidence": 0,
#             "status": "error",
#             "action": "reject"
#         }


import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
from exif import Image as ExifImage

MODEL_PATH = "models/model.tflite"
LABELS_PATH = "models/labels.txt"
IMG_SIZE = (224, 224)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def is_likely_fake_photo(image_bytes: bytes) -> bool:
    exif_checked = False
    blur_checked = False
    is_fake = False

    try:
        exif = ExifImage(io.BytesIO(image_bytes))
        camera_make = getattr(exif, "make", "").lower()
        camera_model = getattr(exif, "model", "").lower()
        if any(k in camera_make for k in ["apple", "samsung", "xiaomi", "oppo", "vivo", "huawei", "google"]) or \
           any(k in camera_model for k in ["iphone", "galaxy", "pixel", "redmi", "android"]):
            is_fake = True
        exif_checked = True
    except Exception as e:
        print("EXIF check failed:", e)

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            is_fake = True
        blur_checked = True
    except Exception as e:
        print("Blurriness check failed:", e)

    if not exif_checked and not blur_checked:
        print("⚠️ Both checks failed. Defaulting to fake.")
        return True

    return is_fake

def is_ambiguous(output: np.ndarray, threshold: float = 0.15) -> bool:
    sorted_probs = np.sort(output[0])
    return (sorted_probs[-1] - sorted_probs[-2]) < threshold

def classify_image(image_bytes: bytes) -> dict:
    try:
        # Step 1: Preprocess and predict
        input_data = preprocess(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        predicted_idx = int(np.argmax(output))
        confidence = float(np.max(output))
        label = class_labels[predicted_idx]

        print(f"Predicted label: {label}")
        print(f"Confidence: {confidence:.4f}")


        if label == "none-accident":
            return {
                "label": label,
                "confidence": round(confidence, 4),
                "status": "invalid",
                "action": "reject",
                "reason": "Image does not indicate a fire or road incident."
            }
        # Step 2: Check if the photo is likely fake
        if is_likely_fake_photo(image_bytes):
            return {
                "label": label,
                "confidence": round(confidence, 4),
                "status": "invalid",
                "action": "reject",
                "reason": "Detected as a photo of a photo or manipulated"
            }

        # Step 3: Reject if prediction is ambiguous
        if is_ambiguous(output, threshold=0.15):
            return {
                "label": "none-accident",
                "confidence": round(confidence, 4),
                "status": "invalid",
                "action": "reject",
                "reason": "Unsure result — needs human review"
            }

        # Step 4: Reject if confidence is too low
        if confidence < 0.75:
            return {
                "label": label,
                "confidence": round(confidence, 4),
                "status": "invalid",
                "action": "reject",
                "reason": "Invalid image - undertiminable"
            }

        # Step 5: Valid prediction (can be fire, road, or none-accident)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "status": "valid",
            "action": "accept",
            "reason": "Valid"
        }

    except Exception as e:
        return {
            "error": str(e),
            "label": None,
            "confidence": 0,
            "status": "error",
            "action": "reject",
            "reason": "Invalid Image"
        }