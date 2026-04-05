import argparse
import json
import numpy as np
import cv2
from tensorflow import keras


# Load the tag mapping table and convert the string keys in the JSON to integers.
# For example, convert {“0”: “Alice”, “1”: “Bob”} to {0: ‘Alice’, 1: “Bob”}
def load_label_map(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# Read and preprocess the input image so that its format exactly matches that used during training
def preprocess_image(img_path, img_size=128):
    # 1. Load an image (OpenCV loads images in BGR format by default)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # 2. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Resize the image to the dimensions required by the model (default: 128x128)
    img = cv2.resize(img, (img_size, img_size))
    # 4. Normalize pixel values (scale 0–255 to 0–1 to match the training data)
    img = img.astype("float32") / 255.0
    # 5. Expand the dimensions. The model input must be four-dimensional (batch_size, height, width, channels)
    # Since we only have one image, we need to add a batch dimension at the 0th position, resulting in (1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
    return img


def main():
    # ── Parsing Command-Line Arguments ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--model", default="models/face_recognition_model.keras")
    parser.add_argument("--label-map", default="models/label_map.json")
    args = parser.parse_args()

    # ── 2. Image Processing ─────────────────────────────────────────────────────────
    print("[INFO] Loading model...")
    model = keras.models.load_model(args.model)

    label_map = load_label_map(args.label_map)

    # ── 2. Image Processing ─────────────────────────────────────────────────────────
    print("[INFO] Processing image...")
    img = preprocess_image(args.image)

    # ── 3. Model Prediction ─────────────────────────────────────────────────────────
    # predict() returns a two-dimensional array [[prob1, prob2, ...]]
    # We only need the first row of data (since there is only one image in the batch), so we add [0]
    preds = model.predict(img, verbose=0)[0]

    # Get the maximum value in an array of probability values, along with its corresponding index
    max_prob = preds.max()
    pred_idx = preds.argmax()

    # --- Below which assigns a category regardless of how low the probability is) ---
    name = label_map[pred_idx]

    # if < threshold => unknow
    # if max_prob >= args.threshold:
    #     name = label_map[pred_idx]
    # else:
    #     name = "Unknown"

    print(f"[DEBUG] Probabilities: {preds}")
    print(f"[DEBUG] Max prob: {max_prob:.3f}")

    print(f"\n Prediction: {name} ({max_prob:.2f})")


if __name__ == "__main__":
    main()

