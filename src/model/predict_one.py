import argparse
import json
import numpy as np
import cv2
from tensorflow import keras


def load_label_map(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def preprocess_image(img_path, img_size=128):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--model", default="models/face_recognition_model.keras")
    parser.add_argument("--label-map", default="models/label_map.json")
    args = parser.parse_args()

    print("[INFO] Loading model...")
    model = keras.models.load_model(args.model)

    label_map = load_label_map(args.label_map)

    print("[INFO] Processing image...")
    img = preprocess_image(args.image)

    preds = model.predict(img, verbose=0)[0]

    # max_prob = preds.max()
    # pred_idx = preds.argmax()

    # print(f"[DEBUG] Probabilities: {preds}")
    # print(f"[DEBUG] Max prob: {max_prob:.3f}")

    # if max_prob >= args.threshold:
    #     name = label_map[pred_idx]
    # else:
    #     name = "Unknown"

    max_prob = preds.max()
    pred_idx = preds.argmax()
    name = label_map[pred_idx]

    print(f"[DEBUG] Probabilities: {preds}")
    print(f"[DEBUG] Max prob: {max_prob:.3f}")

    print(f"\n✅ Prediction: {name} ({max_prob:.2f})")


if __name__ == "__main__":
    main()

