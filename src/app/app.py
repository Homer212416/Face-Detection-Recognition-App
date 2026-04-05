"""
app.py
------
Real-time face detection & recognition application.

Requirements: trained model at models/face_recognition_model.keras
              label map at models/label_map.json

Usage:
    python src/app/app.py [--threshold 0.60] [--camera 0]

Controls:
    q  – quit
    t  – raise threshold by 0.05
    g  – lower threshold by 0.05
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Lazy-import TensorFlow to keep startup message brief
import tensorflow as tf
from tensorflow import keras

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
IMG_SIZE = 128
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Colour palette (BGR)
COLOR_KNOWN = (50, 205, 50)       # green
COLOR_UNKNOWN = (0, 0, 220)       # red
COLOR_HUD = (255, 220, 0)         # cyan-yellow


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face recognition app")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Minimum confidence to label a face as known (default 0.60)")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "face_recognition_model.keras"))
    p.add_argument("--label-map", default=os.path.join(MODELS_DIR, "label_map.json"))
    return p.parse_args()


def load_label_map(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, convert to RGB, normalise, and add batch dim."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_norm = face_resized.astype(np.float32) / 255.0
    return np.expand_dims(face_norm, axis=0)  # (1, H, W, 3)


def draw_face_box(frame, x, y, w, h, label, confidence, threshold):
    """Overlay bounding box and label on the frame in-place."""
    is_known = label != "Unknown"
    color = COLOR_KNOWN if is_known else COLOR_UNKNOWN
    thickness = 2

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Label background
    label_text = f"{label} ({confidence:.0%})" if is_known else "Unknown"
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 4, y), color, -1)

    # Label text
    cv2.putText(
        frame, label_text,
        (x + 2, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


def draw_hud(frame, threshold, fps, n_faces):
    h, w = frame.shape[:2]
    lines = [
        f"Threshold : {threshold:.2f}  (t/g to adjust)",
        f"Faces     : {n_faces}",
        f"FPS       : {fps:.1f}",
        "Press  q  to quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            COLOR_HUD, 1, cv2.LINE_AA,
        )


def main():
    args = parse_args()
    threshold = args.threshold

    # ── Load model & labels ───────────────────────────────────────────────────
    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model not found: {args.model}\n"
                 "        Run  python src/model/train.py  first.")
    if not os.path.exists(args.label_map):
        sys.exit(f"[ERROR] Label map not found: {args.label_map}")

    print("[INFO] Loading model …")
    # Compatibility shim: older Keras saved renorm* kwargs that current Keras 3 dropped.
    # Monkey-patch from_config so load_model can deserialise the saved architecture.
    _orig_bn_from_config = keras.layers.BatchNormalization.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _bn_from_config_compat(cls, config):
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            config.pop(key, None)
        return _orig_bn_from_config(cls, config)

    keras.layers.BatchNormalization.from_config = _bn_from_config_compat
    try:
        model = keras.models.load_model(args.model)
    finally:
        keras.layers.BatchNormalization.from_config = classmethod(_orig_bn_from_config)
    label_map = load_label_map(args.label_map)
    print(f"[INFO] Classes: {list(label_map.values())}")

    # ── Face detector ─────────────────────────────────────────────────────────
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {args.camera}")

    print(f"[INFO] Starting real-time recognition (threshold={threshold:.2f}) …")
    print("[INFO] Press  q  to quit |  t  raise threshold |  g  lower threshold")

    # FPS tracking
    fps = 0.0
    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # ── Face detection ────────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        n_faces = len(detections) if isinstance(detections, np.ndarray) else 0

        # ── Recognition ───────────────────────────────────────────────────────
        if n_faces > 0:
            face_tensors = []
            for x, y, w, h in detections:
                # Add 20% padding to match preprocess.py crop convention
                padding = int(0.20 * max(w, h))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue
                face_tensors.append(preprocess_face(face_bgr))

            if face_tensors:
                batch = np.concatenate(face_tensors, axis=0)   # (N, H, W, 3)
                preds = model.predict(batch, verbose=0)         # (N, n_classes)

                for i, (x, y, w, h) in enumerate(detections):
                    if i >= len(preds):
                        break
                    confidence = float(preds[i].max())
                    pred_idx = int(preds[i].argmax())

                    if confidence >= threshold:
                        label = label_map.get(pred_idx, "Unknown")
                    else:
                        label = "Unknown"

                    draw_face_box(frame, x, y, w, h, label, confidence, threshold)

        # ── HUD ───────────────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed > 0:
            fps = frame_count / elapsed

        draw_hud(frame, threshold, fps, n_faces)
        cv2.imshow("Face Recognition – q to quit", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            threshold = min(0.99, round(threshold + 0.05, 2))
            print(f"[INFO] Threshold → {threshold:.2f}")
        elif key == ord("g"):
            threshold = max(0.01, round(threshold - 0.05, 2))
            print(f"[INFO] Threshold → {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
