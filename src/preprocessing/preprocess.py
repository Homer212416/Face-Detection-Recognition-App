"""
preprocess.py
-------------
Pipeline:
  1. Detect & crop faces from raw images using OpenCV Haar cascade.
  2. Resize to IMG_SIZE x IMG_SIZE.
  3. Normalize pixel values to [0, 1].
  4. Apply data augmentation (offline) to the training split.
  5. Save train / val / test splits preserving class folder structure.

Usage:
    python src/preprocessing/preprocess.py

Outputs:
    data/processed/<person>/  – cropped & resized faces (pre-split)
    data/splits/train/<person>/
    data/splits/val/<person>/
    data/splits/test/<person>/
"""

import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = os.path.join("data", "splits")

IMG_SIZE = 128          # pixels (square)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO  (implicit)

AUGMENT_FACTOR = 4      # how many augmented copies to create per training image
RANDOM_SEED = 42

# ── Face detector ─────────────────────────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_and_crop(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Return list of face crops (BGR) found in *img_bgr*."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    crops = []
    for x, y, w, h in faces:
        # Add 20 % padding around the detected bounding box
        pad_w, pad_h = int(0.2 * w), int(0.2 * h)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_bgr.shape[1], x + w + pad_w)
        y2 = min(img_bgr.shape[0], y + h + pad_h)
        crops.append(img_bgr[y1:y2, x1:x2])
    return crops


def resize_and_normalize(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE²; return float32 array normalised to [0,1]."""
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    return resized.astype(np.float32) / 255.0


def save_array_as_image(arr: np.ndarray, path: str):
    """Save float [0,1] BGR array as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr_uint8 = (arr * 255).astype(np.uint8)
    cv2.imwrite(path, bgr_uint8)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_pil(pil_img: Image.Image) -> Image.Image:
    """Apply random augmentation to a PIL RGB image."""
    # Horizontal flip
    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotation ±20°
    angle = random.uniform(-20, 20)
    pil_img = pil_img.rotate(angle)

    # Brightness 0.7 – 1.3
    factor = random.uniform(0.7, 1.3)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    # Contrast 0.8 – 1.2
    factor = random.uniform(0.8, 1.2)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    return pil_img


def augment_image_file(src_path: str, dst_dir: str, base_name: str, n: int):
    """Generate *n* augmented copies of *src_path* into *dst_dir*."""
    img = Image.open(src_path).convert("RGB")
    for i in range(n):
        aug = augment_pil(img)
        out_path = os.path.join(dst_dir, f"{base_name}_aug{i:02d}.png")
        aug.save(out_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def step1_crop_all():
    """Detect faces in raw images and save crops to PROCESSED_DIR."""
    print("\n[STEP 1] Detecting and cropping faces …")
    persons = [p for p in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, p))]
    if not persons:
        raise FileNotFoundError(f"No person folders found in {RAW_DIR}")

    total_saved = 0
    for person in sorted(persons):
        person_raw = os.path.join(RAW_DIR, person)
        person_proc = os.path.join(PROCESSED_DIR, person)
        os.makedirs(person_proc, exist_ok=True)

        images = [
            f for f in os.listdir(person_raw)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        saved = 0
        for img_file in images:
            img_bgr = cv2.imread(os.path.join(person_raw, img_file))
            if img_bgr is None:
                continue
            crops = detect_and_crop(img_bgr)
            if not crops:
                # Fall back: use entire image if no face detected
                crops = [img_bgr]
            for idx, crop in enumerate(crops):
                arr = resize_and_normalize(crop)
                stem = os.path.splitext(img_file)[0]
                out_path = os.path.join(person_proc, f"{stem}_face{idx}.png")
                save_array_as_image(arr, out_path)
                saved += 1
        total_saved += saved
        print(f"  {person:20s}: {saved} face crops from {len(images)} raw images")

    print(f"  Total crops saved: {total_saved}")


def step2_split():
    """Split processed images into train / val / test."""
    print("\n[STEP 2] Splitting into train / val / test …")
    random.seed(RANDOM_SEED)

    # Clean previous splits
    for split in ("train", "val", "test"):
        split_dir = os.path.join(SPLITS_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    persons = [p for p in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, p))]

    for person in sorted(persons):
        person_proc = os.path.join(PROCESSED_DIR, person)
        files = [
            f for f in os.listdir(person_proc)
            if f.lower().endswith(".png")
        ]
        random.shuffle(files)

        n = len(files)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))

        splits = {
            "train": files[:n_train],
            "val": files[n_train: n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dst_dir = os.path.join(SPLITS_DIR, split_name, person)
            os.makedirs(dst_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(person_proc, f), os.path.join(dst_dir, f))

        print(
            f"  {person:20s}: train={len(splits['train'])}  "
            f"val={len(splits['val'])}  test={len(splits['test'])}"
        )


def step3_augment_train():
    """Augment training images offline."""
    print("\n[STEP 3] Augmenting training set …")
    train_dir = os.path.join(SPLITS_DIR, "train")
    persons = [p for p in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, p))]

    for person in sorted(persons):
        person_train = os.path.join(train_dir, person)
        originals = [
            f for f in os.listdir(person_train)
            if f.lower().endswith(".png") and "_aug" not in f
        ]
        aug_count = 0
        for f in originals:
            src = os.path.join(person_train, f)
            stem = os.path.splitext(f)[0]
            augment_image_file(src, person_train, stem, AUGMENT_FACTOR)
            aug_count += AUGMENT_FACTOR
        print(f"  {person:20s}: +{aug_count} augmented images")


def print_summary():
    print("\n[SUMMARY] Dataset split sizes:")
    for split in ("train", "val", "test"):
        split_dir = os.path.join(SPLITS_DIR, split)
        total = 0
        for person in os.listdir(split_dir):
            p_dir = os.path.join(split_dir, person)
            if os.path.isdir(p_dir):
                total += len(os.listdir(p_dir))
        print(f"  {split:6s}: {total} images")


if __name__ == "__main__":
    step1_crop_all()
    step2_split()
    step3_augment_train()
    print_summary()
    print("\n[DONE] Preprocessing complete.")
