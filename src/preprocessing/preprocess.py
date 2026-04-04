import cv2
import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance

# Constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

IMG_SIZE = 128
AUGMENT_FACTOR = 4

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"


def step1_crop_all():
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    person_folder_names = os.listdir(RAW_DIR)
    for p in person_folder_names:
        raw_input_path = os.path.join(RAW_DIR, p)
        processed_output_path = os.path.join(PROCESSED_DIR, p)
        os.makedirs(processed_output_path, exist_ok=True)
        for i in os.listdir(raw_input_path):
            img = cv2.imread(os.path.join(raw_input_path, i))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(boxes) > 0:
                box = boxes[0]
                x, y, w, h = box
                padding = int(0.20 * max(w, h))
                y1 = max(0, y - padding)
                x1 = max(0, x - padding)
                y2 = min(y + h + padding, img.shape[0])
                x2 = min(x + w + padding, img.shape[1])
                crop = img[y1:y2, x1:x2]
            else:
                crop = img
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            stem = Path(i).stem
            out_path = os.path.join(processed_output_path, stem + ".png")
            cv2.imwrite(out_path, resized)


def step2_split():
    shutil.rmtree(SPLITS_DIR)
    os.makedirs(SPLITS_DIR)

    random.seed(RANDOM_SEED)
    for p in os.listdir(PROCESSED_DIR):
        files = os.listdir(os.path.join(PROCESSED_DIR, p))
        random.shuffle(files)

        n_train = int(len(files) * TRAIN_RATIO)
        n_val = int(len(files) * VAL_RATIO)

        train_files = files[:n_train]
        val_files = files[n_train: n_train + n_val]
        test_files = files[n_train + n_val:]

        for split_name, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            dst_dir = os.path.join(SPLITS_DIR, split_name, p)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(PROCESSED_DIR, p, fname)
                shutil.copy(src, dst_dir)


def step3_augment_train():
    random.seed(RANDOM_SEED)
    for p in os.listdir(os.path.join(SPLITS_DIR, "train")):
        train_person_dir = os.path.join(SPLITS_DIR, "train", p)
        files = os.listdir(train_person_dir)
        # Only augment originals — skip already-augmented files to prevent
        # multiplying the set on re-runs and introducing near-duplicates
        originals = [f for f in files if "_aug" not in f]
        for fname in originals:
            filepath = os.path.join(train_person_dir, fname)
            for i in range(AUGMENT_FACTOR):
                # Open fresh each iteration so transforms don't stack
                img = Image.open(filepath).convert("RGB")

                # Horizontal flip (50% chance)
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                # Random rotation ±20°
                img = img.rotate(random.uniform(-20, 20))

                # Random brightness 0.7× – 1.3×
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

                # Random contrast 0.8× – 1.2×
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

                stem = Path(fname).stem
                out_name = f"{stem}_aug{i:02d}.png"
                out_path = os.path.join(train_person_dir, out_name)
                img.save(out_path)


if __name__ == "__main__":
    for directory in [PROCESSED_DIR, SPLITS_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    step1_crop_all()
    step2_split()
    step3_augment_train()
