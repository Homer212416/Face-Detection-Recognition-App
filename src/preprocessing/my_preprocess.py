import cv2
import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance

# path
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

# path
IMG_SIZE = 128
AUGMENT_FACTOR = 4

# path
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"


def step1_crop_all():
    haar_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    person_folder_names = os.listdir(RAW_DIR)
    # For each person, build the raw input path and the processed output path
    for p in person_folder_names:
        raw_input_path = os.path.join(RAW_DIR, p)
        processed_output_path = os.path.join(PROCESSED_DIR, p)
        os.makedirs(processed_output_path, exist_ok=True)
        # For each image of a person
        for i in os.listdir(raw_input_path):
            # read
            img = cv2.imread(os.path.join(raw_input_path,i))
            if img is None:
                continue
            # grey scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect and crop
            boxes = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(boxes) > 0:
                box = boxes[0]
                x, y, w, h = box
                padding = int(0.20 * max(w,h))
                y1 = max(0, y-padding)
                x1 = max(0, x-padding)
                y2 = min(y+h+padding, img.shape[0])
                x2 = min(x+w+padding, img.shape[1])
                crop = img[y1:y2, x1:x2]
            else:
                crop = img
            # save
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            stem = Path(i).stem
            out_path = os.path.join(processed_output_path, stem + ".png")
            cv2.imwrite(out_path, resized)


def step2_split(): # <-- TODO
    
    # wipe
    
    shutil.rmtree(SPLITS_DIR)
    os.makedirs(SPLITS_DIR)
    
    # shuffle
    # for each person
    
    random.seed(RANDOM_SEED)
    for p in os.listdir(PROCESSED_DIR):
        files = os.listdir(os.path.join(PROCESSED_DIR, p))
        random.shuffle(files)

        n_train = int(len(files) * TRAIN_RATIO)
        n_val = int(len(files) * VAL_RATIO)

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]





        


# def step3_augment_train():


if __name__ == "__main__":
    for directory in [PROCESSED_DIR, SPLITS_DIR]: 
        if os.path.exists(directory):
            shutil.rmtree(directory) # clear the data if they already exist
        os.makedirs(directory)
    step1_crop_all()
    #step2_split()
    #step3_augment_train()


