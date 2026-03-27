import cv2
import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance


IMG_SIZE = 128
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
AUGMENT_FACTOR = 4
RANDOM_SEED = 42

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"


def step1_crop_all():
    haar_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    raw_person_ls = os.listdir(RAW_DIR)

    ls_size = len(raw_folder_ls)

    raw_input_person_path_ls = []
    processed_output_person_path_ls = []
    
    for i in range(ls_size):
        raw_input_person_path = os.path.join(RAW_DIR, raw_person_ls[i])
        processed_output_person_path = os.path.join(PROCESSED_DIR, raw_person_ls[i])
        raw_input_person_ls.append(raw_input_person_path)
        processed_output_person_ls.append(processed_output_person_path)
        # create the output folder
        os.makedirs(processed_output_dir, exist_ok=True)
        # loop over the raw files
        raw_input_person_path_ls = os.listdir(raw_input_person_path)
        for j in range(len(raw_input_person_path_ls)):
            img = cv2.imread(raw_input_person_path_ls[j])
            if not img:
                continue
            img = cv2.cvtColor(img,cv2.COLOE_BGR2GRAY)
            boxes = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
            # at least one face is detected
            if len(boxes) > 0:  # <------------

    
    
def step2_split():

def step3_augment_train():


if __name__ == "__main__":
    for directory in [PROCESS_DIR, SPLITS_DIR]: 
        if os.path.exists(directory):
            shutil.rmtree(directory) # clear the data if they already exist
        os.makedirs(directory)
    step1_crop_all()
    step2_split()
    step3_augment_train()


