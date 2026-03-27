
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

    # load the cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    os.listdir(RAW_DIR)
        



# def step2_split():

# def step3_augment_train():


if __name__ == "__main__":
    for directory in [PROCESSED_DIR, SPLITS_DIR]: 
        if os.path.exists(directory):
            shutil.rmtree(directory) # clear the data if they already exist
        os.makedirs(directory)
    step1_crop_all()
    #step2_split()
    #step3_augment_train()


