# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMP 6721 (Applied AI) final project at Concordia University, Winter 2026. A real-time face detection and recognition desktop application built with a custom Keras CNN trained from scratch on a team-collected dataset.

## Current Status

- **Code:** All scripts complete (`capture_images.py`, `preprocess.py`, `cnn_model.py`, `train.py`, `evaluate.py`, `app.py`).
- **Data collection:** Booth — 100 images, Yikai — 100 images. Celebrity images (Angelina_Jolie, Jennifer_Lawrence, Johnny_Depp, Leonardo_DiCaprio, Robert_Downey_Jr) sourced externally and placed directly in `data/splits/`.
- **Next step:** Run `train.py` — splits are populated and augmented, ready to train.

## Commands

All commands must be run from the **project root**.

```bash
# Install dependencies
pip install -r requirements.txt          # pip
conda env create -f environment.yml      # OR conda/micromamba (env name: face)

# 1. Collect images (run once per person)
python src/data_collection/capture_images.py --person "Name" --count 50

# 2. Preprocess (crop → resize → split → augment)
python src/preprocessing/preprocess.py

# 3. Train
python src/model/train.py --epochs 40 --batch-size 32

# 4. Evaluate + threshold analysis
python src/model/evaluate.py --threshold 0.60

# 5. Run real-time app
python src/app/app.py --threshold 0.60 --camera 0

# Quick model architecture sanity check
python src/model/cnn_model.py
```

## Architecture

The pipeline is strictly sequential — each stage produces outputs consumed by the next:

```
capture_images.py  →  data/raw/<person>/
       ↓
preprocess.py      →  data/processed/<person>/   (face crops, 128×128 PNG)
                   →  data/splits/{train,val,test}/<person>/   (70/15/15 split)
                      + 4× offline augmentation in train/
       ↓
train.py           →  models/face_recognition_model.keras
                   →  models/label_map.json        ({int_index: "person_name"})
                   →  models/training_history.png
       ↓
evaluate.py        →  models/confusion_matrix.png
                   →  models/threshold_curve.png
       ↓
app.py             (real-time inference, reads model + label_map)
```

### Key design decisions

**`train.py` imports `cnn_model` as a sibling module** (`from cnn_model import build_model`), so it must be run from the project root (not from inside `src/model/`). `app.py` manually inserts the project root into `sys.path` at startup to resolve imports.

**Two-stage augmentation:** `preprocess.py` generates 4× offline augmented PNGs on disk (PIL: flip, rotation ±20°, brightness/contrast jitter). `train.py` additionally applies light online augmentation via `ImageDataGenerator` (flip, rotation ±10°, brightness ±15%). Re-running `preprocess.py` is safe — it wipes `data/processed/` and `data/splits/` at startup before rebuilding.

**Unknown detection** is threshold-based, not a trained class. The model outputs softmax probabilities over the N known identities; any face whose max probability falls below `--threshold` is labelled "Unknown". `evaluate.py --threshold` plots the accuracy-vs-coverage trade-off to help pick the right value.

**Label map** (`models/label_map.json`) maps `int → person_name` (keys are stored as strings in JSON; `evaluate.py` and `app.py` both cast them to `int` on load). It is written at the start of training so inference can work even if training is interrupted.

### CNN architecture (`src/model/cnn_model.py`)

4 convolutional blocks doubling filters (32 → 64 → 128 → 256), each with 2× Conv2D + BatchNorm + ReLU + MaxPool + Dropout, followed by a refinement block (256, no pool), GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.4) → Dense(N_classes, Softmax). Input: 128×128×3 RGB float32 normalised to [0, 1].

Training callbacks: `ModelCheckpoint` (monitor val_accuracy), `ReduceLROnPlateau` (patience 5, factor 0.5), `EarlyStopping` (patience 10).

## Important Constants

Constants that must stay consistent across scripts (changing one requires updating all):

| Constant | Value | Defined in |
|----------|-------|-----------|
| `IMG_SIZE` | 128 | `preprocess.py`, `cnn_model.py`, `train.py`, `app.py` |
| `TRAIN_RATIO / VAL_RATIO` | 0.70 / 0.15 | `preprocess.py` |
| `AUGMENT_FACTOR` | 4 | `preprocess.py` |
| Default threshold | 0.60 | `evaluate.py`, `app.py` |

## Data & Model Files

`data/raw/`, `data/processed/`, `data/splits/`, and `models/*.keras` / `models/*.h5` are all gitignored. Only `models/label_map.json` and plot PNGs should be committed if needed. Each person's raw images must go in `data/raw/<ExactPersonName>/` — folder names become class labels.
