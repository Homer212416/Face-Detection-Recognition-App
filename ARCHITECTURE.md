# Architecture — Face Detection & Recognition App
**COMP 6721 Applied AI · Concordia University · Winter 2026**

---

## Pipeline

```
capture_images.py  →  data/raw/<Person>/
preprocess.py      →  data/processed/<Person>/   (128×128 PNG crops)
                   →  data/splits/{train,val,test}/<Person>/  +  4× augmentation in train/
train.py           →  models/face_recognition_model.keras
                   →  models/label_map.json
evaluate.py        →  models/confusion_matrix.png
                   →  models/threshold_curve.png
app.py             →  live webcam window
```

Each stage produces files consumed by the next. Run from the **project root**.

---

## Work Division

| | Part A — Data Pipeline | Part B — Model & App |
|---|---|---|
| **Files** | `capture_images.py`, `preprocess.py` | `cnn_model.py`, `train.py`, `evaluate.py`, `app.py` |
| **Libraries** | OpenCV, Pillow | TensorFlow/Keras, Matplotlib, scikit-learn |

Part A must finish before Part B can train, but both can be developed in parallel.

---

## Shared Constants

| Constant | Value | Used in |
|---|---|---|
| `IMG_SIZE` | `128` | preprocess, cnn_model, train, app |
| `TRAIN_RATIO` / `VAL_RATIO` | `0.70` / `0.15` | preprocess |
| `AUGMENT_FACTOR` | `4` | preprocess |
| Default `threshold` | `0.60` | evaluate, app |

---

## Part A — Data Pipeline

### `capture_images.py`
```bash
python src/data_collection/capture_images.py --person "Alice" --count 50
```
Opens webcam, shows live Haar cascade bounding boxes (drawn on a `display` copy, never on the saved frame), auto-captures one frame every `--interval` seconds until `--count` images are saved to `data/raw/<Person>/`. Filenames are `<Person>_<NNNN>.jpg`, indexed from the existing file count so re-runs don't overwrite.

**Args:** `--person` (required), `--count` (default 50), `--camera` (default 0), `--interval` (default 0.5s)  
**Key:** `q` to quit.

---

### `preprocess.py`
```bash
python src/preprocessing/preprocess.py
```
Three sequential steps:

**`step1_crop_all()`** — For each image in `data/raw/<Person>/`: detect face with Haar cascade, add 20% padding, clamp to image bounds, resize to `128×128`, save as PNG to `data/processed/<Person>/`. Falls back to the full image if no face is detected.

**`step2_split()`** — Wipes `data/splits/`, then for each person: shuffles with `RANDOM_SEED=42`, slices 70/15/15, copies (not moves) files into `data/splits/{train,val,test}/<Person>/`.

**`step3_augment_train()`** — For each original (non-`_aug`) file in `data/splits/train/<Person>/`, generates `AUGMENT_FACTOR=4` variants. Each variant opens the image fresh (prevents transform stacking) and applies: 50% horizontal flip → rotation ±20° → brightness 0.7–1.3× → contrast 0.8–1.2×. Saved as `<stem>_aug{i:02d}.png` alongside the originals.

> **Note:** Re-running `preprocess.py` is safe — it wipes `data/processed/` and `data/splits/` at startup before rebuilding.

**Expected output (100 images/person):**
```
data/processed/<Person>/     → 100 PNGs
data/splits/train/<Person>/  → 70 originals + 280 _aug = 350 total
data/splits/val/<Person>/    → 15 files
data/splits/test/<Person>/   → 15 files
```

---

## Part B — Model & App

### `cnn_model.py`
Defines `build_model(n_classes, img_size=128, dropout_rate=0.4, lr=3e-4)`. Imported by `train.py`; run directly for a `model.summary()` sanity check.

**Architecture:**
```
Input (128×128×3)
→ 4× conv_block(32→64→128→256):  Conv2D→BN→ReLU→Conv2D→BN→ReLU→MaxPool→Dropout
→ Refinement block (256):         Conv2D→BN→ReLU→Conv2D→BN→ReLU  (no pool, stays at 8×8)
→ GlobalAveragePooling2D
→ Dense(128, relu) → Dropout(0.4) → Dense(N_classes, softmax)
```
L2(1e-4) regularisation on all conv/dense layers. `use_bias=False` on Conv2D (BN provides shift). Compiled with Adam + `categorical_crossentropy`.

---

### `train.py`
```bash
python src/model/train.py --epochs 40 --batch-size 32
```
Loads splits via `ImageDataGenerator.flow_from_directory`, saves `label_map.json` before training starts, calls `build_model`, trains with three callbacks (ModelCheckpoint on `val_accuracy`, ReduceLROnPlateau on `val_loss`, EarlyStopping on `val_accuracy` patience=10), plots loss/accuracy curves.

**Note:** Imports `cnn_model` as a sibling module — must be run from the project root.

---

### `evaluate.py`
```bash
python src/model/evaluate.py --threshold 0.60
```
Runs inference on `data/splits/test/`, prints a classification report (no threshold), then reports unknown-rejection stats at the given threshold. Saves `confusion_matrix.png` and `threshold_curve.png` (accuracy + coverage vs. threshold on a dual-axis chart).

---

### `app.py`
```bash
python src/app/app.py --threshold 0.60 --camera 0
```
Loads model + label map, opens webcam at 1280×720, runs Haar cascade detection each frame, batches all face crops into a single `model.predict` call, labels faces green (known, with confidence %) or red (Unknown). HUD shows threshold, face count, FPS.

**Keys:** `q` quit · `t` raise threshold +0.05 · `g` lower threshold −0.05

---

## Integration Notes

- `IMG_SIZE=128` must be consistent across preprocess, cnn_model, train, app, and evaluate.
- All scripts must be run from the **project root**.
- `data/raw/` folder names become class labels — capitalisation must match exactly.
- `data/splits/` must be populated before training; `label_map.json` is written at the start of `train.py` so inference works even if training is interrupted.
