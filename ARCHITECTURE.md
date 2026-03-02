# Project Architecture — Face Detection & Recognition App

**COMP 6721 Applied AI · Concordia University · Winter 2026**

---

## Overview

This project is a real-time face recognition desktop application. A custom Convolutional Neural Network (CNN) is trained from scratch on photos collected with a webcam. At runtime the app opens a live camera feed, detects every face in each frame, and labels it with the person's name (or "Unknown" if confidence is too low).

The full pipeline is **strictly sequential** — each stage produces files that the next stage reads:

```
capture_images.py
      │  saves raw JPGs
      ▼
 data/raw/<Person>/

preprocess.py
      │  detects faces, crops, resizes, splits 70/15/15, augments ×4
      ▼
 data/splits/{train,val,test}/<Person>/

train.py  ←  cnn_model.py
      │  fits the CNN, saves best checkpoint
      ▼
 models/face_recognition_model.keras
 models/label_map.json

evaluate.py
      │  plots confusion matrix + threshold curve
      ▼
 models/confusion_matrix.png
 models/threshold_curve.png

app.py
      │  opens webcam, detects + recognises in real time
      ▼
 (live window on screen)
```

---

## How the Work Is Divided

The project is split into two self-contained parts. Each programmer can work independently once the shared constants are agreed on. **Part A** must finish before Part B can begin training, but both parts can be developed and unit-tested in parallel.

| | Part A — Data Pipeline | Part B — Model & App |
|---|---|---|
| **Files** | `capture_images.py`, `preprocess.py` | `cnn_model.py`, `train.py`, `evaluate.py`, `app.py` |
| **Input** | Webcam / filesystem | `data/splits/` (produced by Part A) |
| **Output** | `data/raw/`, `data/splits/` | `models/*.keras`, real-time window |
| **Main libraries** | OpenCV, Pillow | TensorFlow/Keras, Matplotlib, scikit-learn |

---

## Shared Constants

These values are used by **both parts**. Agree on them once and never change them without telling your partner:

| Constant | Value | Where it is used |
|---|---|---|
| `IMG_SIZE` | `128` | preprocess.py, cnn_model.py, train.py, app.py |
| `TRAIN_RATIO` | `0.70` | preprocess.py |
| `VAL_RATIO` | `0.15` | preprocess.py |
| `AUGMENT_FACTOR` | `4` | preprocess.py |
| Default `threshold` | `0.60` | evaluate.py, app.py |

---

---

# Part A — Data Pipeline

**Assigned files:** `src/data_collection/capture_images.py` · `src/preprocessing/preprocess.py`

**Goal:** Collect raw face images from a webcam and turn them into clean, labelled, augmented dataset splits ready for training.

---

## Part A — File 1: `src/data_collection/capture_images.py`

### Purpose

An interactive script that opens the laptop's webcam and saves raw JPEG frames into `data/raw/<PersonName>/`. It must be run once per person.

### How to run

```bash
python src/data_collection/capture_images.py --person "Alice" --count 50
```

- `--person` — the person's name (becomes the folder name and the class label).
- `--count` — how many images to collect (default 50).
- `--camera` — camera index (default 0, i.e. the built-in webcam).
- `--auto-interval` — seconds between frames in auto mode (default 0.5 s).

### Controls while the window is open

| Key | Action |
|---|---|
| `SPACE` | Save current frame |
| `a` | Toggle auto-capture mode |
| `q` | Quit / finish this person |

### What the script does — step by step

1. **Parse arguments** with `argparse` and create the output folder `data/raw/<PersonName>/`.
2. **Find the next available image index** by counting files already in the folder, so re-runs never overwrite previous images.
3. **Open the webcam** with `cv2.VideoCapture`. Raise a `RuntimeError` if the camera cannot be opened.
4. **Load the Haar face detector** (`haarcascade_frontalface_default.xml`) for live preview only — the green bounding boxes drawn on screen help the user position their face, but the full raw frame (not just the crop) is what gets saved.
5. **Enter the capture loop:**
   - Read a frame from the camera.
   - Convert to grayscale, detect faces, draw a green rectangle around each detected face on the display copy.
   - Show a HUD: person name, progress counter, and mode (MANUAL / AUTO).
   - Wait 1 ms for a keypress:
     - `SPACE` → set a `should_capture` flag.
     - `a` → toggle `auto_mode`; in auto mode, `should_capture` becomes True every `auto_interval` seconds.
     - `q` → break out of the loop.
   - If `should_capture` is True, write the **original** frame (not the annotated one) to `data/raw/<PersonName>/<PersonName>_NNNN.jpg` and increment the counter.
6. **Release** the camera and close all OpenCV windows.

### Key implementation details for beginners

- Save `frame` (the original), not `display` (the one with rectangles drawn on it). Rectangles in the saved image would confuse the CNN later.
- Use `img_index = len(existing)` so new images continue from where the last run stopped.
- `cv2.waitKey(1) & 0xFF` — the `& 0xFF` masks the return value to a single byte so key comparisons work on all platforms.
- The folder name is the class label. Capitalisation must match exactly between teammates (e.g. `"Alice"` and `"alice"` would be treated as different people).

---

## Part A — File 2: `src/preprocessing/preprocess.py`

### Purpose

Takes the raw images in `data/raw/` and produces the three dataset splits (`train`, `val`, `test`) in `data/splits/`. It also performs offline data augmentation to artificially increase the training set size.

### How to run

```bash
python src/preprocessing/preprocess.py
```

> **Warning:** Do not run this script a second time without first clearing `data/processed/` and `data/splits/`. Otherwise, augmented copies from the first run will be augmented again, causing data leakage and duplicate images.

### What the script does — step by step

The script is organised into three functions called in order:

#### Step 1 — `step1_crop_all()`: Face detection and cropping

For every person folder in `data/raw/`:
1. Read each image with `cv2.imread`.
2. Convert to grayscale and run `face_cascade.detectMultiScale` to find face bounding boxes.
3. For each detected bounding box, add 20 % padding on all sides (clamped to image edges) and crop that region.
4. If **no face is detected**, fall back to using the entire image as the crop. This avoids silently discarding images.
5. Resize the crop to `IMG_SIZE × IMG_SIZE` (128 × 128 pixels).
6. Normalise pixel values to the range `[0.0, 1.0]` by dividing by 255.
7. Re-multiply by 255 and save as a PNG file in `data/processed/<Person>/`.

#### Step 2 — `step2_split()`: Train / val / test split

For each person in `data/processed/`:
1. List all PNG files and shuffle them with `random.seed(42)` (a fixed seed makes the split reproducible).
2. Take the first 70 % as training, the next 15 % as validation, and the remainder as test.
3. Copy (not move) files into `data/splits/{train,val,test}/<Person>/`.
4. Wipe and recreate the `data/splits/` folders at the start so previous runs are not mixed in.

#### Step 3 — `step3_augment_train()`: Offline augmentation

For every original image in `data/splits/train/<Person>/`:
1. Open the PNG with Pillow (PIL) and convert to RGB.
2. Generate `AUGMENT_FACTOR` (4) augmented copies by randomly applying:
   - **Horizontal flip** (50 % chance).
   - **Rotation** between −20° and +20°.
   - **Brightness** scaled between 0.7× and 1.3×.
   - **Contrast** scaled between 0.8× and 1.2×.
3. Save each copy as `<stem>_augNN.png` in the same training folder.

After step 3, a person with 35 training images will have 35 + 35×4 = **175 training images**.

### Key implementation details for beginners

- `random.seed(RANDOM_SEED)` in `step2_split` is important — always call it before `random.shuffle` so every run produces exactly the same split.
- The `_aug` substring in augmented filenames (`"_aug" not in f`) is used in `step3_augment_train` to skip augmented copies when re-running only augmentation — always include it in the naming convention.
- `shutil.copy` vs `shutil.move` — use `copy`, not `move`, because `data/processed/` is the canonical store. `data/splits/` should be regenerable at any time.
- The fallback (use entire image when no face is detected) means the dataset will never silently shrink. However, you should verify visually that captured images actually contain a face before running preprocessing.

---

---

# Part B — Model & App

**Assigned files:** `src/model/cnn_model.py` · `src/model/train.py` · `src/model/evaluate.py` · `src/app/app.py`

**Goal:** Define the CNN architecture, train it on the dataset that Part A produced, evaluate its accuracy, and deploy it as a real-time webcam application.

**Prerequisite:** `data/splits/` must be populated by Part A before training can start.

---

## Part B — File 1: `src/model/cnn_model.py`

### Purpose

Defines the CNN architecture and returns a compiled Keras model. It is imported by `train.py`; it is never run as part of the pipeline, but can be run directly for a quick sanity check.

### How to run (sanity check only)

```bash
python src/model/cnn_model.py
```

This builds a model with 5 dummy classes and prints `model.summary()`.

### CNN architecture

The network takes a **128 × 128 × 3** RGB image (pixel values in `[0, 1]`) and outputs a softmax probability vector of length `N_classes`.

```
Input: (128, 128, 3)

Block 1:  Conv2D(32, 3×3) → BatchNorm → ReLU
          Conv2D(32, 3×3) → BatchNorm → ReLU
          MaxPool2D(2×2)                    → (64, 64, 32)

Block 2:  Conv2D(64, 3×3) → BatchNorm → ReLU
          Conv2D(64, 3×3) → BatchNorm → ReLU
          MaxPool2D(2×2)                    → (32, 32, 64)

Block 3:  Conv2D(128, 3×3) → BatchNorm → ReLU
          Conv2D(128, 3×3) → BatchNorm → ReLU
          MaxPool2D(2×2)                    → (16, 16, 128)

Block 4:  Conv2D(256, 3×3) → BatchNorm → ReLU
          Conv2D(256, 3×3) → BatchNorm → ReLU
                                            → (16, 16, 256)

GlobalAveragePooling2D                      → (256,)
Dense(256, relu)
Dropout(0.4)
Dense(N_classes, softmax)                   → (N_classes,)
```

- **BatchNormalization** stabilises training by normalising each layer's activations. Always placed before the activation.
- **GlobalAveragePooling2D** replaces `Flatten` + large `Dense` to reduce parameters and overfitting.
- **Dropout(0.4)** randomly zeroes 40 % of neurons during training to prevent co-adaptation.
- **`use_bias=False`** on `Conv2D` layers — when BatchNorm follows, a bias term is redundant because BatchNorm has its own learnable shift parameter (`β`).

The model is compiled with:
- **Optimizer:** Adam, learning rate `1e-3`.
- **Loss:** `categorical_crossentropy` (standard for multi-class classification with one-hot labels).
- **Metric:** `accuracy`.

### Key implementation detail for beginners

`build_model` is a **function**, not a class. It is called with `n_classes` so the output layer size adjusts automatically as more people are added to the dataset. Never hard-code the number of classes.

---

## Part B — File 2: `src/model/train.py`

### Purpose

Loads the dataset splits, builds the model, trains it, saves the best checkpoint, and plots the training curves.

### How to run

```bash
python src/model/train.py --epochs 40 --batch-size 32
```

> Must be run from the **project root**, not from `src/model/`, because `train.py` imports `cnn_model` as a sibling module with `from cnn_model import build_model`.

### Outputs

| File | Description |
|---|---|
| `models/face_recognition_model.keras` | Best model checkpoint (highest val accuracy) |
| `models/label_map.json` | `{"0": "Alice", "1": "Bob", …}` |
| `models/training_history.png` | Loss and accuracy curves side by side |

### What the script does — step by step

1. **Parse arguments** (`--epochs`, `--batch-size`, `--img-size`, `--lr`).
2. **Create data generators** with `ImageDataGenerator`:
   - Training generator applies light *online* augmentation on top of the offline-augmented images: horizontal flip, ±10° rotation, ±15% brightness.
   - Validation generator applies only `rescale=1/255` (no augmentation — we want honest metrics).
   - Both generators use `flow_from_directory`, which reads the folder structure `data/splits/{train,val}/<Person>/` and automatically assigns integer labels based on alphabetical folder order.
3. **Save the label map immediately** — the JSON file maps `{int_index: "person_name"}` and is written before training starts so that `evaluate.py` and `app.py` can work even if training is interrupted.
4. **Build the model** by calling `build_model(n_classes, img_size)`.
5. **Define callbacks:**
   - `ModelCheckpoint` — saves the model only when `val_accuracy` improves.
   - `ReduceLROnPlateau` — halves the learning rate if `val_loss` does not improve for 5 consecutive epochs (minimum lr `1e-6`).
   - `EarlyStopping` — stops training if `val_accuracy` does not improve for 10 epochs and restores the best weights.
6. **Call `model.fit`** — Keras handles the training loop automatically.
7. **Plot training history** — two side-by-side subplots (accuracy and loss) saved as a PNG.

### Key implementation details for beginners

- `ImageDataGenerator.flow_from_directory` maps folder names to class indices alphabetically. This is why `save_label_map` reverses `class_indices` — the generator gives `{"Alice": 0, "Bob": 1}`, but inference needs `{0: "Alice", 1: "Bob"}`.
- `save_best_only=True` in `ModelCheckpoint` means the `.keras` file on disk is always the best epoch, even if later epochs are worse.
- `restore_best_weights=True` in `EarlyStopping` means the in-memory model is also reverted to the best epoch weights after the callback fires.

---

## Part B — File 3: `src/model/evaluate.py`

### Purpose

Loads the trained model and the held-out test set, produces a classification report, draws a confusion matrix, and plots an accuracy-vs-threshold curve to help choose the best confidence threshold.

### How to run

```bash
python src/model/evaluate.py --threshold 0.60
```

### Outputs

| File | Description |
|---|---|
| `models/confusion_matrix.png` | Heatmap of true vs. predicted labels |
| `models/threshold_curve.png` | Accuracy and coverage vs. confidence threshold |

### What the script does — step by step

1. **Load the model** with `keras.models.load_model`.
2. **Load the label map** from `models/label_map.json`; cast string keys to integers.
3. **Create a test generator** with `rescale=1/255` and `shuffle=False` (order must be preserved to align predictions with ground truth labels).
4. **Run inference:** `model.predict(test_gen)` returns an `(N, n_classes)` array of softmax probabilities.
5. **Classification report (no threshold):** Take `argmax` of each row to get the predicted class index, then use `sklearn.metrics.classification_report` to print per-class precision, recall, and F1.
6. **Threshold-gated results:** Any prediction whose maximum probability is below `--threshold` is treated as Unknown (index −1). Report how many images were rejected and the accuracy on the accepted ones.
7. **Confusion matrix:** Build with `sklearn.metrics.confusion_matrix` and visualise with `seaborn.heatmap`.
8. **Threshold curve:** Sweep thresholds from 0 to 0.99, compute accuracy and coverage (fraction of images above threshold) at each point, and plot both on a dual-axis chart. Use this chart to pick a threshold — high threshold means fewer mistakes but more images flagged Unknown.

### Key implementation details for beginners

- `shuffle=False` on the test generator is critical. Predictions come back in the same order as `test_gen.classes`, so they must not be shuffled.
- The confusion matrix uses raw top-1 predictions (no threshold) to show per-class error patterns; this is separate from the threshold analysis.
- `np.linspace(0.0, 0.99, 100)` generates 100 evenly spaced threshold values to sweep. The sweet spot is typically where the accuracy curve is high and the coverage curve has not dropped too steeply.

---

## Part B — File 4: `src/app/app.py`

### Purpose

Real-time desktop application. Opens a webcam, detects faces in every frame using the Haar cascade, crops each face, runs it through the trained CNN, and overlays a name label and confidence score on screen.

### How to run

```bash
python src/app/app.py --threshold 0.60 --camera 0
```

### Controls while running

| Key | Action |
|---|---|
| `q` | Quit |
| `t` | Raise threshold by 0.05 |
| `g` | Lower threshold by 0.05 |

### What the script does — step by step

1. **sys.path fix** — inserts the project root so Python can resolve imports from `src/`.
2. **Argument parsing** — threshold, camera index, model path, label map path.
3. **Load model and label map** — exits with a helpful error message if either file is missing.
4. **Open the webcam** with `cv2.VideoCapture`; set preferred resolution to 1280×720.
5. **Enter the main loop** (runs until `q` is pressed or the camera fails):
   a. **Grab a frame** from the camera.
   b. **Face detection** — convert to grayscale, run `detectMultiScale`, collect bounding boxes.
   c. **Batch preprocessing** — for each detected bounding box:
      - Clamp coordinates to frame boundaries to avoid out-of-bounds crops.
      - Convert BGR → RGB (OpenCV reads BGR; the model was trained on RGB images).
      - Resize to 128 × 128 and normalise to `[0, 1]`.
      - Stack all face tensors into a single batch `(N, 128, 128, 3)`.
   d. **Batch inference** — call `model.predict(batch, verbose=0)` once for all faces in the frame (more efficient than one call per face).
   e. **Label assignment** — for each face, if `max(softmax_probs) >= threshold` → use `label_map[argmax]`; otherwise → `"Unknown"`.
   f. **Draw overlays** — coloured bounding box (green = known, red = unknown), name + confidence badge above the box.
   g. **Draw HUD** — threshold value, face count, FPS in the top-left corner.
   h. **Key handling** — `q` breaks the loop; `t` / `g` adjust the threshold on the fly.
6. **Release** the camera and destroy all windows.

### Key implementation details for beginners

- **BGR vs RGB** — OpenCV reads images as BGR. The model was trained on images that PIL / Pillow saved as RGB. Always call `cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)` before feeding a frame to the model.
- **Batch predict, not one-by-one** — `np.concatenate(face_tensors, axis=0)` stacks all faces into one array, then `model.predict` runs once. This keeps the frame rate higher than calling `predict` in a loop.
- **`verbose=0`** in `model.predict` suppresses the Keras progress bar, which would spam the terminal.
- **FPS calculation** — `fps = frame_count / elapsed` is a running average from the start of the session. It starts inaccurate (low frame count) and stabilises over time.
- **Threshold range guard** — `min(0.99, ...)` and `max(0.01, ...)` prevent the threshold from hitting 0 or 1, which would make every face Unknown or always labelled as something.

---

## Integration Checklist

Use this checklist when merging Part A and Part B together for the first time:

- [ ] `IMG_SIZE = 128` matches in `preprocess.py`, `cnn_model.py`, `train.py`, `app.py`, and `evaluate.py`.
- [ ] `data/splits/train/<Person>/` is populated (run `preprocess.py` first).
- [ ] `models/label_map.json` exists (created at the start of `train.py`).
- [ ] `models/face_recognition_model.keras` exists (created by `train.py` after at least one epoch).
- [ ] Person folder names in `data/raw/` use exactly the same capitalisation as the names you intend to display in the app.
- [ ] All scripts are run from the **project root**, not from inside `src/`.
