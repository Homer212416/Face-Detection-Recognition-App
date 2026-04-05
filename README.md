# Face Detection & Recognition App
**COMP 6721 – Applied Artificial Intelligence | Final Project | Winter 2026**

A real-time face detection and recognition application built from scratch using a custom CNN trained on a team-collected dataset.

---

## Quick Start (Run the App)

If a trained model is already provided, you can run the app directly without collecting data or retraining.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure `models/face_recognition_model.keras` and `models/label_map.json` are present.

3. Launch the app:
   ```bash
   python src/app/app.py --threshold 0.60 --camera 0
   ```

**Live controls:**
| Key | Action |
|-----|--------|
| `t` | Raise confidence threshold by 0.05 |
| `g` | Lower confidence threshold by 0.05 |
| `q` | Quit |

---

## Project Structure

```
Face-Detection-Recognition-App/
├── data/
│   ├── raw/                  # Raw captured images, one folder per person
│   ├── processed/            # Cropped & resized face images
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
├── models/                   # Saved model weights, label map, training plots
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── data_collection/
│   │   └── capture_images.py   # Webcam image capture tool
│   ├── preprocessing/
│   │   └── preprocess.py       # Face crop, resize, augment, split
│   ├── model/
│   │   ├── cnn_model.py        # CNN architecture (Keras)
│   │   ├── train.py            # Training script
│   │   └── evaluate.py         # Evaluation & metrics
│   └── app/
│       └── app.py              # Real-time application
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup

### Option A – pip (virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B – Conda

```bash
conda env create -f environment.yml
conda activate face-recognition
```

---

## Step-by-Step Workflow

### 1. Collect Images

Run for each person in your dataset (minimum 5 people, 30 images each):

```bash
python src/data_collection/capture_images.py --person "Alice" --count 50
python src/data_collection/capture_images.py --person "Bob"   --count 50
# … repeat for every person
```

The script auto-captures one frame every 0.5 seconds. Press `q` to quit once enough images are collected.

Images are saved to `data/raw/<person_name>/`.

> **Tip:** Vary your pose, lighting, angle, and expression across captures.

---

### 2. Preprocess

```bash
python src/preprocessing/preprocess.py
```

This script:
1. Detects and crops faces from every raw image (OpenCV Haar cascade).
2. Resizes crops to 128 × 128 pixels.
3. Splits into train (70 %) / val (15 %) / test (15 %).
4. Generates 4× augmented copies for each training image (flips, rotation, brightness/contrast jitter).

---

### 3. Train

```bash
python src/model/train.py --epochs 40 --batch-size 32
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 40 | Maximum training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--img-size` | 128 | Input image resolution |
| `--lr` | 0.001 | Initial learning rate |

Outputs saved to `models/`:
- `face_recognition_model.keras` – best checkpoint (by val accuracy)
- `label_map.json` – `{index: person_name}` mapping
- `training_history.png` – loss & accuracy curves

---

### 4. Evaluate

```bash
python src/model/evaluate.py --threshold 0.60
```

Prints a per-class classification report and saves:
- `models/confusion_matrix.png`
- `models/threshold_curve.png` – accuracy vs. confidence threshold

---

### 5. Run the Real-Time Application

> **Skip steps 1–4 if a trained model is already provided.** You only need `models/face_recognition_model.keras` and `models/label_map.json` to run the app.

```bash
python src/app/app.py --threshold 0.60 --camera 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold` | 0.60 | Min confidence to label a face as known |
| `--camera` | 0 | Webcam device index |

**Live controls:**
| Key | Action |
|-----|--------|
| `t` | Raise confidence threshold by 0.05 |
| `g` | Lower confidence threshold by 0.05 |
| `q` | Quit |

The UI overlays:
- Green bounding box + predicted name + confidence for **known** people.
- Red bounding box + "Unknown" for low-confidence detections.

---

## Model Architecture

Custom CNN with 4 convolutional blocks:

```
Input (128×128×3)
→ [Conv2D(32) → BN → ReLU] × 2 → MaxPool          → 64×64×32
→ [Conv2D(64) → BN → ReLU] × 2 → MaxPool          → 32×32×64
→ [Conv2D(128) → BN → ReLU] × 2 → MaxPool         → 16×16×128
→ [Conv2D(256) → BN → ReLU] × 2
→ GlobalAveragePooling
→ Dense(128, ReLU) → Dropout(0.4)
→ Dense(N_classes, Softmax)
```

Trained with:
- Optimiser: Adam (lr = 1e-3, with ReduceLROnPlateau)
- Loss: categorical cross-entropy
- Callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## Unknown Detection

A face is labelled **Unknown** when the model's maximum softmax output is below the confidence threshold.
The threshold can be tuned using `evaluate.py --threshold <value>` which plots the accuracy/coverage trade-off curve.

---

## Limitations & Future Work

- Haar cascade detection is fast but less accurate than deep detectors (MTCNN, RetinaFace) under challenging lighting.
- The dataset is small; transfer learning from a pre-trained face model (e.g., FaceNet, ArcFace) would improve accuracy.
- The Unknown class is not explicitly trained — future work could add an open-set recognition head or use distance-based thresholding on embeddings.
- Currently desktop-only; a small Flask/FastAPI web UI would make it accessible across devices.
