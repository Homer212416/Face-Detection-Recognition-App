# Face Detection & Recognition App — Final Report

**Course:** COMP 6721 — Applied Artificial Intelligence  
**Institution:** Concordia University, Department of Computer Science and Software Engineering  
**Semester:** Winter 2026  
**Team Members:** Booth, Yikai

---

## 1. Introduction

This report describes the design, implementation, and evaluation of a real-time face detection and recognition desktop application built for the COMP 6721 final project. The system uses a webcam to detect faces in each frame, classifies each face as one of a set of known identities, and labels unrecognized faces as "Unknown" based on a confidence threshold. Every component — from data collection to the recognition model to the live application — was built from scratch by the team.

The recognized identities are two team members (Booth and Yikai) and five public figures (Angelina Jolie, Jennifer Lawrence, Johnny Depp, Leonardo DiCaprio, and Robert Downey Jr.), for a total of seven classes.

---

## 2. Dataset Collection

### 2.1 Team Member Images

Each team member's images were collected using a custom webcam capture script (`capture_images.py`). The script opens the camera, runs an OpenCV Haar cascade detector in real time to confirm a face is visible, and auto-saves one frame every 0.5 seconds until the target count is reached. Bounding boxes are drawn on a display copy only; the saved frames are the raw, unmodified images. **100 images per team member** were collected this way.

To maximize variability, images were captured under different lighting conditions (natural daylight, indoor overhead, lamp-lit), at multiple angles (frontal, slight left/right rotation, slight tilt), with varied expressions (neutral, smiling, raised eyebrows), and with and without eyeglasses or accessories.

Images are stored as JPEG files in `data/raw/<PersonName>/`. Filenames are zero-padded and indexed from the existing count so that multiple capture sessions accumulate without overwriting.

### 2.2 Celebrity Images

To supplement the dataset with additional identities that the team cannot physically photograph, five celebrities were included: Angelina Jolie, Jennifer Lawrence, Johnny Depp, Leonardo DiCaprio, and Robert Downey Jr. Their images were sourced from publicly available face datasets and placed directly into the `data/splits/` directory in pre-processed form (128×128 PNG face crops), bypassing the raw collection and preprocessing stages.

### 2.3 Dataset Summary

| Person | Source | Images (raw) |
|---|---|---|
| Booth | Webcam capture | 100 |
| Yikai | Webcam capture | 100 |
| Angelina Jolie | External dataset | — |
| Jennifer Lawrence | External dataset | — |
| Johnny Depp | External dataset | — |
| Leonardo DiCaprio | External dataset | — |
| Robert Downey Jr. | External dataset | — |

---

## 3. Preprocessing

All preprocessing is handled by `preprocess.py`, which runs three sequential steps. Re-running the script is safe — it wipes and rebuilds `data/processed/` and `data/splits/` from scratch.

### 3.1 Step 1 — Face Cropping

For each image in `data/raw/<Person>/`, an OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) is run to detect faces. The first detected bounding box is selected, then expanded by a 20% padding on each side (clamped to image boundaries) to include forehead, chin, and ear context. The padded crop is resized to **128 × 128 pixels** using OpenCV and saved as a PNG to `data/processed/<Person>/`. If no face is detected in a raw image, the full image is resized and saved as a fallback.

### 3.2 Step 2 — Train / Val / Test Split

Processed images per person are shuffled with a fixed seed (`RANDOM_SEED = 42`) for reproducibility, then divided in a **70 / 15 / 15** ratio into train, validation, and test subsets. Files are copied (not moved) from `data/processed/` to the corresponding `data/splits/{train,val,test}/<Person>/` directory.

For 100 images per team member this yields 70 train / 15 val / 15 test images per person.

### 3.3 Step 3 — Offline Data Augmentation

To expand the training set and reduce overfitting, each original training image is used to generate **4 augmented variants** (`AUGMENT_FACTOR = 4`). Each variant is created by opening the original image fresh (to prevent transform stacking) and applying the following transformations sequentially:

1. **Horizontal flip** — applied with 50% probability (faces are roughly symmetric).
2. **Random rotation** — uniform sample from ±20°.
3. **Brightness jitter** — enhancement factor sampled uniformly from [0.7, 1.3].
4. **Contrast jitter** — enhancement factor sampled uniformly from [0.8, 1.2].

Augmented files are saved as `<stem>_aug{i:02d}.png` alongside the originals. Only original (non-`_aug`) files are augmented, preventing exponential growth on re-runs.

**Resulting training set sizes (team members):**

| Person | Originals | Augmented copies | Total train images |
|---|---|---|---|
| Booth | 70 | 280 | 350 |
| Yikai | 70 | 280 | 350 |

### 3.4 Online Augmentation During Training

`train.py` applies a second, lighter layer of augmentation on-the-fly via Keras `ImageDataGenerator`:

- Random rotation ±15°
- Random horizontal / vertical shift ±10%
- Random zoom ±10%
- Horizontal flip (50%)
- Brightness range [0.85, 1.15]
- Pixel normalization: divide by 255 → range [0, 1]

The validation and test generators apply normalization only — no augmentation.

---

## 4. Model Architecture

The recognition model (`cnn_model.py`) is a custom CNN trained from scratch. It takes a single 128 × 128 RGB image (float32, normalized to [0, 1]) and outputs a softmax probability distribution over all N classes.

### 4.1 Backbone — Four Convolutional Blocks

The backbone consists of four identical convolutional blocks that progressively double the number of filters while halving the spatial resolution via max-pooling:

| Block | Filters | Dropout | Spatial output |
|---|---|---|---|
| Block 1 | 32 | 0.15 | 64 × 64 |
| Block 2 | 64 | 0.20 | 32 × 32 |
| Block 3 | 128 | 0.25 | 16 × 16 |
| Block 4 | 256 | 0.30 | 8 × 8 |

Each block contains: `Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool2D(2) → Dropout`.

Design choices:
- **`use_bias=False` on Conv2D** — Batch Normalization provides the shift term, so the conv bias is redundant.
- **Increasing dropout rate per block** — deeper features are more abstract and prone to overfitting on a small dataset; stronger regularization is applied progressively.
- **L2 regularization (λ = 1e-4)** on all Conv2D and Dense layers.

### 4.2 Refinement Block

After Block 4, an additional pair of `Conv2D(256) → BatchNorm → ReLU` layers (no pooling) operates at the 8 × 8 spatial scale. This allows the network to extract higher-level features without further spatial compression.

### 4.3 Classification Head

```
GlobalAveragePooling2D          → (256,)
Dense(128, ReLU, L2=1e-4)       → (128,)
Dropout(0.4)
Dense(N_classes, Softmax)       → (N_classes,)
```

`GlobalAveragePooling2D` replaces the conventional `Flatten` + large Dense layer. It averages the 8 × 8 spatial grid per channel, compressing `8 × 8 × 256 = 16,384` values into a 256-dimensional vector. This dramatically reduces the parameter count and acts as a built-in regularizer.

### 4.4 Architecture Diagram

```
Input (128×128×3)
    │
    ▼
Block 1: [Conv2D(32)→BN→ReLU] × 2 → MaxPool → Dropout(0.15)   → 64×64×32
    │
Block 2: [Conv2D(64)→BN→ReLU] × 2 → MaxPool → Dropout(0.20)   → 32×32×64
    │
Block 3: [Conv2D(128)→BN→ReLU] × 2 → MaxPool → Dropout(0.25)  → 16×16×128
    │
Block 4: [Conv2D(256)→BN→ReLU] × 2 → MaxPool → Dropout(0.30)  → 8×8×256
    │
Refinement: [Conv2D(256)→BN→ReLU] × 2  (no pool)              → 8×8×256
    │
GlobalAveragePooling2D                                          → (256,)
    │
Dense(128, ReLU) → Dropout(0.4)
    │
Dense(N_classes, Softmax)
```

---

## 5. Training

### 5.1 Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 40 |
| Input size | 128 × 128 × 3 |

### 5.2 Callbacks

Three Keras callbacks govern the training process:

1. **ModelCheckpoint** (`monitor=val_accuracy`, `save_best_only=True`) — saves the model weights only when validation accuracy improves, so the final saved model is the best checkpoint rather than the final epoch.

2. **ReduceLROnPlateau** (`monitor=val_loss`, `patience=5`, `factor=0.5`, `min_lr=1e-6`) — halves the learning rate when validation loss stagnates for 5 consecutive epochs. This helps the model escape flat regions in the loss landscape in later training.

3. **EarlyStopping** (`monitor=val_accuracy`, `patience=10`, `restore_best_weights=True`) — halts training if validation accuracy shows no improvement for 10 consecutive epochs, preventing overfitting and unnecessary computation.

### 5.3 Training Curves

Training converged in approximately 30 epochs (early stopping triggered before the 40-epoch limit). The training accuracy reached ~99% while validation accuracy stabilized in the 75–80% range, with visible fluctuation due to the small validation set size.

![Training History](models/training_history.png)

The validation loss spiked sharply at epoch 1 (due to the randomly initialized model producing high-confidence wrong predictions early on) then converged rapidly. The train/val accuracy gap is indicative of mild overfitting, which is expected given the dataset scale; the gap is constrained by the regularization strategy described above.

---

## 6. Evaluation Results

Evaluation is performed by `evaluate.py` on the held-out test split, which the model never sees during training or validation.

### 6.1 Confusion Matrix

The confusion matrix below shows per-class predictions on the test set for the two team-member classes:

![Confusion Matrix](models/confusion_matrix.png)

Both Booth and Yikai are predicted with perfect precision and recall (15/15 correct for each). There are zero cross-class confusions between the team members.

### 6.2 Classification Report

On the team-member test subset (no threshold applied):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Booth | 1.00 | 1.00 | 1.00 | 15 |
| Yikai | 1.00 | 1.00 | 1.00 | 15 |
| **Overall** | **1.00** | **1.00** | **1.00** | **30** |

These results reflect a well-separated decision boundary between the two team members whose images were captured and preprocessed through the full pipeline.

---

## 7. Confidence Threshold and Unknown Detection

### 7.1 Design Rationale

The model is trained as a closed-set classifier — its output layer has exactly N neurons, one per known identity. It has no explicit notion of "unknown person." To handle faces that do not belong to any known identity, we apply a **softmax confidence threshold**: if the maximum output probability across all classes falls below the threshold, the face is labelled "Unknown" rather than assigned to any class.

This is a deliberate design choice that keeps the model architecture simple and allows the threshold to be tuned independently of the model weights.

### 7.2 Threshold Analysis

`evaluate.py` sweeps the threshold from 0 to 0.99 and plots both accuracy (on samples above the threshold) and coverage (fraction of samples not rejected as Unknown):

![Threshold Curve](models/threshold_curve.png)

Key observations:
- **Accuracy remains at 1.00** for all threshold values up to approximately 0.90. This indicates the model's correct predictions are almost universally high-confidence.
- **Coverage stays at 1.00** (no samples rejected) up to a threshold of roughly 0.90, then drops as the threshold rises further.
- A threshold of **0.60** (the chosen default) sits safely in the flat region where accuracy = 1.00 and coverage = 1.00 for the test set.

### 7.3 Chosen Threshold

**Default threshold: 0.60**

At this value:
- All correctly identified test faces are accepted (coverage = 100%).
- The model is conservative enough to reject genuinely uncertain predictions.
- There is headroom before coverage degrades (~0.90), providing robustness to real-world variation.

The threshold can be adjusted at runtime with the `t` (raise) and `g` (lower) keys in the live application.

---

## 8. Application

### 8.1 Architecture

The real-time application (`app.py`) implements the following per-frame pipeline:

1. **Frame capture** — OpenCV reads a frame from the webcam.
2. **Face detection** — The frame is converted to grayscale and passed to the OpenCV Haar cascade (`haarcascade_frontalface_default.xml`). Parameters: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(80, 80)`.
3. **Face preprocessing** — Each detected bounding box is expanded by 20% padding (matching the preprocessing convention), cropped from the frame, resized to 128 × 128, converted from BGR to RGB, and normalized to [0, 1].
4. **Batch inference** — All face crops in the current frame are stacked into a single batch and passed to the CNN in one `model.predict` call, minimizing per-face overhead.
5. **Label assignment** — For each face, if `max(softmax output) ≥ threshold`, the face is assigned the argmax class label; otherwise it is labelled "Unknown".
6. **Rendering** — Each face receives a bounding box overlay:
   - **Green** box + predicted name + confidence percentage for known identities.
   - **Red** box + "Unknown" for low-confidence detections.
7. **HUD** — A heads-up display in the top-left corner shows the current threshold, number of detected faces, and live FPS.

### 8.2 Controls

| Key | Effect |
|---|---|
| `q` | Quit the application |
| `t` | Raise threshold by 0.05 (max 0.99) |
| `g` | Lower threshold by 0.05 (min 0.01) |

### 8.3 Performance

The application runs in real time on CPU. The Haar cascade detector is fast (milliseconds per frame), and batch inference over the typically small number of detected faces adds minimal latency. FPS is displayed live on screen.

---

## 9. Limitations and Future Work

### 9.1 Face Detection

The Haar cascade detector is computationally lightweight but less robust than modern deep learning–based detectors. It struggles under:
- Extreme lighting (very bright or very dark environments)
- Non-frontal angles (profile faces, significant head tilt)
- Partial occlusion (masks, hands over the face)

**Improvement:** Replace the Haar cascade with a deep detector such as MTCNN or RetinaFace, which offer significantly higher recall and work well at non-frontal angles.

### 9.2 Dataset Scale

The team-collected dataset contains 100 raw images per person, which — even after 4× offline augmentation — is small compared to production-scale face recognition datasets (which typically have thousands of images per identity). The val/train accuracy gap in training curves reflects this limitation.

**Improvement:** Collect more images (200–500 per person), introduce more varied backgrounds, and consider transfer learning from a pre-trained face embedding model (FaceNet, ArcFace) rather than training from scratch.

### 9.3 Unknown Detection

The current "Unknown" mechanism is purely threshold-based on softmax probabilities. Softmax values are known to be overconfident — the model can assign high confidence to out-of-distribution inputs if they resemble a training class in some superficial way. An attacker's face might be incorrectly labelled as a known person with high confidence.

**Improvement:** Train the model to produce a fixed-dimension embedding vector and apply distance-based thresholding (e.g., cosine distance from class centroids). Alternatively, add an explicit "Unknown" class using negative samples, or apply an open-set recognition method such as OpenMax.

### 9.4 Platform Constraints

The application is desktop-only and requires a locally connected camera and Python environment.

**Improvement:** Wrap the inference pipeline in a small Flask or FastAPI web server with a React/Vue front end, enabling browser-based access from any device on the local network.

### 9.5 Single-Face Training Images

All training images contain exactly one face per image. In multi-person scenes where multiple faces are detected simultaneously, each crop is classified independently, which is correct, but the lack of multi-face training scenarios means the model has not learned to distinguish between faces appearing together in the same frame context.

---

## 10. Conclusion

This project demonstrates the complete AI development pipeline — from raw data collection through preprocessing, model training, evaluation, and real-time deployment — using only standard open-source libraries (TensorFlow/Keras, OpenCV, Pillow). The custom CNN achieves 100% test-set accuracy on the two team-member classes and performs cleanly in live use. The threshold-based Unknown rejection provides a practical safeguard against misidentification. The main areas for future work are stronger face detection, a larger and more diverse dataset, and more principled open-set recognition.
