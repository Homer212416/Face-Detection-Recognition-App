# Project 1: Face Detection & Recognition App

**Applied Artificial Intelligence — COMP 6721 (4 credits)**
**Dept. of Computer Science and Software Engineering · Concordia University · Syllabus – Winter 2026**

---

## 1. What You Will Build

For this project, your team will create a working application that:

- Uses a **camera** to detect faces in real time.
- Recognizes **specific people** that you trained the model on.
- Labels anyone else as **"Unknown"** (or ignores them) based on a confidence threshold.

You'll go through the full AI pipeline:

- collecting data
- preprocessing
- training a CNN
- integrating it into a working application.

---

## 2. Build Your Own Dataset

You will create a dataset of **minimum 5 people**:

- Your team members must be included.
- The remaining people can be friends, classmates, or colleagues.
- Everyone must give consent to be photographed.

### Image Requirements

- **Minimum 30 images per person.**
- Capture a variety of:
  - Poses
  - Lighting conditions
  - Facial expressions
  - Angles (frontal + non-frontal)
  - Styles (with or without makeup)

### Preprocessing

You will:

- Detect and **crop** faces from the raw images.
- Resize and normalize them.
- Apply **data augmentation** (rotation, brightness changes, flips, etc.).
- Split into **train / validation / test** sets.

---

## 3. Build the Recognition System

### Face Detection

Choose any reasonable method, such as:

- OpenCV (Recommended)
- MTCNN
- YOLO-based detectors
- Other justified detectors

Your detector must:

- Work in real time.
- Crop faces and send them to your recognition model.

### Face Recognition (CNN)

Train a CNN model that can classify the people in your dataset.

Your model must output:

- A predicted identity
- A confidence score

You can use Keras or PyTorch to build and to train your CNN model.

---

## 4. Build the Application

Your final app must **work in real time**:

- Use a webcam feed.
- Detect faces in each frame.
- Recognize people **only if the confidence score is above your chosen threshold**.

**And handle Unknown People:** if the confidence is too low:

- Display **"Unknown"**, or
- Ignore the detection

**It should include a Simple UI:** this can be a desktop or a small web interface. It must show:

- Bounding boxes around faces
- Predicted name or "Unknown"
- Confidence score

---

## 5. What You Must Submit

### 5.1 Code Repository on GitHub

- Clean, organized code including:
  - Data preprocessing
  - Model training
  - Model evaluation
  - Real-time application
- **README.MD** with setup and run instructions
- Required dependencies (either of the following):
  - PIP env: `requirements.txt`
  - Conda env: `environment.yml`

### 5.2 Final Report (6–10 pages)

Include:

- How you collected your dataset
- Preprocessing steps
- Model architecture + training details
- Evaluation results
- How you chose your threshold to identify a person as Unknown
- How your application works
- Limitations + ideas for improvement
