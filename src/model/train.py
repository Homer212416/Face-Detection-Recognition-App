"""
train.py
--------
Train the face-recognition CNN on the preprocessed splits.

Usage:
    python src/model/train.py [--epochs 30] [--batch-size 32] [--img-size 128]

Outputs:
    models/face_recognition_model.keras  – best checkpoint
    models/label_map.json                – {index: person_name}
    models/training_history.png          – loss/accuracy curves
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_model import build_model

SPLITS_DIR = os.path.join("data", "splits")
MODELS_DIR = "models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def make_generators(img_size: int, batch_size: int):
    """Create train / val ImageDataGenerators (augmentation already done offline)."""
    # Minimal online augmentation on top of offline augmented data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        brightness_range=[0.85, 1.15],
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def save_label_map(class_indices: dict, path: str):
    """Save {index: class_name} JSON for inference time."""
    idx_to_class = {v: k for k, v in class_indices.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[INFO] Label map saved → {path}")


def plot_history(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {out_path}")


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[INFO] Loading datasets …")
    train_gen, val_gen = make_generators(args.img_size, args.batch_size)

    n_classes = train_gen.num_classes
    print(f"[INFO] Classes ({n_classes}): {list(train_gen.class_indices.keys())}")

    # Save label map immediately so inference works even if training is interrupted
    label_map_path = os.path.join(MODELS_DIR, "label_map.json")
    save_label_map(train_gen.class_indices, label_map_path)

    print("[INFO] Building model …")
    # model = build_model(n_classes=n_classes, img_size=args.img_size)

    # support change lr
    # python src/model/train.py --lr 0.001
    # python src/model/train.py --lr 0.0005
    # python src/model/train.py --lr 0.0001

    model = build_model(n_classes=n_classes, img_size=args.img_size, lr=args.lr)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(MODELS_DIR, "face_recognition_model.keras")
    # When val_loss stops decreasing → automatically reduce lr
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("[INFO] Training …")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    plot_history(history, os.path.join(MODELS_DIR, "training_history.png"))

    print(f"\n[DONE] Best model saved → {checkpoint_path}")
    best_val_acc = max(history.history["val_accuracy"])
    print(f"[DONE] Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
