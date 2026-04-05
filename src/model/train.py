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
# Uses the ‘Agg’ backend, a non-interactive graphics engine.
# Designed for generating and saving images on Linux servers without a GUI (monitor), preventing errors.
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_model import build_model

SPLITS_DIR = os.path.join("data", "splits")
MODELS_DIR = "models"


def parse_args():
    # Parse command-line arguments to allow hyperparameters to be adjusted without modifying the code
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def make_generators(img_size: int, batch_size: int):
    """Create train / val ImageDataGenerators (augmentation already done offline)."""
    # === Training Set Generator ===
    # Incorporates extensive online data augmentation, 
    # randomly transforming images each time they are read to prevent the model from memorizing patterns (overfitting)

    # Tried
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0 / 255,
    #     horizontal_flip=True,
    #     rotation_range=10,
    #     brightness_range=[0.85, 1.15],
    # )
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0 / 255,
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     zoom_range=0.1,
    #     horizontal_flip=True
    # )
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0 / 255,
    #     horizontal_flip=True,
    # )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,      # Core operation: Normalize pixel values from 0–255 to the range 0–1
        rotation_range=15,      # Rotate image randomly by ±15 degrees
        width_shift_range=0.1,  # Random horizontal shift of 10% of the total width
        height_shift_range=0.1, # Random vertical offset of 10% of the total height
        zoom_range=0.1,         # Scale image randomly by 10%
        horizontal_flip=True,   # 50% probability of random horizontal flipping (since faces are usually symmetrical, flipping is very effective)
        brightness_range=[0.85, 1.15]   # Randomly adjust brightness to simulate different lighting conditions
    )

    # === Validation Set Generator ===
    # Data augmentation must never be applied to the validation set! 
    # Only normalize the data to ensure the objectivity and accuracy of the evaluation metrics.
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Scan a specified directory and automatically generate data streams in batches
    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "train"),
        target_size=(img_size, img_size),   # Automatically resize all images to the same size
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,                       # Shuffle the training set
        seed=42,                            # Fix the random seed to ensure the reproducibility of experimental results
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


# Save the label mapping to a JSON file.
# During model training, the format used is {class_name: index}, such as {“Alice”: 0, “Bob”: 1}.
# However, during prediction (inference), the model outputs 0, 
# and we need to look up the table to convert it back to “Alice”; therefore, the dictionary key-value pairs are reversed here.
def save_label_map(class_indices: dict, path: str):
    """Save {index: class_name} JSON for inference time."""
    idx_to_class = {v: k for k, v in class_indices.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[INFO] Label map saved → {path}")

# Plot the changes in accuracy and loss during training on a dual-line graph and save it
def plot_history(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plotting an Accuracy Chart
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Plotting the loss values
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

    # Dynamically retrieve the total number of categories
    n_classes = train_gen.num_classes
    print(f"[INFO] Classes ({n_classes}): {list(train_gen.class_indices.keys())}")


    # Save label map immediately so inference works even if training is interrupted
    label_map_path = os.path.join(MODELS_DIR, "label_map.json")
    save_label_map(train_gen.class_indices, label_map_path)

    print("[INFO] Building model …")

    model = build_model(n_classes=n_classes, img_size=args.img_size, lr=args.lr)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(MODELS_DIR, "face_recognition_model.keras")
    # When val_loss stops decreasing → automatically reduce lr
    callbacks = [
        # 1. Model checkpoint: Check the validation set accuracy at the end of each round, 
        # and save only the model weights corresponding to the best historical performance.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # 2. Learning rate decay: If the validation set loss does not decrease for 5 consecutive epochs, halve the learning rate (factor=0.5)
        # This helps the model overcome the “plateau” in the later stages of training and converge more precisely to the global optimum
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # 3. Early stopping mechanism: If the accuracy on the validation set shows no improvement for 10 consecutive iterations, 
        # training is terminated immediately to prevent overfitting and save time.
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
    # After training, plot the loss and accuracy curves
    plot_history(history, os.path.join(MODELS_DIR, "training_history.png"))

    print(f"\n[DONE] Best model saved → {checkpoint_path}")
    best_val_acc = max(history.history["val_accuracy"])
    print(f"[DONE] Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
