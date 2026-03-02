"""
evaluate.py
-----------
Evaluate the trained CNN on the held-out test split.

Produces:
  • Classification report (precision / recall / F1 per class)
  • Confusion matrix (saved as PNG)
  • Top-1 accuracy & average confidence on correct / incorrect predictions
  • Threshold analysis: accuracy vs. confidence threshold curve

Usage:
    python src/model/evaluate.py [--threshold 0.6]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODELS_DIR = "models"
SPLITS_DIR = os.path.join("data", "splits")
IMG_SIZE = 128
BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold below which a face is labelled Unknown",
    )
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "face_recognition_model.keras"))
    p.add_argument("--label-map", default=os.path.join(MODELS_DIR, "label_map.json"))
    return p.parse_args()


def load_label_map(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Keys are stored as strings in JSON; convert to int
    return {int(k): v for k, v in raw.items()}


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {out_path}")


def threshold_curve(y_true_idx, y_pred_probs, out_path):
    """Plot accuracy vs confidence threshold (shows how many are labelled Unknown)."""
    thresholds = np.linspace(0.0, 0.99, 100)
    accuracies = []
    coverages = []  # fraction of images above threshold

    for t in thresholds:
        max_probs = y_pred_probs.max(axis=1)
        mask = max_probs >= t
        if mask.sum() == 0:
            accuracies.append(0.0)
            coverages.append(0.0)
            continue
        preds_above = y_pred_probs[mask].argmax(axis=1)
        true_above = np.array(y_true_idx)[mask]
        accuracies.append((preds_above == true_above).mean())
        coverages.append(mask.mean())

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(thresholds, accuracies, color="steelblue", label="Accuracy (above thresh)")
    ax2.plot(thresholds, coverages, color="orange", linestyle="--", label="Coverage")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy", color="steelblue")
    ax2.set_ylabel("Coverage (fraction recognised)", color="orange")
    ax1.set_title("Accuracy vs Confidence Threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Threshold curve saved → {out_path}")


def main():
    args = parse_args()

    print("[INFO] Loading model …")
    model = keras.models.load_model(args.model)

    label_map = load_label_map(args.label_map)
    class_names = [label_map[i] for i in range(len(label_map))]

    print("[INFO] Loading test set …")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print("[INFO] Running inference …")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_true_idx = test_gen.classes

    # ── Metrics without threshold (raw top-1) ────────────────────────────────
    y_pred_idx = y_pred_probs.argmax(axis=1)
    print("\n── Classification Report (no threshold) ──")
    print(classification_report(y_true_idx, y_pred_idx, target_names=class_names))

    # ── Metrics with threshold ────────────────────────────────────────────────
    max_probs = y_pred_probs.max(axis=1)
    y_pred_thresh = np.where(max_probs >= args.threshold, y_pred_idx, -1)

    known_mask = y_pred_thresh != -1
    unknown_count = (~known_mask).sum()
    print(f"\n── Threshold = {args.threshold} ──")
    print(f"  Labelled Unknown : {unknown_count}/{len(y_true_idx)} ({100*unknown_count/len(y_true_idx):.1f}%)")
    if known_mask.sum() > 0:
        acc_known = (y_pred_thresh[known_mask] == y_true_idx[known_mask]).mean()
        print(f"  Accuracy (known) : {acc_known:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plot_confusion_matrix(cm, class_names, os.path.join(MODELS_DIR, "confusion_matrix.png"))
    threshold_curve(y_true_idx, y_pred_probs, os.path.join(MODELS_DIR, "threshold_curve.png"))

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
