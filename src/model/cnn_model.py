"""
cnn_model.py
------------
Custom CNN architecture for face recognition.

The model takes (IMG_SIZE, IMG_SIZE, 3) RGB images and outputs a softmax
probability distribution over N_CLASSES identities.

Architecture overview
---------------------
  Input → [Conv→BN→ReLU→Pool] x3 → [Conv→BN→ReLU] x2 → GlobalAvgPool
        → Dense(256, ReLU) → Dropout(0.4) → Dense(N_CLASSES, Softmax)

This is intentionally lightweight so it can train from scratch on a small
dataset (≥ 30 images / person) without GPU requirements, while still being
deep enough to learn discriminative facial features.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_classes: int, img_size: int = 128, dropout_rate: float = 0.4) -> keras.Model:
    """
    Build and return a compiled Keras face-recognition CNN.

    Parameters
    ----------
    n_classes : int
        Number of identities in the dataset.
    img_size : int
        Spatial dimension of square input images (pixels).
    dropout_rate : float
        Dropout probability before the final Dense layer.

    Returns
    -------
    keras.Model
        Compiled model ready for .fit().
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)   # 64×64

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)   # 32×32

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)   # 16×16

    # ── Block 4 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="FaceRecognitionCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_model(n_classes=5)
    m.summary()
