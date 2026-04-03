# """
# cnn_model.py
# ------------
# Custom CNN architecture for face recognition.

# The model takes (IMG_SIZE, IMG_SIZE, 3) RGB images and outputs a softmax
# probability distribution over N_CLASSES identities.

# Architecture overview
# ---------------------
#   Input → [Conv→BN→ReLU→Pool] x3 → [Conv→BN→ReLU] x2 → GlobalAvgPool
#         → Dense(256, ReLU) → Dropout(0.4) → Dense(N_CLASSES, Softmax)

# This is intentionally lightweight so it can train from scratch on a small
# dataset (≥ 30 images / person) without GPU requirements, while still being
# deep enough to learn discriminative facial features.
# """

# from tensorflow import keras
# from tensorflow.keras import layers

# """Input
# → [Conv→BN→ReLU → Conv→BN→ReLU → MaxPool] * 3
# → [Conv→BN→ReLU → Conv→BN→ReLU]
# → GlobalAveragePooling2D
# → Dense(256, ReLU)
# → Dropout(0.4)
# → Dense(N_CLASSES, Softmax)"""

# # def build_model(n_classes: int, img_size: int = 128, dropout_rate: float = 0.4) -> keras.Model:
# def build_model(n_classes: int, img_size: int = 128, dropout_rate: float = 0.4, lr: float = 1e-3) -> keras.Model:
#     """
#     Build and return a compiled Keras face-recognition CNN.

#     Parameters
#     ----------
#     n_classes : int
#         Number of identities in the dataset.
#     img_size : int
#         Spatial dimension of square input images (pixels).
#     dropout_rate : float
#         Dropout probability before the final Dense layer.

#     Returns
#     -------
#     keras.Model
#         Compiled model ready for .fit().
#     """

#     if n_classes < 2:
#         raise ValueError("n_classes must be at least 2 for multi-class classification.")
    
#     inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

#     # ── Block 1 ──────────────────────────────────────────────────────────────
#     x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_1")(inputs)
#     x = layers.BatchNormalization(name="bn1_1")(x)
#     x = layers.Activation("relu")(x)    
#     # x = layers.ReLU(name="relu1_1")(x)
#     # x = layers.Activation("relu", name="relu1_1")(x)
#     x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_2")(x)
#     x = layers.BatchNormalization(name="bn1_2")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, name="pool1")(x)   # 64×64

#     # ── Block 2 ──────────────────────────────────────────────────────────────
#     x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_1")(x)
#     x = layers.BatchNormalization(name="bn2_1")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_2")(x)
#     x = layers.BatchNormalization(name="bn2_2")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, name="pool2")(x)   # 32×32

#     # ── Block 3 ──────────────────────────────────────────────────────────────
#     x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_1")(x)
#     x = layers.BatchNormalization(name="bn3_1")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_2")(x)
#     x = layers.BatchNormalization(name="bn3_2")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, name="pool3")(x)   # 16×16

#     # ── Block 4 ──────────────────────────────────────────────────────────────
#     x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_1")(x)
#     x = layers.BatchNormalization(name="bn4_1")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_2")(x)
#     x = layers.BatchNormalization(name="bn4_2")(x)
#     x = layers.Activation("relu")(x)

#     # ── Head ─────────────────────────────────────────────────────────────────
#     x = layers.GlobalAveragePooling2D(name="gap")(x)
#     x = layers.Dense(256, activation="relu", name="fc1")(x)
#     x = layers.Dropout(dropout_rate, name="dropout")(x)
#     outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

#     model = keras.Model(inputs, outputs, name="FaceRecognitionCNN")

#     # model.compile(
#     #     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     #     loss="categorical_crossentropy",
#     #     metrics=["accuracy"],
#     # )

#     model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=lr),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
#     )
#     return model


# if __name__ == "__main__":
#     # Quick sanity check
#     m = build_model(n_classes=5)
#     m.summary()

# try something new
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def conv_block(x, filters, block_name, dropout_rate=0.2):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{block_name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{block_name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu2")(x)

    x = layers.MaxPooling2D(2, name=f"{block_name}_pool")(x)
    x = layers.Dropout(dropout_rate, name=f"{block_name}_drop")(x)
    return x


def build_model(
    n_classes: int,
    img_size: int = 128,
    dropout_rate: float = 0.4,
    lr: float = 3e-4,
) -> keras.Model:
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2 for multi-class classification.")

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # 128 -> 64
    x = conv_block(inputs, 32, "block1", dropout_rate=0.15)

    # 64 -> 32
    x = conv_block(x, 64, "block2", dropout_rate=0.20)

    # 32 -> 16
    x = conv_block(x, 128, "block3", dropout_rate=0.25)

    # 16 -> 8
    x = conv_block(x, 256, "block4", dropout_rate=0.30)

    # Extra feature refinement at 8x8
    x = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name="refine_conv1",
    )(x)
    x = layers.BatchNormalization(name="refine_bn1")(x)
    x = layers.Activation("relu", name="refine_relu1")(x)

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name="refine_conv2",
    )(x)
    x = layers.BatchNormalization(name="refine_bn2")(x)
    x = layers.Activation("relu", name="refine_relu2")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc1",
    )(x)
    x = layers.Dropout(dropout_rate, name="final_dropout")(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="FaceRecognitionCNN_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model(n_classes=5)
    m.summary()
