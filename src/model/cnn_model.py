from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Two convolutions for feature extraction --> one pooling operation for dimensionality reduction --> one Dropout layer to prevent overfitting.
def conv_block(x, filters, block_name, dropout_rate=0.2):
    x = layers.Conv2D(
        filters,
        3,  # 3x3 Convolution Kernel
        padding="same", # Preserve edge information; keep the size of the feature map unchanged after convolution
        use_bias=False, # Since BatchNorm follows immediately after, and BN already includes a bias, we set this to False to save on parameters
        kernel_regularizer=regularizers.l2(1e-4),   # L2 regularization, to prevent overfitting caused by excessive weights
        name=f"{block_name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)  # Batch Normalization: Faster Convergence, Stable Training
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)    # ReLU Activation Function: Nonlinearity

    # === Second Convolution Layer ===
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

    x = layers.MaxPooling2D(2, name=f"{block_name}_pool")(x)    # 2*2 Max Pooling: Reduce spatial dimensions by half, while keeping the number of channels unchanged. This helps to reduce computational load and extract more abstract features.
    x = layers.Dropout(dropout_rate, name=f"{block_name}_drop")(x)
    return x


    # Build a multi-class face recognition CNN model
    # Parameters:
    #     n_classes: Total number of classes in the classification task
    #     img_size: Dimensions of the input image (default 128x128)
    #     dropout_rate: Final dropout rate before the fully connected layers (default 0.4)
    #     lr: Initial learning rate (default 3e-4, a common conservative initial value for the Adam optimizer)

def build_model(
    n_classes: int,
    img_size: int = 128,
    dropout_rate: float = 0.4,
    lr: float = 3e-4,
) -> keras.Model:
    # safe check need atleast 2 people
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2 for multi-class classification.")

    # 1. Define the input layer (default image size 128x128, 3 color channels: RGB)
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # 2. Backbone: Gradually extract features, reduce the dimensionality of the feature space, and increase the number of channels
    # As the network becomes deeper, features become increasingly abstract, and the dropout rate increases accordingly
    # 128 -> 64
    x = conv_block(inputs, 32, "block1", dropout_rate=0.15)

    # 64 -> 32
    x = conv_block(x, 64, "block2", dropout_rate=0.20)

    # 32 -> 16
    x = conv_block(x, 128, "block3", dropout_rate=0.25)

    # 16 -> 8
    x = conv_block(x, 256, "block4", dropout_rate=0.30)

    # 3. Refinement Block
    # At the 8x8 scale, no further dimensionality reduction is performed; 
    # instead, higher-level features are extracted using additional convolutional layers
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

    # 4. Classification Head
    # Replaces the traditional Flatten operation with Global Average Pooling (GAP), 
    # significantly reducing the number of parameters by compressing an 8x8x256 tensor into a one-dimensional vector of length 256
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # Fully Connected Hidden Layer (MLP)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc1",
    )(x)
    # Final Dropout Protection
    x = layers.Dropout(dropout_rate, name="final_dropout")(x)

    # Output layer: The number of neurons equals the number of classes; softmax is used to convert the output into a probability distribution
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="predictions",
    )(x)

    # Wrap the input and output into a model instance
    model = keras.Model(inputs, outputs, name="FaceRecognitionCNN_v2")

    # 5. Compile the model
    model.compile(
        # Loss function: Note that the labels must be in one-hot encoding format.
        # (If the labels are integers, such as 0, 1, 2, please use “sparse_categorical_crossentropy” instead.)
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model(n_classes=5)
    m.summary()


    #             Input Image
    #             (128 x 128 x 3)
    #                    │
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │             Block 1                 │
    # │  - Conv2D (32) + BN + ReLU          │
    # │  - Conv2D (32) + BN + ReLU          │
    # │  - MaxPooling2D (2x2)               │
    # │  - Dropout (0.15)                   │
    # └─────────────────────────────────────┘
    #                    │ Output: (64, 64, 32)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │             Block 2                 │
    # │  - Conv2D (64) + BN + ReLU          │
    # │  - Conv2D (64) + BN + ReLU          │
    # │  - MaxPooling2D (2x2)               │
    # │  - Dropout (0.20)                   │
    # └─────────────────────────────────────┘
    #                    │ Output: (32, 32, 64)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │             Block 3                 │
    # │  - Conv2D (128) + BN + ReLU         │
    # │  - Conv2D (128) + BN + ReLU         │
    # │  - MaxPooling2D (2x2)               │
    # │  - Dropout (0.25)                   │
    # └─────────────────────────────────────┘
    #                    │ Output: (16, 16, 128)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │             Block 4                 │
    # │  - Conv2D (256) + BN + ReLU         │
    # │  - Conv2D (256) + BN + ReLU         │
    # │  - MaxPooling2D (2x2)               │
    # │  - Dropout (0.30)                   │
    # └─────────────────────────────────────┘
    #                    │ Output: (8, 8, 256)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │         Refinement Block            │
    # │  - Conv2D (256) + BN + ReLU         │
    # │  - Conv2D (256) + BN + ReLU         │
    # │  (no pooling)             │
    # └─────────────────────────────────────┘
    #                    │ Output: (8, 8, 256)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │       GlobalAveragePooling2D        │
    # │                                     │ <-- Flatten an 8x8 spatial dimension to an average
    # └─────────────────────────────────────┘
    #                    │ Output: (256,)  <-- Convert to a one-dimensional vector
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │              Dense (128)            │
    # │  - Activation: ReLU                 │
    # │  - L2 Regularization (1e-4)         │
    # └─────────────────────────────────────┘
    #                    │ Output: (128,)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │             Dropout (0.4)           │
    # └─────────────────────────────────────┘
    #                    │ Output: (128,)
    #                    ▼
    # ┌─────────────────────────────────────┐
    # │           Dense (n_classes)         │
    # │  - Activation: Softmax              │
    # └─────────────────────────────────────┘
    #                    │ Output: (n_classes,) 
    #                    ▼
    #           Class Probabilities
   
    #             [ Input Tensor ]
    #            Shape: (128, 128, 3) 
    #      (RGB Image with 3 color channels)
    #                        │
    #                        ▼
    # ┌───────────────────────────────────────────────┐
    # │                 1. Conv2D(32)                 │
    # ├───────────────────────────────────────────────┤
    # │ • Kernel Size: 3x3                            │
    # │ • Number of Filters: 32                       │
    # │ • Padding: "same"                             │
    # │ • Bias: False (Intentionally omitted)         │
    # │                                               │
    # │ Math Operation: Y = W * X                     │
    # │ Intuition: Using 32 different "magnifying     │
    # │            glasses" to find 32 distinct local │
    # │            features (like edges or textures). │
    # └───────────────────────────────────────────────┘
    #                        │ 
    #                        │ Spatial dimensions unchanged (due to padding="same")
    #                        │ Number of channels jumps: 3 -> 32
    #                        ▼
    #                 [ Hidden Tensor ]
    #            Shape: (128, 128, 32) 
    #                        │
    #                        ▼
    # ┌───────────────────────────────────────────────┐
    # │            2. BatchNormalization              │
    # ├───────────────────────────────────────────────┤
    # │ • Statistics: Calculates the mean (μ) and     │
    # │               variance (σ²) across the batch  │
    # │               for each channel.               │
    # │ • Learnable Params: Scale (γ) and Shift (β).  │
    # │                                               │
    # │ Math Operation: Y = γ * (X - μ) / √(σ² + ε)+β │
    # │ Intuition: Forces the feature data back to a  │
    # │            standard distribution, preventing  │
    # │            "internal covariate shift". This   │
    # │            speeds up training and mitigates   │
    # │            vanishing gradients.               │
    # └───────────────────────────────────────────────┘
    #                        │ 
    #                        │ Dimensions & channels unchanged
    #                        │ Data distribution is normalized and shifted
    #                        ▼
    #                 [ Hidden Tensor ]
    #            Shape: (128, 128, 32) 
    #                        │
    #                        ▼
    # ┌───────────────────────────────────────────────┐
    # │              3. ReLU Activation               │
    # ├───────────────────────────────────────────────┤
    # │                                               │
    # │ Math Operation: Y = max(0, X)                 │
    # │ Intuition: Introduces non-linearity. Zeros    │
    # │            out all negative features and      │
    # │            keeps positive ones. Without this, │
    # │            stacking multiple conv layers      │
    # │            would just mathematically collapse │
    # │            into a single linear layer.        │
    # └───────────────────────────────────────────────┘
    #                        │ 
    #                        │ Dimensions & channels unchanged
    #                        │ Negative values become 0
    #                        ▼
    #              [ Output Tensor ]
    #            Shape: (128, 128, 32)
    #   (Passed to the next Conv2D or MaxPooling layer)


# old model
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
