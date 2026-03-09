"""
Custom CNN Model – Built From Scratch
Multiclass Fish Image Classification Project
"""

from tensorflow.keras import layers, models, regularizers


def build_cnn(num_classes: int, img_size: tuple = (224, 224), input_channels: int = 3):
    """
    Build a custom Convolutional Neural Network from scratch.

    Architecture
    ────────────
    Block 1 : Conv(32)  → BN → ReLU → Conv(32)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2 : Conv(64)  → BN → ReLU → Conv(64)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3 : Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 4 : Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Dropout(0.25)
    Head    : GlobalAvgPool → Dense(512, ReLU) → BN → Dropout(0.5)
              → Dense(num_classes, Softmax)

    Parameters
    ----------
    num_classes    : Number of fish species to classify.
    img_size       : (height, width) of input images.
    input_channels : Colour channels (3 for RGB).

    Returns
    -------
    Compiled Keras Sequential model.
    """
    input_shape = (*img_size, input_channels)

    model = models.Sequential(name="CustomCNN")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2 ───────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3 ───────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 4 ───────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(256, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Classification Head ───────────────────────────────────────────────────
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
