"""
Transfer Learning Model Builders
Supports: VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
Multiclass Fish Image Classification Project
"""

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    InceptionV3,
    EfficientNetB0,
)


# ─── Backbone Registry ────────────────────────────────────────────────────────
_BACKBONE_MAP = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0,
}

# InceptionV3 requires at least 75×75; EfficientNetB0 works best at 224.
# We use 224×224 for all (safe for every backbone listed above).
_MIN_SIZE = {
    "VGG16": (224, 224),
    "ResNet50": (224, 224),
    "MobileNetV2": (224, 224),
    "InceptionV3": (224, 224),
    "EfficientNetB0": (224, 224),
}


def build_transfer_model(
    backbone_name: str,
    num_classes: int,
    img_size: tuple = (224, 224),
    fine_tune_at: int = None,
):
    """
    Construct a transfer-learning model with a frozen pre-trained backbone
    and a custom classification head.  Optionally un-freeze layers for
    fine-tuning.

    Parameters
    ----------
    backbone_name : One of VGG16 | ResNet50 | MobileNetV2 | InceptionV3 | EfficientNetB0
    num_classes   : Output classes.
    img_size      : (height, width).
    fine_tune_at  : If provided, un-freeze all layers from this index onwards
                    in the backbone (used in Phase-2 fine-tuning).

    Returns
    -------
    Keras Model (uncompiled).
    """
    if backbone_name not in _BACKBONE_MAP:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choose from {list(_BACKBONE_MAP.keys())}")

    input_shape = (*img_size, 3)
    BackboneCls = _BACKBONE_MAP[backbone_name]

    # ── Load pre-trained backbone (ImageNet weights, no top) ─────────────────
    backbone = BackboneCls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False  # freeze initially

    # ── Optional partial un-freezing for fine-tune phase ─────────────────────
    if fine_tune_at is not None:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False

    # ── Custom classification head ────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=backbone_name)
    return model


def get_all_transfer_models(num_classes: int, img_size: tuple = (224, 224)) -> dict:
    """
    Convenience factory: returns a dict of all five transfer-learning models.

    Returns
    -------
    {backbone_name: Keras Model}
    """
    return {
        name: build_transfer_model(name, num_classes, img_size)
        for name in _BACKBONE_MAP
    }
