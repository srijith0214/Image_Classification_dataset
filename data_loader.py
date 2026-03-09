"""
Data Loading and Preprocessing Utilities
Multiclass Fish Image Classification Project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def get_data_generators(data_dir: str, img_size: tuple = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """
    Build train / val / test ImageDataGenerators with augmentation on train set.

    Args:
        data_dir  : Root directory that contains train/, val/, test/ sub-folders.
        img_size  : Target (height, width) for all images.
        batch_size: Mini-batch size.

    Returns:
        train_gen, val_gen, test_gen  – Keras DirectoryIterators
        class_names                   – Sorted list of class labels
    """
    # ── Train augmentation ────────────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    # ── Validation / Test  –  only rescale ───────────────────────────────────
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"[DataLoader] Classes ({len(class_names)}): {class_names}")
    print(f"[DataLoader] Train samples : {train_gen.samples}")
    print(f"[DataLoader] Val   samples : {val_gen.samples}")
    print(f"[DataLoader] Test  samples : {test_gen.samples}")

    return train_gen, val_gen, test_gen, class_names


def visualize_samples(generator, class_names: list, n: int = 9, save_path: str = None):
    """Plot a grid of sample images with class labels."""
    images, labels = next(generator)
    n = min(n, len(images))

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[np.argmax(labels[i])], fontsize=11)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Training Images", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[DataLoader] Sample grid saved → {save_path}")
    plt.close()


def plot_class_distribution(generator, class_names: list, save_path: str = None):
    """Bar-chart of samples per class."""
    counts = {cls: 0 for cls in class_names}
    for cls, idx in generator.class_indices.items():
        n = np.sum(np.array(generator.classes) == idx)
        counts[cls] = n

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.9), 5))
    bars = ax.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="black", alpha=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Fish Species", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution – Training Set", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[DataLoader] Class distribution saved → {save_path}")
    plt.close()
