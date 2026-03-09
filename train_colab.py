# ============================================================
# Multiclass Fish Image Classification — Google Colab Training
# Run each cell in order top → bottom
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive & set project root
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

import os

# ✏️  Change this to where you uploaded the project on Drive
PROJECT_ROOT = "/content/drive/MyDrive/fish_classification"
os.chdir(PROJECT_ROOT)
print("Working directory:", os.getcwd())
print("Files found:", os.listdir("."))


# ─────────────────────────────────────────────────────────────
# CELL 2 — Install / verify dependencies
# ─────────────────────────────────────────────────────────────
# TensorFlow, scikit-learn, seaborn are pre-installed on Colab.
# Only install what is missing.
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import seaborn
except ImportError:
    install("seaborn")

try:
    import sklearn
except ImportError:
    install("scikit-learn")

print("✓ All dependencies ready.")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Clone dataset from GitHub into Colab /content
# ─────────────────────────────────────────────────────────────
import subprocess

DATASET_DIR = "/content/Image_Classification_dataset"

if not os.path.exists(DATASET_DIR):
    print("Cloning dataset …")
    subprocess.run([
        "git", "clone",
        "https://github.com/srijith0214/Image_Classification_dataset.git",
        DATASET_DIR
    ], check=True)
    print("✓ Dataset cloned →", DATASET_DIR)
else:
    print("✓ Dataset already exists →", DATASET_DIR)

# Verify splits exist
for split in ["train", "val", "test"]:
    path = os.path.join(DATASET_DIR, split)
    if os.path.isdir(path):
        classes = os.listdir(path)
        print(f"  {split:5s} → {len(classes)} classes, e.g. {classes[:3]}")
    else:
        print(f"  ⚠️  '{split}' folder NOT found — check repo structure")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Config  (edit these values as needed)
# ─────────────────────────────────────────────────────────────
# ── Paths ─────────────────────────────────────────────────────
DATA_DIR   = DATASET_DIR                          # cloned repo
OUTPUT_DIR = "/content/drive/MyDrive/fish_outputs"  # saved to Drive

# ── Hyperparameters ───────────────────────────────────────────
IMG_SIZE         = 224      # pixels (height = width)
BATCH_SIZE       = 32
EPOCHS           = 25       # phase-1 epochs per model
FINE_TUNE_EPOCHS = 10       # phase-2 fine-tune epochs
LR               = 1e-3     # phase-1 learning rate
FINE_TUNE_LR     = 1e-5     # phase-2 learning rate

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Config set. Outputs → {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────
# CELL 5 — GPU check
# ─────────────────────────────────────────────────────────────
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU available: {gpus[0].name}")
else:
    print("⚠️  No GPU — go to Runtime → Change runtime type → GPU")

print("TensorFlow version:", tf.__version__)


# ─────────────────────────────────────────────────────────────
# CELL 6 — Imports
# ─────────────────────────────────────────────────────────────
import sys
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids Colab display issues
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0,
)

print("✓ All imports successful.")


# ─────────────────────────────────────────────────────────────
# CELL 7 — Data pipeline
# ─────────────────────────────────────────────────────────────
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
SEED = 42

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_gen = eval_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

test_gen = eval_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = list(train_gen.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)

print(f"\n✓ Data loaded")
print(f"  Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"  Train  : {train_gen.samples} images")
print(f"  Val    : {val_gen.samples} images")
print(f"  Test   : {test_gen.samples} images")

# Save class names for the Streamlit app
with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)
print(f"  class_names.json saved → {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────
# CELL 8 — EDA: sample grid + class distribution
# ─────────────────────────────────────────────────────────────
# Sample images grid
images, labels = next(train_gen)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.set_title(CLASS_NAMES[np.argmax(labels[i])], fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Training Images", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=120, bbox_inches="tight")
plt.show()

# Class distribution bar chart
counts = {cls: int(np.sum(np.array(train_gen.classes) == idx))
          for cls, idx in train_gen.class_indices.items()}
fig, ax = plt.subplots(figsize=(max(10, NUM_CLASSES), 4))
bars = ax.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="black", alpha=0.85)
ax.bar_label(bars, padding=3)
ax.set_title("Class Distribution – Training Set", fontsize=13, fontweight="bold")
ax.set_xlabel("Species"); ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=120, bbox_inches="tight")
plt.show()
print("✓ EDA plots saved.")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Helper functions (callbacks, evaluation, plots)
# ─────────────────────────────────────────────────────────────
def make_callbacks(model_name: str, out_dir: str):
    ckpt = os.path.join(out_dir, f"{model_name}_best.h5")
    return [
        callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-7, verbose=1),
    ]


def evaluate_model(model, gen):
    gen.reset()
    y_pred_proba = model.predict(gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = gen.classes
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "precision":       precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":          recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":              f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report":          classification_report(y_true, y_pred,
                                                 target_names=CLASS_NAMES, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_history(history: dict, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["accuracy"],     label="Train", color="royalblue",  lw=2)
    axes[0].plot(history["val_accuracy"], label="Val",   color="darkorange", lw=2, ls="--")
    axes[0].set_title(f"{model_name} – Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss"],     label="Train", color="royalblue",  lw=2)
    axes[1].plot(history["val_loss"], label="Val",   color="darkorange", lw=2, ls="--")
    axes[1].set_title(f"{model_name} – Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_history.png"), dpi=120, bbox_inches="tight")
    plt.show()


def plot_cm(cm, model_name: str):
    fsz = max(7, NUM_CLASSES * 0.85)
    fig, ax = plt.subplots(figsize=(fsz, fsz * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} – Confusion Matrix", fontweight="bold")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"),
                dpi=120, bbox_inches="tight")
    plt.show()


print("✓ Helper functions defined.")


# ─────────────────────────────────────────────────────────────
# CELL 10 — Model builders
# ─────────────────────────────────────────────────────────────
def build_custom_cnn(num_classes, img_size=224):
    inp = (img_size, img_size, 3)
    m = models.Sequential(name="CustomCNN")
    m.add(layers.Input(shape=inp))

    for filters in [32, 64, 128, 256]:
        m.add(layers.Conv2D(filters, (3,3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
        m.add(layers.BatchNormalization()); m.add(layers.Activation("relu"))
        m.add(layers.Conv2D(filters, (3,3), padding="same",
                            kernel_regularizer=regularizers.l2(1e-4)))
        m.add(layers.BatchNormalization()); m.add(layers.Activation("relu"))
        m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation("relu"))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(num_classes, activation="softmax"))
    return m


_BACKBONE_MAP = {
    "VGG16":         VGG16,
    "ResNet50":      ResNet50,
    "MobileNetV2":   MobileNetV2,
    "InceptionV3":   InceptionV3,
    "EfficientNetB0": EfficientNetB0,
}

_FINE_TUNE_AT = {
    "VGG16": 15, "ResNet50": 143, "MobileNetV2": 100,
    "InceptionV3": 249, "EfficientNetB0": 200,
}


def build_transfer_model(name, num_classes, img_size=224):
    backbone = _BACKBONE_MAP[name](
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = False

    inp = layers.Input(shape=(img_size, img_size, 3))
    x   = backbone(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name=name)


print("✓ Model builder functions defined.")


# ─────────────────────────────────────────────────────────────
# CELL 11 — Training loop
# ─────────────────────────────────────────────────────────────
ALL_RESULTS = {}   # stores metrics for every model

MODEL_CONFIGS = {
    "CustomCNN":     {"builder": lambda: build_custom_cnn(NUM_CLASSES, IMG_SIZE),
                      "fine_tune": False, "fine_tune_at": None},
    "VGG16":         {"builder": lambda: build_transfer_model("VGG16",         NUM_CLASSES, IMG_SIZE),
                      "fine_tune": True,  "fine_tune_at": _FINE_TUNE_AT["VGG16"]},
    "ResNet50":      {"builder": lambda: build_transfer_model("ResNet50",      NUM_CLASSES, IMG_SIZE),
                      "fine_tune": True,  "fine_tune_at": _FINE_TUNE_AT["ResNet50"]},
    "MobileNetV2":   {"builder": lambda: build_transfer_model("MobileNetV2",   NUM_CLASSES, IMG_SIZE),
                      "fine_tune": True,  "fine_tune_at": _FINE_TUNE_AT["MobileNetV2"]},
    "InceptionV3":   {"builder": lambda: build_transfer_model("InceptionV3",   NUM_CLASSES, IMG_SIZE),
                      "fine_tune": True,  "fine_tune_at": _FINE_TUNE_AT["InceptionV3"]},
    "EfficientNetB0":{"builder": lambda: build_transfer_model("EfficientNetB0",NUM_CLASSES, IMG_SIZE),
                      "fine_tune": True,  "fine_tune_at": _FINE_TUNE_AT["EfficientNetB0"]},
}

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*60}\n  ▶  Training : {model_name}\n{'='*60}")

    model = cfg["builder"]()
    model.summary(line_length=90)

    # ── Phase 1 : feature extraction ─────────────────────────────────────────
    model.compile(optimizer=optimizers.Adam(LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    t0   = time.time()
    hist = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                     callbacks=make_callbacks(model_name, OUTPUT_DIR), verbose=1)
    combined = {k: list(v) for k, v in hist.history.items()}

    # ── Phase 2 : fine-tuning (transfer models only) ─────────────────────────
    if cfg["fine_tune"] and cfg["fine_tune_at"] is not None:
        ft_at = cfg["fine_tune_at"]
        print(f"\n[Fine-tune] Unfreezing backbone from layer {ft_at} …")
        for layer in model.layers:
            if hasattr(layer, "layers"):          # the backbone sub-model
                for sub in layer.layers[ft_at:]:
                    sub.trainable = True

        model.compile(optimizer=optimizers.Adam(FINE_TUNE_LR),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        hist2 = model.fit(train_gen, epochs=FINE_TUNE_EPOCHS, validation_data=val_gen,
                          callbacks=make_callbacks(f"{model_name}_ft", OUTPUT_DIR), verbose=1)
        for k, v in hist2.history.items():
            combined[k].extend(list(v))

    elapsed = time.time() - t0

    # ── Save model ────────────────────────────────────────────────────────────
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_final.h5")
    model.save(save_path)
    print(f"[Saved] {save_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(combined, model_name)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, test_gen)
    metrics["training_time_s"] = round(elapsed, 1)
    plot_cm(metrics["confusion_matrix"], model_name)

    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Time      : {metrics['training_time_s']}s")
    print(metrics["report"])

    ALL_RESULTS[model_name] = metrics

    # ── Free GPU memory before next model ─────────────────────────────────────
    del model
    tf.keras.backend.clear_session()

print("\n✓ All models trained.")


# ─────────────────────────────────────────────────────────────
# CELL 12 — Model comparison chart
# ─────────────────────────────────────────────────────────────
metric_keys   = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
model_names   = list(ALL_RESULTS.keys())
x = np.arange(len(model_names))
w = 0.2

fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 2), 6))
for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
    vals = [ALL_RESULTS[m][key] for m in model_names]
    bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85,
                  edgecolor="black", lw=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(model_names, rotation=20, ha="right")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score"); ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=130, bbox_inches="tight")
plt.show()
print("✓ Comparison chart saved.")


# ─────────────────────────────────────────────────────────────
# CELL 13 — Save comparison report & best model
# ─────────────────────────────────────────────────────────────
# ── Markdown report ───────────────────────────────────────────
report_lines = [
    "# Model Comparison Report\n\n",
    f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(s)':>10}\n",
    "-" * 75 + "\n",
]
best_model_name = max(ALL_RESULTS, key=lambda m: ALL_RESULTS[m]["accuracy"])
for name, m in ALL_RESULTS.items():
    tag = " ← BEST" if name == best_model_name else ""
    report_lines.append(
        f"{name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
        f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['training_time_s']:>10}{tag}\n"
    )
report_lines.append("\n\n## Per-Class Reports\n")
for name, m in ALL_RESULTS.items():
    report_lines.append(f"\n### {name}\n```\n{m['report']}\n```\n")

report_path = os.path.join(OUTPUT_DIR, "comparison_report.md")
with open(report_path, "w") as f:
    f.writelines(report_lines)
print(f"✓ Comparison report saved → {report_path}")

# ── Copy best model to canonical path ────────────────────────
best_src = os.path.join(OUTPUT_DIR, f"{best_model_name}_final.h5")
best_dst = os.path.join(OUTPUT_DIR, "best_model.h5")
shutil.copy2(best_src, best_dst)
print(f"✓ Best model : '{best_model_name}'  "
      f"(acc={ALL_RESULTS[best_model_name]['accuracy']:.4f})")
print(f"  Saved as   : {best_dst}")

# ── Summary JSON ──────────────────────────────────────────────
summary = {
    name: {k: float(v) if isinstance(v, (float, np.floating)) else v
           for k, v in m.items() if k not in ("report", "confusion_matrix")}
    for name, m in ALL_RESULTS.items()
}
summary["best_model"] = best_model_name
with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ results_summary.json saved.")


# ─────────────────────────────────────────────────────────────
# CELL 14 — Quick inference test (sanity check)
# ─────────────────────────────────────────────────────────────
from tensorflow.keras.models import load_model as keras_load
from PIL import Image as PILImage

best_model = keras_load(best_dst)
print(f"✓ Best model loaded from {best_dst}")

# Grab one test image for a quick sanity check
test_gen.reset()
sample_imgs, sample_labels = next(test_gen)
sample_img = sample_imgs[0:1]                  # shape (1, 224, 224, 3)

probs      = best_model.predict(sample_img, verbose=0)[0]
pred_idx   = np.argmax(probs)
true_idx   = np.argmax(sample_labels[0])

print(f"\n  True class      : {CLASS_NAMES[true_idx]}")
print(f"  Predicted class : {CLASS_NAMES[pred_idx]}")
print(f"  Confidence      : {probs[pred_idx]*100:.2f}%")

plt.imshow(sample_img[0])
plt.title(f"True: {CLASS_NAMES[true_idx]}  |  Pred: {CLASS_NAMES[pred_idx]} "
          f"({probs[pred_idx]*100:.1f}%)")
plt.axis("off")
plt.tight_layout()
plt.show()

print("\n🎉 Training pipeline complete!")
print(f"   All outputs saved to: {OUTPUT_DIR}")
