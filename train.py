"""
Main Training Script – Multiclass Fish Image Classification
Trains 1 custom CNN + 5 transfer-learning models, evaluates all, saves the
best model, and produces a full comparison report with visualisations.

Usage
-----
    python train.py --data_dir /path/to/dataset --epochs 25 --output_dir outputs/
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import get_data_generators, visualize_samples, plot_class_distribution
from utils.evaluation import (
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    plot_model_comparison,
    save_comparison_report,
)
from models.custom_cnn import build_cnn
from models.transfer_models import build_transfer_model


# ─── CLI Arguments ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Fish Image Classification Trainer")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Root directory with train/, val/, test/ sub-folders")
    parser.add_argument("--output_dir",  type=str, default="outputs",
                        help="Directory to save models, plots, and reports")
    parser.add_argument("--epochs",      type=int, default=25,
                        help="Number of training epochs per model")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                        help="Additional epochs for fine-tuning transfer models")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning phase")
    return parser.parse_args()


# ─── Callbacks Factory ────────────────────────────────────────────────────────
def make_callbacks(model_name: str, ckpt_dir: str, monitor: str = "val_accuracy"):
    """Return a list of standard Keras callbacks."""
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.h5")
    return [
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(ckpt_dir, "logs", model_name),
            histogram_freq=0,
        ),
    ]


# ─── Single-Model Training Pipeline ──────────────────────────────────────────
def train_model(
    model,
    model_name: str,
    train_gen,
    val_gen,
    test_gen,
    class_names: list,
    epochs: int,
    output_dir: str,
    lr: float,
    fine_tune: bool = False,
    fine_tune_epochs: int = 10,
    fine_tune_lr: float = 1e-5,
    fine_tune_at: int = None,
):
    """
    Compile, train, optionally fine-tune, and evaluate a model.

    Returns
    -------
    dict with training history and evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Feature extraction ───────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    t0 = time.time()
    hist1 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=make_callbacks(model_name, output_dir),
        verbose=1,
    )
    elapsed = time.time() - t0

    combined_history = {k: list(v) for k, v in hist1.history.items()}

    # ── Phase 2: Fine-tuning (transfer models only) ───────────────────────────
    if fine_tune and fine_tune_at is not None:
        print(f"\n[Fine-tune] Unfreezing from layer {fine_tune_at} …")
        # Un-freeze backbone layers
        for layer in model.layers:
            if hasattr(layer, "layers"):          # sub-model (backbone)
                for sub in layer.layers[fine_tune_at:]:
                    sub.trainable = True

        model.compile(
            optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        hist2 = model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=make_callbacks(f"{model_name}_finetune", output_dir),
            verbose=1,
        )
        for k, v in hist2.history.items():
            combined_history[k].extend(list(v))

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, f"{model_name}_final.h5")
    model.save(final_path)
    print(f"[Train] Model saved → {final_path}")

    # ── Plot training history ─────────────────────────────────────────────────
    plot_training_history(combined_history, model_name, save_dir=output_dir)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\n[Eval] Evaluating {model_name} on test set …")
    metrics = evaluate_model(model, test_gen, class_names)
    metrics["training_time_s"] = round(elapsed, 1)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, model_name,
                          save_dir=output_dir)

    print(f"\n[Result] {model_name}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Train time: {metrics['training_time_s']}s")
    print(metrics["report"])

    return {"history": combined_history, "metrics": metrics}


# ─── Fine-tune layer index heuristics ─────────────────────────────────────────
_FINE_TUNE_AT = {
    "VGG16": 15,
    "ResNet50": 143,
    "MobileNetV2": 100,
    "InceptionV3": 249,
    "EfficientNetB0": 200,
}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    IMG_SIZE = (args.img_size, args.img_size)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── GPU setup ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Setup] GPUs available: {len(gpus)}")
    else:
        print("[Setup] No GPU found – running on CPU.")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen, class_names = get_data_generators(
        args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    num_classes = len(class_names)

    # ── Save class names for Streamlit app ────────────────────────────────────
    classes_path = os.path.join(args.output_dir, "class_names.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Setup] Class names saved → {classes_path}")

    # ── EDA visualisations ────────────────────────────────────────────────────
    visualize_samples(train_gen, class_names,
                      save_path=os.path.join(args.output_dir, "sample_images.png"))
    plot_class_distribution(train_gen, class_names,
                             save_path=os.path.join(args.output_dir, "class_distribution.png"))

    # ─────────────────────────────────────────────────────────────────────────
    # Build model catalogue
    # ─────────────────────────────────────────────────────────────────────────
    model_configs = {
        "CustomCNN": {
            "model": build_cnn(num_classes, IMG_SIZE),
            "fine_tune": False,
            "fine_tune_at": None,
        },
        "VGG16": {
            "model": build_transfer_model("VGG16", num_classes, IMG_SIZE),
            "fine_tune": True,
            "fine_tune_at": _FINE_TUNE_AT["VGG16"],
        },
        "ResNet50": {
            "model": build_transfer_model("ResNet50", num_classes, IMG_SIZE),
            "fine_tune": True,
            "fine_tune_at": _FINE_TUNE_AT["ResNet50"],
        },
        "MobileNetV2": {
            "model": build_transfer_model("MobileNetV2", num_classes, IMG_SIZE),
            "fine_tune": True,
            "fine_tune_at": _FINE_TUNE_AT["MobileNetV2"],
        },
        "InceptionV3": {
            "model": build_transfer_model("InceptionV3", num_classes, IMG_SIZE),
            "fine_tune": True,
            "fine_tune_at": _FINE_TUNE_AT["InceptionV3"],
        },
        "EfficientNetB0": {
            "model": build_transfer_model("EfficientNetB0", num_classes, IMG_SIZE),
            "fine_tune": True,
            "fine_tune_at": _FINE_TUNE_AT["EfficientNetB0"],
        },
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Train all models
    # ─────────────────────────────────────────────────────────────────────────
    all_results = {}

    for model_name, cfg in model_configs.items():
        result = train_model(
            model=cfg["model"],
            model_name=model_name,
            train_gen=train_gen,
            val_gen=val_gen,
            test_gen=test_gen,
            class_names=class_names,
            epochs=args.epochs,
            output_dir=args.output_dir,
            lr=args.lr,
            fine_tune=cfg["fine_tune"],
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_lr=args.fine_tune_lr,
            fine_tune_at=cfg["fine_tune_at"],
        )
        # Keep only serialisable metric fields
        all_results[model_name] = {
            k: v for k, v in result["metrics"].items()
            if k not in ("confusion_matrix", "y_true", "y_pred", "y_pred_proba", "report")
        }
        all_results[model_name]["report"] = result["metrics"]["report"]
        all_results[model_name]["confusion_matrix"] = result["metrics"]["confusion_matrix"]

    # ─────────────────────────────────────────────────────────────────────────
    # Comparison plots & report
    # ─────────────────────────────────────────────────────────────────────────
    plot_model_comparison(all_results, save_dir=args.output_dir)
    best_model_name = save_comparison_report(
        all_results, save_path=os.path.join(args.output_dir, "comparison_report.md")
    )

    # ── Copy best model to a canonical path ──────────────────────────────────
    import shutil
    best_src = os.path.join(args.output_dir, f"{best_model_name}_final.h5")
    best_dst = os.path.join(args.output_dir, "best_model.h5")
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\n[Best] '{best_model_name}' (acc={all_results[best_model_name]['accuracy']:.4f})")
        print(f"[Best] Saved as → {best_dst}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        name: {k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in m.items()
               if k not in ("report", "confusion_matrix")}
        for name, m in all_results.items()
    }
    summary["best_model"] = best_model_name

    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Training complete. All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
