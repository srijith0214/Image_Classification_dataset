"""
Model Evaluation Utilities
Multiclass Fish Image Classification Project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, test_gen, class_names: list):
    """
    Evaluate a Keras model on the test generator.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1,
                    report (str), confusion_matrix (ndarray),
                    y_true, y_pred
    """
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def plot_training_history(history, model_name: str, save_dir: str = None):
    """Plot accuracy and loss curves for a single model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ─────────────────────────────────────────────────────────────
    axes[0].plot(history["accuracy"], label="Train Acc", color="royalblue", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Val Acc", color="darkorange",
                 linewidth=2, linestyle="--")
    axes[0].set_title(f"{model_name} – Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Loss ─────────────────────────────────────────────────────────────────
    axes[1].plot(history["loss"], label="Train Loss", color="royalblue", linewidth=2)
    axes[1].plot(history["val_loss"], label="Val Loss", color="darkorange",
                 linewidth=2, linestyle="--")
    axes[1].set_title(f"{model_name} – Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{model_name}_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Training history saved → {path}")
    plt.close()


def plot_confusion_matrix(cm, class_names: list, model_name: str, save_dir: str = None):
    """Plot and optionally save a styled confusion matrix heatmap."""
    fig_size = max(8, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{model_name} – Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Confusion matrix saved → {path}")
    plt.close()


def plot_model_comparison(results: dict, save_dir: str = None):
    """
    Bar chart comparing accuracy, precision, recall, F1 across all models.

    Parameters
    ----------
    results : dict  {model_name: {accuracy, precision, recall, f1, ...}}
    """
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 2), 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison – Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Model comparison chart saved → {path}")
    plt.close()


def save_comparison_report(results: dict, save_path: str):
    """Write a markdown/text comparison table to disk."""
    lines = ["# Model Comparison Report\n",
             "## Fish Image Classification – All Models\n",
             f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n",
             "-" * 70 + "\n"]

    best_model = max(results, key=lambda m: results[m]["accuracy"])

    for name, metrics in results.items():
        tag = " ← BEST" if name == best_model else ""
        lines.append(
            f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}{tag}\n"
        )

    lines.append("\n\n## Per-Class Classification Reports\n")
    for name, metrics in results.items():
        lines.append(f"\n### {name}\n```\n{metrics['report']}\n```\n")

    with open(save_path, "w") as f:
        f.writelines(lines)

    print(f"[Eval] Comparison report saved → {save_path}")
    return best_model
