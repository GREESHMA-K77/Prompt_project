"""
Evaluation Metrics calculation for Sentiment Analysis and Multi-Label Complaint Classification.
Calculates Macro F1-score, Micro F1, per-class metrics, confusion matrices, and generates visual plots.
"""

import os
import tempfile

# Configure fast non-interactive backend and writable cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))
import matplotlib
matplotlib.use("Agg")

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    hamming_loss,
    jaccard_score
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.schemas import ALL_COMPLAINT_CATEGORIES


SENTIMENT_CLASSES = ["Positive", "Negative", "Neutral"]


def compute_sentiment_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute Macro F1, precision, recall, accuracy, and confusion matrix for sentiment."""
    macro_f1 = f1_score(y_true, y_pred, labels=SENTIMENT_CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=SENTIMENT_CLASSES, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=SENTIMENT_CLASSES, average="macro", zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, labels=SENTIMENT_CLASSES, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_CLASSES)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "per_class_f1": {cls: float(f1) for cls, f1 in zip(SENTIMENT_CLASSES, per_class_f1)},
        "confusion_matrix": cm.tolist()
    }


def compute_multilabel_complaint_metrics(
    y_true_multilabel: List[List[str]],
    y_pred_multilabel: List[List[str]],
    classes: List[str] = ALL_COMPLAINT_CATEGORIES
) -> Dict[str, Any]:
    """Compute Multi-label Macro F1, Micro F1, Hamming Loss, and Jaccard Similarity."""
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true_multilabel)
    y_pred_bin = mlb.transform(y_pred_multilabel)

    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    per_class_f1 = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    jaccard = jaccard_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "hamming_loss": float(h_loss),
        "jaccard_similarity": float(jaccard),
        "per_category_f1": {cls: float(f1) for cls, f1 in zip(classes, per_class_f1)}
    }


def plot_f1_comparison(
    results_summary: Dict[str, Dict[str, Any]],
    output_path: str = "reports/prompt_comparison_chart.png"
):
    """Generate bar chart comparing Macro F1 scores across Zero-Shot, Few-Shot, and CoT."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    techniques = list(results_summary.keys())
    sentiment_f1s = [results_summary[t]["sentiment"]["macro_f1"] for t in techniques]
    complaint_f1s = [results_summary[t]["complaints"]["macro_f1"] for t in techniques]

    x = np.arange(len(techniques))
    width = 0.35

    plt.figure(figsize=(9, 5.5))
    bars1 = plt.bar(x - width/2, sentiment_f1s, width, label="Sentiment Macro F1", color="#3b82f6")
    bars2 = plt.bar(x + width/2, complaint_f1s, width, label="Complaint Categories Macro F1", color="#10b981")

    plt.xlabel("Prompting Technique", fontsize=12, fontweight="bold")
    plt.ylabel("Macro F1-Score", fontsize=12, fontweight="bold")
    plt.title("Comparative Performance Across Prompting Strategies (300 Reviews)", fontsize=13, fontweight="bold")
    plt.xticks(x, [t.replace("_", " ").title() for t in techniques], fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)

    # Attach values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Chart saved to {output_path}")


def plot_confusion_matrices(
    results_summary: Dict[str, Dict[str, Any]],
    output_path: str = "reports/sentiment_confusion_matrices.png"
):
    """Generate side-by-side heatmaps of sentiment confusion matrices."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    techniques = list(results_summary.keys())
    fig, axes = plt.subplots(1, len(techniques), figsize=(5 * len(techniques), 4))

    if len(techniques) == 1:
        axes = [axes]

    for idx, t in enumerate(techniques):
        cm = np.array(results_summary[t]["sentiment"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=SENTIMENT_CLASSES,
            yticklabels=SENTIMENT_CLASSES,
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f"{t.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Confusion matrices saved to {output_path}")
