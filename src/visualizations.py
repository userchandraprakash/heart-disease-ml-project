"""
Visualizations Module
=====================
Generates result visualizations: model comparison, confusion matrices, ROC curves,
and feature importance plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set_style("whitegrid")

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "plots")


def plot_model_comparison(results: dict, save: bool = True) -> None:
    """
    Plot a bar chart comparing all models across key metrics.

    Parameters
    ----------
    results : dict
        Dictionary of model results from train_and_evaluate_all().
    save : bool
        Whether to save the plot.
    """
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data = {m: [results[name][m] for name in model_names] for m in metrics}

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        bars = ax.bar(x + i * width, data[metric], width, label=label, color=color, edgecolor="black", linewidth=0.5)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "model_comparison.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_confusion_matrices(results: dict, y_test, save: bool = True) -> None:
    """
    Plot confusion matrices for all models in a grid.

    Parameters
    ----------
    results : dict
        Model results dictionary.
    y_test : array-like
        True test labels.
    save : bool
        Whether to save the plot.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]

    for i, name in enumerate(model_names):
        cm = results[name]["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            cbar=False,
        )
        axes[i].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "confusion_matrices.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_roc_curves(results: dict, y_test, save: bool = True) -> None:
    """
    Plot overlaid ROC curves for all models.

    Parameters
    ----------
    results : dict
        Model results dictionary.
    y_test : array-like
        True test labels.
    save : bool
        Whether to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (name, metrics), color in zip(results.items(), colors):
        if metrics["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "roc_curves.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_feature_importance(results: dict, feature_names: list, save: bool = True) -> None:
    """
    Plot feature importance from tree-based models.

    Parameters
    ----------
    results : dict
        Model results dictionary.
    feature_names : list
        List of feature names.
    save : bool
        Whether to save the plot.
    """
    tree_models = {}
    for name, data in results.items():
        model = data["model"]
        if hasattr(model, "feature_importances_"):
            tree_models[name] = model.feature_importances_

    if not tree_models:
        print("  [SKIP] No tree-based models found for feature importance.")
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, importances) in zip(axes, tree_models.items()):
        # Sort by importance
        indices = np.argsort(importances)
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_features)))

        ax.barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Feature Importance\n{name}", fontsize=13, fontweight="bold")

        # Add value labels
        for j, (imp, feat) in enumerate(zip(sorted_importances, sorted_features)):
            ax.text(imp + 0.002, j, f"{imp:.3f}", va="center", fontsize=9)

    plt.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "feature_importance.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def generate_all_result_plots(results: dict, y_test, feature_names: list) -> None:
    """
    Generate all result visualization plots.

    Parameters
    ----------
    results : dict
        Model results from train_and_evaluate_all().
    y_test : array-like
        True test labels.
    feature_names : list
        List of feature names.
    """
    print("\n" + "=" * 60)
    print("GENERATING RESULT VISUALIZATIONS")
    print("=" * 60)

    print("\n📊 Creating plots...")
    plot_model_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_feature_importance(results, feature_names)

    print(f"\n  [DONE] All result plots saved to: {PLOTS_DIR}/")
