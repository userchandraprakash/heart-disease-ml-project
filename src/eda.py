"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates statistical summaries and EDA visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "plots")


def print_statistical_summary(df: pd.DataFrame) -> None:
    """
    Print a detailed statistical summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.
    """
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print("\n📊 Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    print(f"\n📊 Target Distribution:")
    target_counts = df["target"].value_counts()
    for val, count in target_counts.items():
        label = "Heart Disease" if val == 1 else "No Disease"
        pct = count / len(df) * 100
        print(f"  {label} ({val}): {count} ({pct:.1f}%)")

    print(f"\n📊 Correlation with Target (top 5):")
    correlations = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
    for feat, corr in correlations.head(5).items():
        print(f"  {feat:15s}: {corr:.3f}")


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot a correlation heatmap of all features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save : bool
        Whether to save the plot to disk.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save : bool
        Whether to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    colors = ["#2ecc71", "#e74c3c"]
    counts = df["target"].value_counts()
    axes[0].bar(
        ["No Disease (0)", "Heart Disease (1)"],
        [counts[0], counts[1]],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_title("Target Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")

    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        [counts[0], counts[1]],
        labels=["No Disease", "Heart Disease"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.05),
        shadow=True,
    )
    axes[1].set_title("Target Proportion", fontweight="bold")

    plt.suptitle("Heart Disease Target Variable Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "target_distribution.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot distributions of numerical features grouped by target class.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save : bool
        Whether to save the plot.
    """
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    available_features = [f for f in numerical_features if f in df.columns]

    n_features = len(available_features)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, feature in enumerate(available_features):
        for target_val, color, label in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
            subset = df[df["target"] == target_val][feature]
            axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="black", linewidth=0.3)

        axes[i].set_title(f"{feature}", fontweight="bold")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=9)

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Target Class", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "feature_distributions.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def plot_categorical_analysis(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot analysis of categorical features vs target.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save : bool
        Whether to save the plot.
    """
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    available_features = [f for f in categorical_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, feature in enumerate(available_features):
        ct = pd.crosstab(df[feature], df["target"])
        ct.plot(kind="bar", ax=axes[i], color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5)
        axes[i].set_title(f"{feature} vs Target", fontweight="bold")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Count")
        axes[i].legend(["No Disease", "Disease"], fontsize=9)
        axes[i].tick_params(axis="x", rotation=0)

    for j in range(len(available_features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Features vs Target", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filepath = os.path.join(PLOTS_DIR, "categorical_analysis.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {filepath}")

    plt.close(fig)


def run_eda(df: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save_plots : bool
        Whether to save plots to disk.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # Statistical summary
    print_statistical_summary(df)

    # Generate plots
    print("\n📊 Generating EDA Plots...")
    plot_correlation_heatmap(df, save=save_plots)
    plot_target_distribution(df, save=save_plots)
    plot_feature_distributions(df, save=save_plots)
    plot_categorical_analysis(df, save=save_plots)

    print("\n  [DONE] All EDA plots generated!")


if __name__ == "__main__":
    from data_loader import load_heart_data

    df = load_heart_data()
    run_eda(df)
