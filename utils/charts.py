import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb limit for large datasets

# Matplotlib style — only use valid rcParams
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
})

BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
RED = "#dc2626"


def plot_distributions(df: pd.DataFrame, num_cols: list) -> list:
    """Return a list of matplotlib figures — one per numeric column (hist + boxplot)."""
    figs = []
    for col in num_cols:
        series = df[col].dropna()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        fig.suptitle(col, fontsize=13, fontweight="bold", y=1.02)

        # Histogram with KDE
        ax1.hist(series, bins=25, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax1.set_xlabel(col)
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution")

        # Add mean/median lines
        ax1.axvline(series.mean(), color=ORANGE, linestyle="--", linewidth=1.5, label=f"Mean: {series.mean():.1f}")
        ax1.axvline(series.median(), color=GREEN, linestyle=":", linewidth=1.5, label=f"Median: {series.median():.1f}")
        ax1.legend(fontsize=9)

        # Boxplot
        bp = ax2.boxplot(series, vert=True, patch_artist=True,
                         boxprops=dict(facecolor=BLUE, alpha=0.5),
                         medianprops=dict(color=ORANGE, linewidth=2),
                         whiskerprops=dict(linewidth=1.2),
                         flierprops=dict(marker="o", color=RED, alpha=0.5, markersize=4))
        ax2.set_ylabel(col)
        ax2.set_title("Box plot")
        ax2.set_xticks([])

        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list):
    """Return a seaborn heatmap figure for numeric correlations."""
    corr = df[num_cols].corr()
    n = len(num_cols)
    fig_size = max(6, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title("Correlation Matrix (lower triangle)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def plot_outliers(df: pd.DataFrame, col: str):
    """Return a side-by-side strip + box plot highlighting outliers."""
    series = df[col].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (series < lower) | (series > upper)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle(f"{col} — Outlier Analysis", fontsize=12, fontweight="bold")

    # Strip plot — sample if huge to avoid giant images
    MAX_POINTS = 2000
    if len(series) > MAX_POINTS:
        idx = np.random.choice(len(series), MAX_POINTS, replace=False)
        s_vals = series.iloc[idx].values
        o_vals = is_outlier.iloc[idx].values
    else:
        s_vals = series.values
        o_vals = is_outlier.values
    colors = [RED if o else BLUE for o in o_vals]
    ax1.scatter(range(len(s_vals)), s_vals, c=colors, alpha=0.4, s=15)
    ax1.axhline(upper, color=RED, linestyle="--", linewidth=1, label=f"Upper bound: {upper:.1f}")
    ax1.axhline(lower, color=ORANGE, linestyle="--", linewidth=1, label=f"Lower bound: {lower:.1f}")
    ax1.set_xlabel("Index")
    ax1.set_ylabel(col)
    ax1.set_title(f"{is_outlier.sum()} outliers (red)")
    ax1.legend(fontsize=8)

    # Box plot
    bp = ax2.boxplot(series, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=BLUE, alpha=0.4),
                     medianprops=dict(color=ORANGE, linewidth=2),
                     flierprops=dict(marker="o", color=RED, alpha=0.6, markersize=5))
    ax2.set_title("Box plot")
    ax2.set_xticks([])

    plt.tight_layout()
    return fig