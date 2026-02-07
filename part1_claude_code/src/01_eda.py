"""Exploratory Data Analysis on the Wine dataset.

Loads the dataset, computes summary statistics, generates distribution
plots, creates a correlation heatmap, checks class balance, and identifies
outliers using the IQR method.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
DISTRIBUTIONS_DIR: str = "output/distributions"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5
TARGET_COLUMN: str = "target"


def _ensure_output_dirs(
    output_dir: str,
    distributions_dir: str,
) -> tuple[Path, Path]:
    """Create output directories if they do not exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dist_path = Path(distributions_dir)
    dist_path.mkdir(parents=True, exist_ok=True)
    return output_path, dist_path


def _load_dataset() -> pl.DataFrame:
    """Load the Wine dataset and return as a polars DataFrame."""
    wine = load_wine()
    feature_names = list(wine.feature_names)
    data = wine.data
    target = wine.target

    df = pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, target.astype(float)))

    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def _compute_summary_statistics(
    df: pl.DataFrame,
) -> dict:
    """Compute summary statistics for all columns."""
    stats = {}
    for col in df.columns:
        col_data = df[col]
        stats[col] = {
            "mean": round(float(col_data.mean()), 4),
            "median": round(float(col_data.median()), 4),
            "std": round(float(col_data.std()), 4),
            "min": round(float(col_data.min()), 4),
            "max": round(float(col_data.max()), 4),
        }

    logger.info(f"Summary statistics:\n{json.dumps(stats, indent=2, default=str)}")
    return stats


def _check_missing_values(
    df: pl.DataFrame,
) -> dict:
    """Check for missing values in each column."""
    missing = {}
    for col in df.columns:
        null_count = df[col].null_count()
        missing[col] = null_count

    total_missing = sum(missing.values())
    logger.info(f"Total missing values: {total_missing}")
    if total_missing > 0:
        logger.warning(f"Missing values found:\n{json.dumps(missing, indent=2, default=str)}")
    else:
        logger.info("No missing values found in the dataset.")

    return missing


def _plot_distributions(
    df: pl.DataFrame,
    dist_path: Path,
) -> None:
    """Generate individual histogram distribution plots for each feature."""
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]

    for col in feature_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = df[col].to_list()
        ax.hist(values, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        safe_name = col.replace("/", "_")
        filepath = dist_path / f"{safe_name}.png"
        plt.savefig(filepath, dpi=FIGURE_DPI)
        plt.close()

    logger.info(f"Distribution plots saved to {dist_path} ({len(feature_columns)} features)")


def _plot_correlation_matrix(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a correlation matrix heatmap."""
    columns = [c for c in df.columns if c != TARGET_COLUMN]
    corr_data = {}
    for col in columns:
        correlations = []
        for other_col in columns:
            corr_value = df.select(pl.corr(col, other_col)).item()
            correlations.append(round(float(corr_value), 3))
        corr_data[col] = correlations

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        pl.DataFrame(corr_data).to_numpy(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=columns,
        yticklabels=columns,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    filepath = output_path / "correlation_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Correlation matrix saved to {filepath}")


def _plot_class_balance(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a bar chart showing class distribution."""
    class_counts = df.group_by(TARGET_COLUMN).len().sort(TARGET_COLUMN)
    classes = [int(v) for v in class_counts[TARGET_COLUMN].to_list()]
    counts = class_counts["len"].to_list()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"Class {c}" for c in classes], counts, edgecolor="black", alpha=0.7)
    ax.set_title("Wine Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center", fontweight="bold")

    plt.tight_layout()
    filepath = output_path / "class_balance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Class balance plot saved to {filepath}")
    logger.info(
        f"Class distribution:\n{json.dumps(dict(zip(classes, counts)), indent=2, default=str)}"
    )


def _detect_outliers(
    df: pl.DataFrame,
    output_path: Path,
) -> dict:
    """Detect outliers using IQR method and generate box plots."""
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    outlier_counts = {}

    for col in feature_columns:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        outlier_count = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).height
        outlier_counts[col] = outlier_count

    logger.info(
        f"Outlier counts (IQR method):\n{json.dumps(outlier_counts, indent=2, default=str)}"
    )

    # Box plot visualization
    n_cols = 3
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_columns):
        axes[i].boxplot(df[col].to_list(), vert=True)
        axes[i].set_title(f"{col} ({outlier_counts[col]} outliers)")

    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = output_path / "outliers.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Outlier box plots saved to {filepath}")

    return outlier_counts


def _save_data(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Save the DataFrame to parquet."""
    filepath = output_path / "wine_data.parquet"
    df.write_parquet(filepath)
    logger.info(f"Dataset saved to {filepath}")


def run_eda() -> None:
    """Run the full exploratory data analysis pipeline."""
    start_time = time.time()
    logger.info("Starting exploratory data analysis...")

    output_path, dist_path = _ensure_output_dirs(OUTPUT_DIR, DISTRIBUTIONS_DIR)
    df = _load_dataset()
    _compute_summary_statistics(df)
    _check_missing_values(df)
    _plot_distributions(df, dist_path)
    _plot_correlation_matrix(df, output_path)
    _plot_class_balance(df, output_path)
    _detect_outliers(df, output_path)
    _save_data(df, output_path)

    elapsed = time.time() - start_time
    logger.info(f"EDA completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_eda()
