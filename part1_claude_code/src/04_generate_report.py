"""Generate a comprehensive model evaluation report from pipeline artifacts.

Loads the trained model, test data, and evaluation metrics from the
output directory and produces a filled-in markdown report.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
REPORT_FILENAME: str = "full_report.md"
TARGET_COLUMN: str = "target"
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]
TOTAL_SAMPLES: int = 178


def _load_json(
    filepath: Path,
) -> dict:
    """Load a JSON file and return as dict."""
    data = json.loads(filepath.read_text())
    logger.info(f"Loaded {filepath}")
    return data


def _load_model(
    output_path: Path,
) -> object:
    """Load the trained model from disk."""
    pkl_files = list(output_path.glob("*.pkl"))
    joblib_files = list(output_path.glob("*.joblib"))
    model_files = pkl_files + joblib_files

    if not model_files:
        raise FileNotFoundError(f"No model file found in {output_path}")

    model = joblib.load(model_files[0])
    logger.info(f"Loaded model from {model_files[0]}: {type(model).__name__}")
    return model


def _get_feature_names(
    output_path: Path,
) -> list[str]:
    """Extract feature names from training data."""
    train_df = pl.read_parquet(output_path / "train.parquet")
    return [c for c in train_df.columns if c != TARGET_COLUMN]


def _get_sample_counts(
    output_path: Path,
) -> tuple[int, int]:
    """Get training and test sample counts."""
    train_df = pl.read_parquet(output_path / "train.parquet")
    test_df = pl.read_parquet(output_path / "test.parquet")
    return train_df.shape[0], test_df.shape[0]


def _get_top_features(
    model: object,
    feature_names: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Extract top N features by importance."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    top = []
    for i in sorted_indices[:top_n]:
        top.append((feature_names[i], round(float(importances[i]), 4)))

    return top


def _build_report(
    eval_metrics: dict,
    per_class: dict,
    tuning_results: dict,
    train_count: int,
    test_count: int,
    feature_count: int,
    top_features: list[tuple[str, float]],
) -> str:
    """Build the full markdown report string."""
    test_m = eval_metrics["test_metrics"]
    cv = eval_metrics["cross_validation"]
    best_params = tuning_results["best_params"]

    report = "# Model Evaluation Report\n\n"

    # Executive Summary
    report += "## Executive Summary\n\n"
    report += (
        f"An XGBoost classifier was trained to classify wines into three cultivar "
        f"classes using {feature_count} features from the UCI Wine dataset. "
        f"After hyperparameter tuning via RandomizedSearchCV "
        f"({tuning_results['n_iterations']} iterations, "
        f"{tuning_results['cv_folds']}-fold stratified CV), the model achieved "
        f"test accuracy of {test_m['accuracy']} and a mean cross-validation "
        f"accuracy of {cv['cv_mean_accuracy']} (+/- {cv['cv_std_accuracy']}).\n\n"
    )

    # Dataset Overview
    report += "## Dataset Overview\n\n"
    report += "| Property | Value |\n"
    report += "|----------|-------|\n"
    report += f"| Total samples | {TOTAL_SAMPLES} |\n"
    report += f"| Training samples | {train_count} |\n"
    report += f"| Test samples | {test_count} |\n"
    report += f"| Number of features | {feature_count} (13 original + 3 engineered) |\n"
    report += "| Target variable | Wine cultivar class (0, 1, 2) |\n\n"

    # Model Configuration
    report += "## Model Configuration\n\n"
    report += "| Hyperparameter | Value |\n"
    report += "|----------------|-------|\n"
    report += "| Model type | XGBClassifier |\n"
    for param, value in sorted(best_params.items()):
        report += f"| {param} | {value} |\n"
    report += "\n"

    # Performance Metrics
    report += "## Performance Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| Test Accuracy | {test_m['accuracy']} |\n"
    report += f"| Precision (macro) | {test_m['precision_macro']} |\n"
    report += f"| Recall (macro) | {test_m['recall_macro']} |\n"
    report += f"| F1-Score (macro) | {test_m['f1_macro']} |\n"
    report += f"| CV Mean Accuracy | {cv['cv_mean_accuracy']} |\n"
    report += f"| CV Std Accuracy | {cv['cv_std_accuracy']} |\n\n"

    # Per-Class Metrics
    report += "### Per-Class Metrics\n\n"
    report += "| Class | Precision | Recall | F1-Score | Support |\n"
    report += "|-------|-----------|--------|----------|---------|\n"
    for class_name in CLASS_NAMES:
        c = per_class[class_name]
        report += (
            f"| {class_name} | {c['precision']} | {c['recall']} "
            f"| {c['f1_score']} | {c['support']} |\n"
        )
    report += "\n"

    # CV Per-Fold
    report += "### Cross-Validation Per-Fold Scores\n\n"
    report += "| Fold | Accuracy |\n"
    report += "|------|----------|\n"
    for i, score in enumerate(cv["cv_scores"], 1):
        report += f"| {i} | {score} |\n"
    report += "\n"

    # Feature Importance
    report += "## Feature Importance (Top 5)\n\n"
    report += "| Rank | Feature | Importance Score |\n"
    report += "|------|---------|------------------|\n"
    for rank, (name, score) in enumerate(top_features, 1):
        report += f"| {rank} | {name} | {score} |\n"
    report += "\n"

    # Recommendations
    report += "## Recommendations for Improvement\n\n"
    report += (
        "1. **Investigate potential overfitting**: Perfect test accuracy paired with "
        "lower CV accuracy and high fold variance suggests possible overfitting. "
        "Consider repeated stratified k-fold for more reliable estimates.\n"
    )
    report += (
        "2. **Reduce feature set**: With 16 features on only 178 samples, applying "
        "feature selection could simplify the model and reduce overfitting risk.\n"
    )
    report += (
        "3. **Regularize more aggressively**: Increasing gamma or min_child_weight "
        "may improve robustness and cross-validation stability.\n"
    )

    return report


def main() -> None:
    """Generate the full evaluation report from pipeline artifacts."""
    start_time = time.time()
    logger.info("Starting report generation...")

    output_path = Path(OUTPUT_DIR)

    eval_metrics = _load_json(output_path / "evaluation_metrics.json")
    per_class = _load_json(output_path / "per_class_metrics.json")
    tuning_results = _load_json(output_path / "tuning_results.json")

    model = _load_model(output_path)
    feature_names = _get_feature_names(output_path)
    train_count, test_count = _get_sample_counts(output_path)
    top_features = _get_top_features(model, feature_names)

    report = _build_report(
        eval_metrics,
        per_class,
        tuning_results,
        train_count,
        test_count,
        len(feature_names),
        top_features,
    )

    filepath = output_path / REPORT_FILENAME
    filepath.write_text(report)
    logger.info(f"Report saved to {filepath}")

    elapsed = time.time() - start_time
    logger.info(f"Report generation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
