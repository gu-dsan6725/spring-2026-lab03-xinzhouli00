"""Train and evaluate an XGBoost classifier on the Wine dataset.

Loads prepared train/test splits, trains an XGBClassifier, performs
cross-validation, runs hyperparameter tuning with RandomizedSearchCV,
evaluates performance, and saves the model and evaluation artifacts.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "wine_model.pkl"
FIGURE_DPI: int = 150
TARGET_COLUMN: str = "target"
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]

# Default hyperparameters (for initial cross-validation check)
N_ESTIMATORS: int = 100
MAX_DEPTH: int = 5
LEARNING_RATE: float = 0.1
RANDOM_STATE: int = 42

# Cross-validation constants
CV_FOLDS: int = 5

# Hyperparameter tuning constants
N_ITER_SEARCH: int = 20
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
}


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test splits from parquet files."""
    path = Path(output_dir)

    train_df = pl.read_parquet(path / "train.parquet")
    test_df = pl.read_parquet(path / "test.parquet")

    feature_columns = [c for c in train_df.columns if c != TARGET_COLUMN]

    x_train = train_df.select(feature_columns).to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy().astype(int)
    x_test = test_df.select(feature_columns).to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy().astype(int)

    logger.info(f"Loaded splits: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, y_train, y_test


def _get_feature_names(
    output_dir: str,
) -> list[str]:
    """Get feature column names from the training data."""
    path = Path(output_dir)
    train_df = pl.read_parquet(path / "train.parquet")
    return [c for c in train_df.columns if c != TARGET_COLUMN]


def _train_default_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBClassifier with default hyperparameters."""
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    model.fit(x_train, y_train)
    logger.info(
        f"Trained default XGBClassifier: n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}"
    )
    return model


def _run_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: XGBClassifier,
) -> dict:
    """Run stratified k-fold cross-validation."""
    logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation...")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    cv_results = {
        "cv_mean_accuracy": round(float(np.mean(scores)), 4),
        "cv_std_accuracy": round(float(np.std(scores)), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
    }

    logger.info(
        f"CV Accuracy: {cv_results['cv_mean_accuracy']} " f"(+/- {cv_results['cv_std_accuracy']})"
    )
    return cv_results


def _run_hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBClassifier, dict]:
    """Run RandomizedSearchCV for hyperparameter tuning."""
    logger.info(
        f"Starting hyperparameter tuning: {N_ITER_SEARCH} iterations, "
        f"{CV_FOLDS}-fold stratified CV"
    )

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )

    search.fit(x_train, y_train)

    best_params = {
        k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 4))
        for k, v in search.best_params_.items()
    }

    logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
    logger.info(f"Best parameters:\n{json.dumps(best_params, indent=2, default=str)}")

    return search.best_estimator_, best_params


def _save_tuning_results(
    best_params: dict,
    best_score: float,
    output_path: Path,
) -> None:
    """Save hyperparameter tuning results to JSON."""
    results = {
        "best_params": best_params,
        "best_cv_accuracy": round(float(best_score), 4),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
    }

    filepath = output_path / "tuning_results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Tuning results saved to {filepath}")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute classification evaluation metrics (macro averages)."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(report["macro avg"]["precision"]), 4),
        "recall_macro": round(float(report["macro avg"]["recall"]), 4),
        "f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
    }

    logger.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> dict:
    """Compute per-class precision, recall, F1 and save to JSON."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    per_class = {}
    for class_name in CLASS_NAMES:
        per_class[class_name] = {
            "precision": round(float(report[class_name]["precision"]), 4),
            "recall": round(float(report[class_name]["recall"]), 4),
            "f1_score": round(float(report[class_name]["f1-score"]), 4),
            "support": int(report[class_name]["support"]),
        }

    filepath = output_path / "per_class_metrics.json"
    filepath.write_text(json.dumps(per_class, indent=2, default=str))
    logger.info(f"Per-class metrics saved to {filepath}")
    logger.info(f"Per-class metrics:\n{json.dumps(per_class, indent=2, default=str)}")
    return per_class


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    filepath = output_path / "confusion_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Confusion matrix saved to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Generate a feature importance bar chart."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(len(importances)),
        importances[sorted_indices],
        align="center",
        alpha=0.8,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_indices],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("XGBoost Feature Importance")

    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")


def _save_model(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    """Save the trained model to disk."""
    filepath = output_path / MODEL_FILENAME
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def _save_evaluation_metrics(
    metrics: dict,
    cv_results: dict,
    best_params: dict,
    output_path: Path,
) -> None:
    """Save evaluation metrics to JSON."""
    combined = {
        "test_metrics": metrics,
        "cross_validation": cv_results,
        "tuned_best_params": best_params,
    }

    filepath = output_path / "evaluation_metrics.json"
    filepath.write_text(json.dumps(combined, indent=2, default=str))
    logger.info(f"Evaluation metrics saved to {filepath}")


def run_training_and_evaluation() -> None:
    """Run the full model training and evaluation pipeline."""
    start_time = time.time()
    logger.info("Starting model training and evaluation...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_splits(OUTPUT_DIR)
    feature_names = _get_feature_names(OUTPUT_DIR)

    # Train default model and run cross-validation
    default_model = _train_default_model(x_train, y_train)
    cv_results = _run_cross_validation(x_train, y_train, default_model)

    # Hyperparameter tuning
    tuned_model, best_params = _run_hyperparameter_tuning(x_train, y_train)
    _save_tuning_results(best_params, cv_results["cv_mean_accuracy"], output_path)

    # Evaluate tuned model on test set
    y_pred = tuned_model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred)
    _compute_per_class_metrics(y_test, y_pred, output_path)

    # Generate plots
    _plot_confusion_matrix(y_test, y_pred, output_path)
    _plot_feature_importance(tuned_model, feature_names, output_path)

    # Save model and metrics
    _save_model(tuned_model, output_path)
    _save_evaluation_metrics(metrics, cv_results, best_params, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Training and evaluation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model on Wine dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    run_training_and_evaluation()
