"""Feature engineering for the Wine dataset.

Creates derived features, handles infinite values, scales numeric
columns, and splits the data into stratified training and test sets.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
INPUT_FILE: str = "output/wine_data.parquet"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "target"
ACID_COLUMN: str = "malic_acid"
COLOR_COLUMN: str = "color_intensity"
DILUTION_COLUMN: str = "od280/od315_of_diluted_wines"
PHENOLS_COLUMN: str = "total_phenols"
FLAVANOIDS_COLUMN: str = "flavanoids"
ALCOHOL_COLUMN: str = "alcohol"


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_data(
    input_file: str,
) -> pl.DataFrame:
    """Load the wine data from parquet."""
    df = pl.read_parquet(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    return df


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create 3 new derived features from existing columns."""
    df = df.with_columns(
        [
            (pl.col(ALCOHOL_COLUMN) / pl.col(ACID_COLUMN)).alias("alcohol_to_acid_ratio"),
            (pl.col(COLOR_COLUMN) / pl.col(DILUTION_COLUMN)).alias("color_intensity_normalized"),
            (pl.col(PHENOLS_COLUMN) * pl.col(FLAVANOIDS_COLUMN)).alias(
                "phenol_flavanoid_interaction"
            ),
        ]
    )

    logger.info(f"Created 3 derived features. New shape: {df.shape}")
    logger.info(f"Columns: {json.dumps(df.columns, indent=2, default=str)}")
    return df


def _handle_infinite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace infinite values with column medians."""
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32]:
            median_val = float(df[col].filter(df[col].is_finite()).median())
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(median_val)
                .otherwise(pl.col(col))
                .alias(col)
            )

    logger.info("Replaced infinite values with column medians")
    return df


def _scale_features(
    df: pl.DataFrame,
    target_column: str,
) -> pl.DataFrame:
    """Scale numeric features using StandardScaler."""
    feature_columns = [c for c in df.columns if c != target_column]

    scaler = StandardScaler()
    feature_values = df.select(feature_columns).to_numpy()
    scaled_values = scaler.fit_transform(feature_values)

    scaled_df = pl.DataFrame({col: scaled_values[:, i] for i, col in enumerate(feature_columns)})
    scaled_df = scaled_df.with_columns(df[target_column])

    logger.info(f"Scaled {len(feature_columns)} features using StandardScaler")
    return scaled_df


def _split_data(
    df: pl.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform stratified train/test split and return combined DataFrames."""
    feature_columns = [c for c in df.columns if c != target_column]

    x_data = df.select(feature_columns).to_numpy()
    y_data = df[target_column].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    train_df = pl.DataFrame(
        {col: x_train[:, i] for i, col in enumerate(feature_columns)}
    ).with_columns(pl.Series(target_column, y_train))

    test_df = pl.DataFrame(
        {col: x_test[:, i] for i, col in enumerate(feature_columns)}
    ).with_columns(pl.Series(target_column, y_test))

    logger.info(f"Train set: {train_df.shape[0]} samples")
    logger.info(f"Test set: {test_df.shape[0]} samples")

    return train_df, test_df


def _save_splits(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Save train/test splits to parquet files."""
    train_df.write_parquet(output_path / "train.parquet")
    test_df.write_parquet(output_path / "test.parquet")
    logger.info(f"Saved train/test splits to {output_path}")


def run_feature_engineering() -> None:
    """Run the full feature engineering pipeline."""
    start_time = time.time()
    logger.info("Starting feature engineering...")

    output_path = _ensure_output_dir(OUTPUT_DIR)

    df = _load_data(INPUT_FILE)
    df = _create_derived_features(df)
    df = _handle_infinite_values(df)
    scaled_df = _scale_features(df, TARGET_COLUMN)

    train_df, test_df = _split_data(
        scaled_df,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )

    _save_splits(train_df, test_df, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering on Wine dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    run_feature_engineering()
