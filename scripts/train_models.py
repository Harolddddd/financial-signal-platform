from pathlib import Path
import logging

import numpy as np
import polars as pl

from dashboard.config import FEATURE_COLS, REGISTRY_DIR
from src.features.duckdb_client import load_training_data
from src.models.base_classifier import BaseClassifier
from src.models.evaluator import evaluate
from src.models.registry import save_model
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.xgboost_model import XGBoostClassifier
from src.models.zoo.lightgbm_model import LightGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = Path("data/features")
_TRAIN_RATIO = 0.8


def train_and_save(
    model: BaseClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
    registry_dir: Path,
) -> Path:
    clean = df.drop_nulls(subset=feature_cols + ["label"]).sort("time")
    if len(clean) == 0:
        raise ValueError("No training data after dropping nulls")

    split = int(len(clean) * _TRAIN_RATIO)
    train_df = clean[:split]
    test_df  = clean[split:]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test  = test_df.select(feature_cols).to_numpy()
    y_test  = test_df["label"].to_numpy()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluation = evaluate(y_test, y_pred)

    path = save_model(
        model=model,
        evaluation=evaluation,
        params=model.default_params,
        feature_cols=feature_cols,
        registry_dir=registry_dir,
    )
    logger.info(
        "Saved %s — acc=%.3f  prec_buy=%.3f  f1_macro=%.3f",
        model.name, evaluation.accuracy, evaluation.precision_buy, evaluation.f1_macro,
    )
    return path


def main() -> None:
    if not _FEATURE_DIR.exists() or not any(_FEATURE_DIR.glob("*.parquet")):
        raise FileNotFoundError(
            "No feature parquets found in data/features/. "
            "Run scripts/build_features.py first."
        )

    logger.info("Loading feature data from %s ...", _FEATURE_DIR)
    df = load_training_data(_FEATURE_DIR)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    models: list[BaseClassifier] = [
        RandomForestClassifier_(),
        XGBoostClassifier(),
        LightGBMClassifier(),
    ]

    for model in models:
        logger.info("Training %s ...", model.name)
        try:
            train_and_save(model, df, FEATURE_COLS, REGISTRY_DIR)
        except Exception as exc:
            logger.error("FAILED %s: %s", model.name, exc)

    print("\nTraining complete. Registry contents:")
    for p in sorted(REGISTRY_DIR.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
