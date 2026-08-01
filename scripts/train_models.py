from pathlib import Path
import logging

import numpy as np
import polars as pl

from config.markets import MARKETS, get_market
from dashboard.ui_config import FEATURE_COLS
from src.features.duckdb_client import load_training_data
from src.models.base_classifier import BaseClassifier
from src.models.evaluator import evaluate
from src.models.registry import save_model
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.xgboost_model import XGBoostClassifier
from src.models.zoo.lightgbm_model import LightGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = get_market("us").data_root / "features"
_TRAIN_RATIO = 0.8


def _market_paths(market: str) -> tuple[Path, Path]:
    market_cfg = get_market(market)
    return market_cfg.data_root / "features", market_cfg.data_root / "registry"


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


def main(market: str = "us") -> None:
    feature_dir, registry_dir = _market_paths(market)

    if not feature_dir.exists() or not any(feature_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No feature parquets found in {feature_dir}/. "
            "Run scripts/build_features.py first."
        )

    logger.info("Loading feature data from %s ...", feature_dir)
    df = load_training_data(feature_dir)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    registry_dir.mkdir(parents=True, exist_ok=True)

    models: list[BaseClassifier] = [
        RandomForestClassifier_(),
        XGBoostClassifier(),
        LightGBMClassifier(),
    ]

    for model in models:
        logger.info("Training %s ...", model.name)
        try:
            train_and_save(model, df, FEATURE_COLS, registry_dir)
        except Exception as exc:
            logger.error("FAILED %s: %s", model.name, exc)

    print("\nTraining complete. Registry contents:")
    for p in sorted(registry_dir.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
