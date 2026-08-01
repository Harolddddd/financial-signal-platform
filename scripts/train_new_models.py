"""Full initial training for the model-zoo entries that have never been
trained (no prior registry version to build on incrementally):

  - logistic_regression, naive_bayes, mlp — full dataset (they scale fine)
  - svm      — RBF-kernel SVC doesn't scale past ~tens of thousands of rows,
               so it trains on a random subsample instead of all 3.48M rows
  - lstm     — its sequence builder materializes a dense (n, seq_len, features)
               array with no windowed/strided view, so full data risks an
               OOM; it trains on a recent chronological window instead
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS, REGISTRY_DIR
from src.features.duckdb_client import load_training_data
from src.models.base_classifier import BaseClassifier
from src.models.evaluator import evaluate
from src.models.registry import save_model
from src.models.zoo.logistic_regression import LogisticRegressionClassifier
from src.models.zoo.naive_bayes import NaiveBayesClassifier
from src.models.zoo.mlp_model import MLPClassifier_
from src.models.zoo.lstm_model import LSTMClassifier
from src.models.zoo.svm_model import SVMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = get_market("us").data_root / "features"
_TRAIN_RATIO = 0.8
_SVM_SAMPLE_SIZE = 30_000
_LSTM_WINDOW_DAYS = 420


def train_and_save(model: BaseClassifier, df: pl.DataFrame, feature_cols: list[str], registry_dir: Path) -> Path:
    clean = df.drop_nulls(subset=feature_cols + ["label"]).sort(["ticker", "time"])
    if len(clean) == 0:
        raise ValueError("No training data after dropping nulls")

    split = int(len(clean) * _TRAIN_RATIO)
    train_df = clean[:split]
    test_df = clean[split:]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["label"].to_numpy()

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
        "Saved %s (n=%d) — acc=%.3f prec_buy=%.3f f1_macro=%.3f",
        model.name, len(clean), evaluation.accuracy, evaluation.precision_buy, evaluation.f1_macro,
    )
    return path


def main() -> None:
    if not _FEATURE_DIR.exists() or not any(_FEATURE_DIR.glob("*.parquet")):
        raise FileNotFoundError("No feature parquets found in markets/us/data/features/.")

    logger.info("Loading full feature data from %s ...", _FEATURE_DIR)
    df = load_training_data(_FEATURE_DIR)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    clean_full = df.drop_nulls(subset=FEATURE_COLS + ["label"])

    full_data_models: list[BaseClassifier] = [
        LogisticRegressionClassifier(),
        NaiveBayesClassifier(),
        MLPClassifier_(),
    ]
    for model in full_data_models:
        logger.info("Training %s on full dataset ...", model.name)
        try:
            train_and_save(model, df, FEATURE_COLS, REGISTRY_DIR)
        except Exception:
            logger.exception("FAILED %s", model.name)

    logger.info("Sampling %d rows for SVM (RBF kernel doesn't scale to full data) ...", _SVM_SAMPLE_SIZE)
    try:
        svm_df = clean_full.sample(n=min(_SVM_SAMPLE_SIZE, len(clean_full)), seed=42, shuffle=True)
        train_and_save(SVMClassifier(), svm_df, FEATURE_COLS, REGISTRY_DIR)
    except Exception:
        logger.exception("FAILED svm")

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LSTM_WINDOW_DAYS)
    logger.info("Windowing last %d days for LSTM (sequence builder isn't memory-efficient at full scale) ...", _LSTM_WINDOW_DAYS)
    try:
        # Some parquet files carry inconsistent tz metadata on `time`
        # (e.g. America/Los_Angeles vs UTC) — compare on plain dates to
        # sidestep tz-aware comparison errors across mixed-tz columns.
        lstm_df = clean_full.filter(pl.col("time").dt.date() >= cutoff.date())
        logger.info("LSTM window: %d rows since %s", len(lstm_df), cutoff.date())
        train_and_save(LSTMClassifier(), lstm_df, FEATURE_COLS, REGISTRY_DIR)
    except Exception:
        logger.exception("FAILED lstm")

    print("\nFull training complete. Registry contents:")
    for p in sorted(REGISTRY_DIR.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
