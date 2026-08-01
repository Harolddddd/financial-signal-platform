"""Retry just the LSTM leg of scripts/train_new_models.py after the tz-comparison fix."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

import polars as pl

from config.markets import get_market
from dashboard.config import FEATURE_COLS, REGISTRY_DIR
from src.features.duckdb_client import load_training_data
from src.models.zoo.lstm_model import LSTMClassifier
from scripts.train_new_models import train_and_save, _LSTM_WINDOW_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = get_market("us").data_root / "features"


def main() -> None:
    df = load_training_data(_FEATURE_DIR)
    clean_full = df.drop_nulls(subset=FEATURE_COLS + ["label"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LSTM_WINDOW_DAYS)
    lstm_df = clean_full.filter(pl.col("time").dt.date() >= cutoff.date())
    logger.info("LSTM window: %d rows since %s", len(lstm_df), cutoff.date())
    train_and_save(LSTMClassifier(), lstm_df, FEATURE_COLS, REGISTRY_DIR)


if __name__ == "__main__":
    main()
