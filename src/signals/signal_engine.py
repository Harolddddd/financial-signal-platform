from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import polars as pl

from src.models.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

_BUY_IDX = 0  # alphabetical: Buy=0, Hold=1, Sell=2


@dataclass
class Signal:
    ticker: str
    date: str
    label: str
    confidence: float
    buy_probability: float
    entry_price: float
    position_size: float
    feature_explanation: dict[str, float] = field(default_factory=dict)


def generate_signals(
    model: BaseClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    latest = df.sort("time").group_by("ticker").tail(1)
    signals: list[Signal] = []

    for row in latest.iter_rows(named=True):
        x = np.array([[row[c] for c in feature_cols]])
        y_pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        buy_prob = float(proba[_BUY_IDX])

        if y_pred != "Buy" or buy_prob < confidence_threshold:
            continue

        signals.append(Signal(
            ticker=str(row["ticker"]),
            date=str(row["time"]),
            label="Buy",
            confidence=buy_prob,
            buy_probability=buy_prob,
            entry_price=float(row["close"]),
            position_size=buy_prob,
            feature_explanation={},
        ))

    return signals


def generate_all_signals(
    parquet_dir: Path,
    registry_dir: Path,
    feature_cols: list[str],
    model_name: str | None = None,
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    from src.features.duckdb_client import load_training_data
    from src.models.registry import load_model, list_models

    df = load_training_data(parquet_dir)

    if model_name:
        model = load_model(model_name, registry_dir)
    else:
        records = list_models(registry_dir)
        if not records:
            raise RuntimeError(f"No models in registry at {registry_dir}")
        model = load_model(records[0].model_name, registry_dir)

    return generate_signals(model, df, feature_cols, confidence_threshold)
