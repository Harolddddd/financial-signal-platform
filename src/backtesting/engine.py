from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl

from src.models.base_classifier import BaseClassifier
from src.backtesting.metrics import Trade, BacktestMetrics, compute_metrics

_BUY_IDX = 0  # "Buy" < "Hold" < "Sell" alphabetically — index 0 in all zoo models


@dataclass
class BacktestResult:
    trades: list[Trade]
    metrics: BacktestMetrics
    equity_curve: list[float]


def run_backtest(
    model: BaseClassifier,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    confidence_threshold: float = 0.5,
) -> BacktestResult:
    X = test_df.select(feature_cols).to_numpy()
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    trades: list[Trade] = []
    for i, row in enumerate(test_df.iter_rows(named=True)):
        if y_pred[i] != "Buy":
            continue
        buy_prob = float(proba[i][_BUY_IDX])
        if buy_prob < confidence_threshold:
            continue

        entry = float(row["close"])
        ret = float(row["forward_return_5d"])
        trades.append(Trade(
            entry_date=str(row["time"]),
            exit_date=str(row["time"]),
            entry_price=entry,
            exit_price=entry * (1 + ret),
            predicted_label="Buy",
            actual_label=str(row["label"]),
            return_pct=ret,
            confidence=buy_prob,
        ))

    metrics = compute_metrics(trades)
    equity_curve = list(np.cumprod(1 + np.array([t.return_pct for t in trades]))) if trades else []

    return BacktestResult(trades=trades, metrics=metrics, equity_curve=equity_curve)
