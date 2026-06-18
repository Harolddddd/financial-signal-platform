from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import logging

import polars as pl

from src.models.base_classifier import BaseClassifier
from src.backtesting.engine import run_backtest
from src.backtesting.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class FoldBacktestResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: BacktestMetrics
    n_trades: int


@dataclass
class WalkForwardBacktestResult:
    folds: list[FoldBacktestResult]
    mean_sharpe: float
    mean_win_rate: float
    mean_precision_buy: float
    worst_drawdown: float


def walk_forward_backtest(
    df: pl.DataFrame,
    model: BaseClassifier,
    feature_cols: list[str],
    label_col: str = "label",
    train_window_days: int = 500,
    test_window_days: int = 21,
    step_days: int = 21,
    min_train_samples: int = 100,
    confidence_threshold: float = 0.5,
) -> WalkForwardBacktestResult:
    df_clean = df.drop_nulls(subset=feature_cols + [label_col, "forward_return_5d"]).sort("time")
    times = df_clean["time"].to_list()
    if not times:
        raise ValueError("DataFrame is empty after dropping nulls")

    t_end = times[-1]
    folds: list[FoldBacktestResult] = []
    fold_idx = 0
    cursor = times[0] + timedelta(days=train_window_days)

    while cursor + timedelta(days=test_window_days) <= t_end + timedelta(days=1):
        train_start = cursor - timedelta(days=train_window_days)
        train_end   = cursor - timedelta(days=1)
        test_start  = cursor
        test_end    = cursor + timedelta(days=test_window_days - 1)

        train_df = df_clean.filter(
            (pl.col("time") >= train_start) & (pl.col("time") <= train_end)
        )
        test_df = df_clean.filter(
            (pl.col("time") >= test_start) & (pl.col("time") <= test_end)
        )

        if len(train_df) < min_train_samples or len(test_df) == 0:
            cursor += timedelta(days=step_days)
            continue

        try:
            X_train = train_df.select(feature_cols).to_numpy()
            y_train = train_df[label_col].to_numpy()
            model.fit(X_train, y_train)
            result = run_backtest(model, test_df, feature_cols, confidence_threshold)
            folds.append(FoldBacktestResult(
                fold=fold_idx,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                metrics=result.metrics,
                n_trades=len(result.trades),
            ))
            fold_idx += 1
        except Exception as e:
            logger.warning("Walk-forward fold %d failed: %s", fold_idx, e)

        cursor += timedelta(days=step_days)

    if not folds:
        raise ValueError("No valid folds — check data range and window sizes")

    n = len(folds)
    return WalkForwardBacktestResult(
        folds=folds,
        mean_sharpe=sum(f.metrics.sharpe_ratio for f in folds) / n,
        mean_win_rate=sum(f.metrics.win_rate for f in folds) / n,
        mean_precision_buy=sum(f.metrics.precision_buy for f in folds) / n,
        worst_drawdown=max(f.metrics.max_drawdown_pct for f in folds),
    )
