from __future__ import annotations
from datetime import timedelta
import logging

import polars as pl

from src.strategies.base import Strategy
from src.backtesting.metrics import Trade, compute_metrics
from src.backtesting.walk_forward import FoldBacktestResult, WalkForwardBacktestResult

logger = logging.getLogger(__name__)

_REQUIRED = {"time", "close", "label", "forward_return_5d"}
_REQUIRED_PASS_COLS = {"time", "ticker", "close", "label", "forward_return_5d"}


def _select_cols(
    df: pl.DataFrame,
    strategy: Strategy,
    ohlcv_cols: list[str],
    feature_cols: list[str],
) -> pl.DataFrame:
    if strategy.data_source == "ohlcv":
        keep = list(_REQUIRED_PASS_COLS | set(ohlcv_cols))
    else:
        keep = list(_REQUIRED_PASS_COLS | set(feature_cols))
    available = [c for c in keep if c in df.columns]
    return df.select(available)


def walk_forward_backtest_strategy(
    df: pl.DataFrame,
    strategy: Strategy,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    label_col: str = "label",
    train_window_days: int = 400,
    test_window_days: int = 21,
    step_days: int = 21,
    min_train_samples: int = 100,
    confidence_threshold: float = 0.5,
) -> WalkForwardBacktestResult:
    drop_cols = list(_REQUIRED)
    df_clean = df.drop_nulls(subset=[c for c in drop_cols if c in df.columns]).sort("time")
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
            train_pd = _select_cols(train_df, strategy, ohlcv_cols, feature_cols).to_pandas()
            test_pd  = _select_cols(test_df,  strategy, ohlcv_cols, feature_cols).to_pandas()
            strategy.fit(train_pd)
            result = strategy.predict(test_pd)
            trades = _build_trades(test_df, result, confidence_threshold)
            metrics = compute_metrics(trades)
            folds.append(FoldBacktestResult(
                fold=fold_idx,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                metrics=metrics,
                n_trades=len(trades),
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


def _build_trades(
    test_df: pl.DataFrame,
    result,
    confidence_threshold: float,
) -> list[Trade]:
    trades: list[Trade] = []
    conf_arr   = result.confidence.to_numpy()
    signal_arr = result.signal.to_numpy()

    for i, row in enumerate(test_df.iter_rows(named=True)):
        if i >= len(signal_arr):
            break
        if str(signal_arr[i]) != "Buy":
            continue
        conf = float(conf_arr[i])
        if conf < confidence_threshold:
            continue
        entry = float(row["close"])
        ret   = float(row["forward_return_5d"])
        trades.append(Trade(
            entry_date=str(row["time"]),
            exit_date=str(row["time"]),
            entry_price=entry,
            exit_price=entry * (1 + ret),
            predicted_label="Buy",
            actual_label=str(row["label"]),
            return_pct=ret,
            confidence=conf,
        ))
    return trades
