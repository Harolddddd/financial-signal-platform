# dashboard/data_loader.py
from __future__ import annotations
from pathlib import Path
import logging

import polars as pl

from src.backtesting.grader import ModelGrade, grade_model, build_leaderboard
from src.backtesting.walk_forward import WalkForwardBacktestResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.base import LiveSignal, Signal
from src.strategies.registry import list_strategies, load_strategy

logger = logging.getLogger(__name__)


def get_data_summary(parquet_dir: Path) -> dict:
    df = load_training_data(parquet_dir)
    tickers = df["ticker"].unique().to_list() if "ticker" in df.columns else []
    time_min = str(df["time"].min()) if "time" in df.columns else "N/A"
    time_max = str(df["time"].max()) if "time" in df.columns else "N/A"
    return {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": sorted(tickers),
        "date_range_start": time_min,
        "date_range_end": time_max,
    }


def get_leaderboard(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
) -> list[ModelGrade]:
    names = list_strategies()
    if not names:
        return []

    df = load_training_data(parquet_dir)
    grades: list[ModelGrade] = []
    for name in names:
        try:
            strategy = load_strategy(name)
            result = walk_forward_backtest_strategy(
                df, strategy, ohlcv_cols, feature_cols,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            avg_metrics = result.folds[-1].metrics
            grades.append(grade_model(name, avg_metrics))
        except Exception as e:
            logger.warning("Leaderboard skipping %s: %s", name, e)

    return build_leaderboard(grades)


def get_backtest_result(
    strategy_name: str,
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    strategy = load_strategy(strategy_name)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest_strategy(
        df, strategy, ohlcv_cols, feature_cols,
        train_window_days=400, test_window_days=21, step_days=21,
    )
    grade = grade_model(strategy_name, result.folds[-1].metrics)
    return result, grade


def get_live_signals(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[LiveSignal]:
    names = list_strategies()
    if not names:
        return []

    strategy = load_strategy(names[0])
    df = load_training_data(parquet_dir)
    df_pd = df.to_pandas()

    strategy.fit(df_pd)
    pred = strategy.predict(df_pd)

    df_with_pred = df.with_columns([
        pl.Series("_conf", pred.confidence.tolist()),
        pl.Series("_sig",  pred.signal.tolist()),
    ])
    latest = df_with_pred.sort("time").group_by("ticker").last()

    live_signals: list[LiveSignal] = []
    for row in latest.iter_rows(named=True):
        conf = float(row["_conf"])
        sig  = str(row["_sig"])
        if sig == "Buy" and conf >= confidence_threshold:
            live_signals.append(LiveSignal(
                ticker=str(row["ticker"]),
                date=str(row["time"]),
                signal=Signal.BUY,
                confidence=conf,
                entry_price=float(row["close"]),
                position_size=conf,
            ))
    return live_signals
