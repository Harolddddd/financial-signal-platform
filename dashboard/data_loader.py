# dashboard/data_loader.py
from __future__ import annotations
import json
import logging
from pathlib import Path

import polars as pl

from src.backtesting.grader import Grade, ModelGrade, grade_model, build_leaderboard
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.walk_forward import FoldBacktestResult, WalkForwardBacktestResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.base import LiveSignal, Signal
from src.strategies.registry import list_strategies, load_strategy

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Cache read failed (%s): %s — falling back to live compute", path, exc)
    return None


def _metrics_from_dict(d: dict) -> BacktestMetrics:
    return BacktestMetrics(
        n_trades=d["n_trades"],
        win_rate=d["win_rate"],
        profit_factor=d["profit_factor"],
        total_return_pct=d["total_return_pct"],
        sharpe_ratio=d["sharpe_ratio"],
        max_drawdown_pct=d["max_drawdown_pct"],
        precision_buy=d["precision_buy"],
        recall_buy=d["recall_buy"],
        f1_buy=d["f1_buy"],
        accuracy=d["accuracy"],
    )


def _grade_from_dict(d: dict) -> ModelGrade:
    return ModelGrade(
        model_name=d["model_name"],
        grade=Grade(d["grade"]),
        composite_score=d["composite_score"],
        metrics=_metrics_from_dict(d["metrics"]),
    )


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Public API — each function tries the cache first, falls back to live compute
# ---------------------------------------------------------------------------

def get_data_summary(parquet_dir: Path) -> dict:
    cached = _load_cache(CACHE_DIR / "data_summary.json")
    if cached:
        return cached

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
    cached = _load_cache(CACHE_DIR / "leaderboard.json")
    if cached:
        return [_grade_from_dict(g) for g in cached["grades"]]

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
            avg_metrics = BacktestMetrics(
                n_trades=sum(f.n_trades for f in result.folds),
                win_rate=result.mean_win_rate,
                profit_factor=0.0,
                total_return_pct=0.0,
                sharpe_ratio=result.mean_sharpe,
                max_drawdown_pct=result.worst_drawdown,
                precision_buy=result.mean_precision_buy,
                recall_buy=0.0,
                f1_buy=0.0,
                accuracy=0.0,
            )
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
    cached = _load_cache(CACHE_DIR / f"backtest_{_safe(strategy_name)}.json")
    if cached:
        folds = [
            FoldBacktestResult(
                fold=f["fold"],
                train_start=f["train_start"],
                train_end=f["train_end"],
                test_start=f["test_start"],
                test_end=f["test_end"],
                metrics=_metrics_from_dict(f["metrics"]),
                n_trades=f["n_trades"],
            )
            for f in cached["folds"]
        ]
        wf = WalkForwardBacktestResult(
            folds=folds,
            mean_sharpe=cached["mean_sharpe"],
            mean_win_rate=cached["mean_win_rate"],
            mean_precision_buy=cached["mean_precision_buy"],
            worst_drawdown=cached["worst_drawdown"],
        )
        return wf, _grade_from_dict(cached["grade"])

    strategy = load_strategy(strategy_name)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest_strategy(
        df, strategy, ohlcv_cols, feature_cols,
        train_window_days=400, test_window_days=21, step_days=21,
    )
    avg_metrics = BacktestMetrics(
        n_trades=sum(f.n_trades for f in result.folds),
        win_rate=result.mean_win_rate,
        profit_factor=0.0,
        total_return_pct=0.0,
        sharpe_ratio=result.mean_sharpe,
        max_drawdown_pct=result.worst_drawdown,
        precision_buy=result.mean_precision_buy,
        recall_buy=0.0,
        f1_buy=0.0,
        accuracy=0.0,
    )
    grade = grade_model(strategy_name, avg_metrics)
    return result, grade


def get_live_signals(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[LiveSignal]:
    cached = _load_cache(CACHE_DIR / "signals.json")
    if cached:
        # Cache stores all signals at threshold=0; filter here so the slider works.
        return [
            LiveSignal(
                ticker=s["ticker"],
                date=s["date"],
                signal=Signal(s["signal"]),
                confidence=s["confidence"],
                entry_price=s["entry_price"],
                position_size=s["position_size"],
            )
            for s in cached["signals"]
            if s["signal"] == "Buy" and s["confidence"] >= confidence_threshold
        ]

    names = list_strategies()
    if not names:
        return []

    strategy = load_strategy(names[0])
    df = load_training_data(parquet_dir)
    df_pd = df.to_pandas()

    train_cutoff = max(0, len(df_pd) - 21)
    strategy.fit(df_pd.iloc[:train_cutoff])
    pred = strategy.predict(df_pd)

    df_with_pred = df.with_columns([
        pl.Series("_conf", pred.confidence.tolist()),
        pl.Series("_sig",  pred.signal.tolist()),
    ])
    latest = df_with_pred.sort("time").group_by("ticker", maintain_order=True).last()

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
