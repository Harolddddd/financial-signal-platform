from __future__ import annotations
from pathlib import Path
import logging

from src.backtesting.grader import ModelGrade, grade_model, build_leaderboard
from src.backtesting.walk_forward import WalkForwardBacktestResult, walk_forward_backtest
from src.features.duckdb_client import load_training_data
from src.models.registry import list_models, load_model
from src.signals.signal_engine import Signal, generate_signals
from src.explainability.xai import attach_explanations

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
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
) -> list[ModelGrade]:
    records = list_models(registry_dir)
    if not records:
        return []

    df = load_training_data(parquet_dir)
    grades: list[ModelGrade] = []
    for record in records:
        try:
            model = load_model(record.model_name, registry_dir)
            result = walk_forward_backtest(
                df, model, feature_cols,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            avg_metrics = result.folds[-1].metrics
            grades.append(grade_model(record.model_name, avg_metrics))
        except Exception as e:
            logger.warning("Leaderboard skipping %s: %s", record.model_name, e)

    return build_leaderboard(grades)


def get_backtest_result(
    model_name: str,
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    model = load_model(model_name, registry_dir)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest(df, model, feature_cols)
    grade = grade_model(model_name, result.folds[-1].metrics)
    return result, grade


def get_live_signals(
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    records = list_models(registry_dir)
    if not records:
        return []
    model = load_model(records[0].model_name, registry_dir)
    df = load_training_data(parquet_dir)
    signals = generate_signals(model, df, feature_cols, confidence_threshold)
    return attach_explanations(signals, model, df, feature_cols)
