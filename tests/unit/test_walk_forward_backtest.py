from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from src.backtesting.walk_forward import walk_forward_backtest, WalkForwardBacktestResult
from src.models.zoo.random_forest import RandomForestClassifier_

_FEATURE_COLS = ["f1", "f2", "f3"]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_df(n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    closes = [100.0 + i * 0.05 for i in range(n)]
    labels = [_CLASSES[i % 3] for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
    })


def test_walk_forward_backtest_returns_result():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    assert isinstance(result, WalkForwardBacktestResult)
    assert len(result.folds) >= 1


def test_each_fold_has_metrics():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert fold.metrics is not None
        assert fold.fold >= 0


def test_mean_sharpe_is_average_of_folds():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    manual = sum(f.metrics.sharpe_ratio for f in result.folds) / len(result.folds)
    assert abs(result.mean_sharpe - manual) < 1e-9


def test_worst_drawdown_is_max_across_folds():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    worst = max(f.metrics.max_drawdown_pct for f in result.folds)
    assert abs(result.worst_drawdown - worst) < 1e-9
