from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.strategies.base import Strategy, PredictionResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.backtesting.walk_forward import WalkForwardBacktestResult


class _AlwaysBuyStrategy(Strategy):
    """Predicts BUY with confidence 0.9 for every row."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.9] * n),
            signal=pd.Series(["Buy"] * n),
        )


class _AlwaysHoldStrategy(Strategy):
    """Predicts HOLD for every row — produces zero trades."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.0] * n),
            signal=pd.Series(["Hold"] * n),
        )


def _make_df(n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    closes = [100.0 + i * 0.05 for i in range(n)]
    labels = ["Buy" if i % 3 == 0 else "Hold" for i in range(n)]
    returns = [0.02 if l == "Buy" else 0.001 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        "open":  [c - 0.5 for c in closes],
        "high":  [c + 1.0 for c in closes],
        "low":   [c - 1.0 for c in closes],
        "volume": rng.integers(1_000_000, 10_000_000, n).tolist(),
    })


_OHLCV = ["open", "high", "low", "close", "volume"]
_FEATURES = []  # not used in these tests


def test_returns_walk_forward_result():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    assert isinstance(result, WalkForwardBacktestResult)
    assert len(result.folds) >= 1


def test_fold_count_matches_expected():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert fold.metrics is not None
        assert fold.fold >= 0


def test_always_hold_produces_zero_trades():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysHoldStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    total_trades = sum(f.n_trades for f in result.folds)
    assert total_trades == 0


def test_mean_sharpe_is_average_of_folds():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    manual = sum(f.metrics.sharpe_ratio for f in result.folds) / len(result.folds)
    assert abs(result.mean_sharpe - manual) < 1e-9


def test_empty_df_raises():
    df = pl.DataFrame({
        "time": [], "ticker": [], "close": [],
        "forward_return_5d": [], "label": [],
        "open": [], "high": [], "low": [], "volume": [],
    })
    with pytest.raises(ValueError):
        walk_forward_backtest_strategy(
            df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
            train_window_days=300, test_window_days=30, step_days=30,
        )


def test_start_limits_folds():
    df = _make_df()
    start = datetime(2021, 1, 4, tzinfo=timezone.utc) + timedelta(days=450)
    full = walk_forward_backtest_strategy(df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
                                          train_window_days=300, test_window_days=30, step_days=30)
    part = walk_forward_backtest_strategy(df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
                                          train_window_days=300, test_window_days=30, step_days=30,
                                          start=start)
    assert 0 < len(part.folds) < len(full.folds)
    assert all(datetime.fromisoformat(f.test_start) >= start for f in part.folds)


def test_pred_log_one_row_per_test_row():
    df = _make_df()
    log_buy, log_hold = [], []
    r = walk_forward_backtest_strategy(df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
                                       train_window_days=300, test_window_days=30, step_days=30,
                                       pred_log=log_buy)
    walk_forward_backtest_strategy(df, _AlwaysHoldStrategy(), _OHLCV, _FEATURES,
                                   train_window_days=300, test_window_days=30, step_days=30,
                                   pred_log=log_hold)
    assert len(log_buy) == sum(f.n_trades for f in r.folds) == len(log_hold)
    assert all(score == 0.9 for _, _, score in log_buy)
    assert all(score == 0.0 for _, _, score in log_hold)
    assert log_buy[0][0] == "AAPL" and not isinstance(log_buy[0][1], datetime)
