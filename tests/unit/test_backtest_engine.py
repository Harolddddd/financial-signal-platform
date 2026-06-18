from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from src.backtesting.engine import run_backtest, BacktestResult
from src.models.zoo.random_forest import RandomForestClassifier_
from sklearn.datasets import make_classification

_FEATURE_COLS = [f"f{i}" for i in range(10)]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_test_df(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    labels = [_CLASSES[i % 3] for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    closes = [150.0 + i * 0.1 for i in range(n)]
    features = {f"f{j}": rng.standard_normal(n).tolist() for j in range(10)}
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "close": closes,
        "forward_return_5d": returns,
        "label": labels,
        **features,
    })


def _trained_rf(df: pl.DataFrame) -> RandomForestClassifier_:
    X = df.select(_FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    clf = RandomForestClassifier_(n_estimators=20)
    clf.fit(X, y)
    return clf


def test_run_backtest_returns_backtest_result():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    assert isinstance(result, BacktestResult)


def test_all_trades_are_buy_signals():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    for trade in result.trades:
        assert trade.predicted_label == "Buy"


def test_trades_respect_confidence_threshold():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result_low  = run_backtest(clf, df[60:], _FEATURE_COLS, confidence_threshold=0.0)
    result_high = run_backtest(clf, df[60:], _FEATURE_COLS, confidence_threshold=1.0)
    assert len(result_low.trades) >= len(result_high.trades)


def test_equity_curve_length_matches_trades():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    assert len(result.equity_curve) == len(result.trades)


def test_buy_class_at_index_zero():
    """Verify the alphabetical-sort invariant: Buy=0, Hold=1, Sell=2."""
    clf = RandomForestClassifier_(n_estimators=5)
    X, y_int = make_classification(
        n_samples=120, n_features=10, n_classes=3,
        n_informative=5, n_redundant=2, random_state=1,
    )
    y = np.array([_CLASSES[i] for i in y_int])
    clf.fit(X, y)
    proba = clf.predict_proba(X[:1])
    assert proba.shape == (1, 3)
    assert clf._model.classes_[0] == "Buy"
