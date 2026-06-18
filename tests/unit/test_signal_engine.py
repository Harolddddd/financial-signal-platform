from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import polars as pl
import pytest

from src.signals.signal_engine import Signal, generate_signals

_FEATURE_COLS = ["f1", "f2", "f3"]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_feature_df(labels: list[str]) -> pl.DataFrame:
    n = len(labels)
    rng = np.random.default_rng(1)
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    returns = [0.03 if l == "Buy" else 0.00 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             [155.0 + i for i in range(n)],
        "forward_return_5d": returns,
        "label":             labels,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
    })


def test_generate_signals_returns_only_buy_signals():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 5 + ["Hold"] * 5)
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert s.label == "Buy"


def test_generate_signals_filters_by_confidence():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 10)
    low  = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    high = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=1.0)
    assert len(low) >= len(high)


def test_signal_position_size_equals_buy_probability():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 10)
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert abs(s.position_size - s.buy_probability) < 1e-9


def test_signal_is_dataclass():
    s = Signal(
        ticker="AAPL", date="2024-06-10", label="Buy",
        confidence=0.82, buy_probability=0.82,
        entry_price=155.0, position_size=0.82,
        feature_explanation={},
    )
    assert s.label == "Buy"
