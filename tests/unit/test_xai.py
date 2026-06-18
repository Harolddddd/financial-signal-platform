import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.explainability.xai import explain_prediction, attach_explanations
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.logistic_regression import LogisticRegressionClassifier
from src.signals.signal_engine import Signal

_CLASSES = ["Buy", "Hold", "Sell"]
_FEATURE_COLS = [f"feat_{i}" for i in range(10)]


def _make_data(n: int = 150):
    X, y_int = make_classification(
        n_samples=n, n_features=10, n_classes=3,
        n_informative=5, n_redundant=2, random_state=42,
    )
    return X, np.array([_CLASSES[i] for i in y_int])


def test_explain_prediction_returns_dict_with_feature_names():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    assert isinstance(result, dict)
    assert set(result.keys()) == set(_FEATURE_COLS)


def test_explain_prediction_values_are_floats():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    for v in result.values():
        assert isinstance(v, float)


def test_explain_sorted_by_abs_value_descending():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    abs_vals = [abs(v) for v in result.values()]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_attach_explanations_fills_feature_explanation():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)

    from datetime import datetime, timezone, timedelta
    import polars as pl
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "time":   [base + timedelta(days=i) for i in range(10)],
        "ticker": ["AAPL"] * 10,
        "close":  [155.0] * 10,
        **{f"feat_{j}": X[:10, j].tolist() for j in range(10)},
    })

    signals = [Signal(
        ticker="AAPL", date=str(base), label="Buy",
        confidence=0.80, buy_probability=0.80,
        entry_price=155.0, position_size=0.80,
        feature_explanation={},
    )]
    result = attach_explanations(signals, clf, df, _FEATURE_COLS)
    assert len(result[0].feature_explanation) == 10
