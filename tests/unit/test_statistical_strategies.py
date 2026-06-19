# tests/unit/test_statistical_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.strategies.statistical.logistic import LogisticStrategy
from src.strategies.statistical.linear import LinearStrategy
from src.strategies.base import PredictionResult

_VALID_SIGNALS = {"Buy", "Hold", "Sell"}
_META_COLS = {"time", "ticker", "label", "forward_return_5d"}


def _make_feature_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.standard_normal(n),
        "label": ["Buy" if i % 3 == 0 else "Hold" for i in range(n)],
        "forward_return_5d": rng.uniform(-0.05, 0.05, n),
    })
    return df


# --- LogisticStrategy ---

def test_logistic_fit_and_predict():
    df = _make_feature_df()
    s = LogisticStrategy(C=1.0, max_iter=200)
    s.fit(df)
    result = s.predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_logistic_confidence_in_range():
    df = _make_feature_df()
    s = LogisticStrategy()
    s.fit(df)
    result = s.predict(df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_logistic_signals_are_valid():
    df = _make_feature_df()
    s = LogisticStrategy()
    s.fit(df)
    result = s.predict(df)
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_logistic_predict_before_fit_raises():
    s = LogisticStrategy()
    df = _make_feature_df(10)
    with pytest.raises((AttributeError, ValueError)):
        s.predict(df)


def test_logistic_excludes_meta_columns():
    df = _make_feature_df()
    df["ticker"] = "AAPL"
    s = LogisticStrategy()
    s.fit(df)
    assert "ticker" not in s._feature_cols
    assert "label" not in s._feature_cols


# --- LinearStrategy ---

def test_linear_fit_and_predict():
    df = _make_feature_df()
    s = LinearStrategy(buy_threshold=0.005)
    s.fit(df)
    result = s.predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_linear_confidence_in_range():
    df = _make_feature_df()
    s = LinearStrategy()
    s.fit(df)
    result = s.predict(df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_linear_signals_are_valid():
    df = _make_feature_df()
    s = LinearStrategy()
    s.fit(df)
    result = s.predict(df)
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_linear_predict_before_fit_raises():
    s = LinearStrategy()
    df = _make_feature_df(10)
    with pytest.raises((AttributeError, ValueError)):
        s.predict(df)


def test_linear_buy_when_predicted_return_above_threshold():
    rng = np.random.default_rng(0)
    n = 200
    # f1 perfectly predicts return — high f1 → high return
    forward = rng.uniform(0.0, 0.02, n)
    df = pd.DataFrame({
        "f1": forward * 100,  # perfect linear predictor
        "forward_return_5d": forward,
        "label": ["Buy" if r >= 0.005 else "Hold" for r in forward],
    })
    s = LinearStrategy(buy_threshold=0.005)
    s.fit(df)
    result = s.predict(df)
    # Rows with high f1 should be BUY
    high_f1_mask = df["f1"] > 0.5
    assert (result.signal[high_f1_mask] == "Buy").any()
