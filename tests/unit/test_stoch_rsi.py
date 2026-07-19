import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.stoch_rsi import StochRSI


@pytest.fixture
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    spread = np.abs(rng.normal(0, 0.5, n)) + 0.5
    high = close + spread
    low = close - spread
    open_ = low + rng.random(n) * (high - low)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def test_predict_correct_length(ohlcv_df):
    s = StochRSI()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = StochRSI()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = StochRSI()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert StochRSI.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = StochRSI()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = StochRSI()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
