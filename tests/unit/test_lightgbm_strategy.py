import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.lightgbm_strategy import LightGBMStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = LightGBMStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert LightGBMStrategy.data_source == "features"


def test_handles_nan_is_true():
    assert LightGBMStrategy.handles_nan is True
