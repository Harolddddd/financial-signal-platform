import numpy as np
import pandas as pd
import pytest

from src.backtesting.strategy_runner import calibration_table
from src.strategies.statistical.conformal_lgbm import ConformalLGBMStrategy
from src.strategies.statistical.cross_sectional_ranker import CrossSectionalRankerStrategy
from src.strategies.statistical.meta_labeling import MetaLabelingStrategy
from src.strategies.statistical.regime_consensus import RegimeConsensusStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]

_STRATEGIES = [
    MetaLabelingStrategy,
    ConformalLGBMStrategy,
    CrossSectionalRankerStrategy,
    RegimeConsensusStrategy,
]


def _panel(n_days: int, n_tickers: int, signal: bool, seed: int = 42) -> pd.DataFrame:
    """Multi-ticker daily panel. With signal=True, rsi_14 > 0.8 means a +5%
    forward return (Buy) 90% of the time; otherwise labels are pure noise."""
    rng = np.random.default_rng(seed)
    n = n_days * n_tickers
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = np.repeat(pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC"), n_tickers)
    data["ticker"] = np.tile([f"T{i}" for i in range(n_tickers)], n_days)
    data["close"] = 100 + rng.random(n)
    fwd = rng.standard_normal(n) * 0.02
    if signal:
        strong = data["rsi_14"] > 0.8
        fwd = np.where(strong & (rng.random(n) < 0.9), 0.05, np.where(strong, -0.05, fwd))
    data["forward_return_5d"] = fwd
    data["label"] = np.where(fwd > 0.02, "Buy", np.where(fwd < -0.02, "Sell", "Hold"))
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def noise_df():
    return _panel(120, 10, signal=False)


@pytest.fixture(scope="module")
def signal_df():
    return _panel(200, 20, signal=True)


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_output_shape_values_and_range(cls, noise_df):
    s = cls()
    s.fit(noise_df)
    result = s.predict(noise_df)
    assert len(result.signal) == len(result.confidence) == len(noise_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})
    assert result.confidence.between(0.0, 1.0).all()
    assert not result.confidence.isna().any()


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_predict_before_fit_raises(cls):
    with pytest.raises(ValueError, match="fit"):
        cls().predict(pd.DataFrame({"rsi_14": [0.5]}))


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_data_source_is_features(cls):
    assert cls.data_source == "features"


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_buy_only_at_or_above_threshold(cls, signal_df):
    s = cls()
    s.fit(signal_df)
    result = s.predict(signal_df)
    buys = result.signal == "Buy"
    assert (result.confidence[buys] >= s.buy_threshold).all()
    assert (result.confidence[~buys] < s.buy_threshold).all()


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_learns_real_signal_with_high_precision(cls, signal_df):
    """Out-of-sample: trained on one panel, scored on a fresh one with the
    same rule. Buys must exist and hit far above the ~20% base rate."""
    s = cls()
    s.fit(signal_df)
    test = _panel(60, 20, signal=True, seed=7)
    result = s.predict(test)
    buys = (result.signal == "Buy").to_numpy()
    assert buys.sum() > 20
    assert (test["label"].to_numpy()[buys] == "Buy").mean() > 0.7


@pytest.mark.parametrize("cls", _STRATEGIES)
def test_abstains_on_pure_noise(cls, noise_df):
    """No real signal: a calibrated model should almost never claim >= 50%
    hit probability when the true Buy rate is ~16%."""
    s = cls()
    s.fit(noise_df)
    result = s.predict(_panel(60, 10, signal=False, seed=7))
    assert (result.signal == "Buy").mean() < 0.02


def test_calibration_table_buckets_and_hit_rates():
    hits = [(0.55, True), (0.58, False), (0.85, True), (0.95, True), (1.0, False)]
    table = calibration_table(hits)
    by_lo = {b["lo"]: b for b in table}
    assert [b["lo"] for b in table] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert by_lo[0.5]["n"] == 2 and by_lo[0.5]["precision"] == 0.5
    assert by_lo[0.6]["n"] == 0 and by_lo[0.6]["precision"] is None
    assert by_lo[0.8]["n"] == 1 and by_lo[0.8]["precision"] == 1.0
    assert by_lo[0.9]["n"] == 2 and by_lo[0.9]["precision"] == 0.5
