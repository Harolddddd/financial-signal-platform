import numpy as np
import pandas as pd
import polars as pl

from src.backtesting.strategy_runner import _select_cols
from src.features.news_macro_features import OVERLAY_COLS
from src.strategies.registry import load_strategy
from src.strategies.statistical.pooled_lgbm import PooledLGBMStrategy

_PRICE = ["rsi_14", "macd", "atr_14"]


def _df(n=300, with_overlay=True, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.random(n) for c in _PRICE}
    if with_overlay:
        for c in OVERLAY_COLS:
            data[c] = rng.random(n)
        data["news_net_3d"][: n // 2] = np.nan  # news history is short: early rows are NaN
    data["time"] = pd.date_range("2025-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_uses_news_and_macro_columns_by_default():
    s = PooledLGBMStrategy()
    s.fit(_df())
    assert set(OVERLAY_COLS) <= set(s._feature_cols)
    assert set(_PRICE) <= set(s._feature_cols)
    assert s.extra_feature_cols == tuple(OVERLAY_COLS)


def test_price_only_variant_ignores_overlay_columns():
    s = PooledLGBMStrategy(use_news_macro=False)
    s.fit(_df())
    assert not set(OVERLAY_COLS) & set(s._feature_cols)
    assert s.extra_feature_cols == ()


def test_predict_handles_nan_news_and_returns_valid_output():
    df = _df()
    s = PooledLGBMStrategy()
    s.fit(df)
    out = s.predict(df)
    assert len(out.signal) == len(df)
    assert set(out.signal.unique()) <= {"Buy", "Hold"}
    assert out.confidence.between(0.0, 1.0).all()


def test_works_without_overlay_columns_present():
    # China feature files have no news/macro columns; the model must still train.
    df = _df(with_overlay=False)
    s = PooledLGBMStrategy()
    s.fit(df)
    assert s._feature_cols == _PRICE
    assert len(s.predict(df).signal) == len(df)


def test_select_cols_adds_strategy_extra_feature_cols():
    df = pl.from_pandas(_df())
    picked = _select_cols(df, PooledLGBMStrategy(), [], _PRICE)
    assert set(OVERLAY_COLS) <= set(picked.columns)
    plain = _select_cols(df, PooledLGBMStrategy(use_news_macro=False), [], _PRICE)
    assert not set(OVERLAY_COLS) & set(plain.columns)


def test_registered_as_pooled_lgbm_news():
    assert isinstance(load_strategy("pooled_lgbm_news"), PooledLGBMStrategy)


def test_pooled_model_is_us_only_in_registry():
    from src.strategies.registry import list_strategies
    assert "pooled_lgbm_news" in list_strategies("us")
    assert "pooled_lgbm_news" not in list_strategies("china")
    assert "pooled_lgbm_news" in list_strategies()
    assert "extra_trees" in list_strategies("china")


def test_pooled_lgbm_china_declares_cn_cols():
    from src.features.news_macro_features import overlay_cols
    assert PooledLGBMStrategy(market="china").extra_feature_cols == tuple(overlay_cols("china"))


def test_pooled_lgbm_news_cn_registered_for_china_only():
    from src.strategies.registry import list_strategies
    assert "pooled_lgbm_news_cn" in list_strategies("china")
    assert "pooled_lgbm_news_cn" not in list_strategies("us")
    assert load_strategy("pooled_lgbm_news_cn").market == "china"
