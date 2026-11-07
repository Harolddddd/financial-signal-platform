import pandas as pd
import pytest

from src.features.news_macro_features import OVERLAY_COLS
from src.signals.overlay import OverlayConfig, OverlayStrategy, apply_overlay
from src.strategies.base import PredictionResult, Strategy

CFG = OverlayConfig()


def _ctx(n=1, **kw):
    base = {c: 0.0 for c in OVERLAY_COLS}
    base["days_since_fomc"] = 999
    base.update(kw)
    return pd.DataFrame({k: [v] * n for k, v in base.items()}, index=range(10, 10 + n))


def _sc(sig="Buy", conf=0.8, n=1):
    idx = range(10, 10 + n)
    return pd.Series([sig] * n, index=idx), pd.Series([conf] * n, index=idx)


def test_veto_turns_buy_to_hold():
    s, c = _sc()
    sig, conf, reason = apply_overlay(s, c, _ctx(news_count_3d=5, news_net_z60=-2.0), CFG)
    assert sig.iloc[0] == "Hold"
    assert reason.iloc[0] == "BUY suppressed: negative news"


def test_veto_needs_min_news():
    s, c = _sc()
    sig, _, reason = apply_overlay(s, c, _ctx(news_count_3d=2, news_net_z60=-3.0), CFG)
    assert sig.iloc[0] == "Buy"
    assert reason.iloc[0] == ""


def test_risk_off_two_flags():
    s, c = _sc(conf=0.8)
    ctx = _ctx(macro_dgs2_chg_5d=0.5, macro_hy_spread_z20=3.0)
    sig, conf, reason = apply_overlay(s, c, ctx, CFG)
    assert conf.iloc[0] == pytest.approx(0.8 * 0.85**2)
    assert sig.iloc[0] == "Hold"
    assert "risk-off: 2y yield jump" in reason.iloc[0]
    assert "risk-off: credit spread" in reason.iloc[0]


def test_hawkish_fomc_flag():
    s, c = _sc(conf=0.9)
    ctx = _ctx(fomc_tone_change=0.3, days_since_fomc=2)
    sig, conf, reason = apply_overlay(s, c, ctx, CFG)
    assert reason.iloc[0] == "risk-off: hawkish FOMC"
    assert conf.iloc[0] == pytest.approx(0.9 * 0.85)


def test_boost_capped_at_one():
    s, c = _sc(conf=0.95)
    sig, conf, reason = apply_overlay(s, c, _ctx(news_count_3d=5, news_net_z60=2.0), CFG)
    assert conf.iloc[0] == 1.0
    assert reason.iloc[0] == "confidence boosted: positive news"


def test_sell_untouched_by_news_and_no_reason():
    s, c = _sc(sig="Sell", conf=0.7)
    sig, conf, reason = apply_overlay(s, c, _ctx(news_count_3d=5, news_net_z60=-3.0), CFG)
    assert sig.iloc[0] == "Sell" and conf.iloc[0] == 0.7 and reason.iloc[0] == ""


def test_missing_ctx_columns_is_noop():
    s, c = _sc(n=3)
    sig, conf, reason = apply_overlay(s, c, pd.DataFrame(index=s.index), CFG)
    assert sig.equals(s) and conf.equals(c)
    assert (reason == "").all()


class _Fake(Strategy):
    data_source = "features"

    def fit(self, df):
        self.fit_cols = list(df.columns)
        bad = [c for c in df.columns if c.startswith(("news_", "macro_", "fomc_"))]
        assert not bad, bad

    def predict(self, df):
        bad = [c for c in df.columns if c.startswith(("news_", "macro_", "fomc_"))]
        assert not bad, bad
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.8] * n, index=df.index),
            signal=pd.Series(["Buy"] * n, index=df.index),
        )


def test_wrapper_inner_never_sees_overlay_cols():
    df = _ctx(n=3, news_count_3d=5, news_net_z60=-2.0)
    df.insert(0, "rsi", 50.0)
    w = OverlayStrategy(_Fake())
    assert w.data_source == "features"
    w.fit(df)
    assert w.inner.fit_cols == ["rsi"]
    res = w.predict(df)
    assert list(res.signal.index) == [10, 11, 12]
    assert (res.signal == "Hold").all()
    assert (w.last_raw_confidence == 0.8).all()
    assert (w.last_reason == "BUY suppressed: negative news").all()
    assert list(w.last_reason.index) == [10, 11, 12]
