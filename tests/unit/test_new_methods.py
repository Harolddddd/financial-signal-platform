from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.news_macro_features import NEWS_COLS, overlay_cols
from src.strategies.registry import list_strategies, load_strategy

_PRICE = ["rsi_14", "macd", "atr_14", "vix_level"]


def _df(market="us", n_days=260, tickers=("A", "B", "C", "D", "E", "F"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for tk in tickers:
        for d in range(n_days):
            r = {c: rng.random() for c in _PRICE}
            for c in overlay_cols(market):
                r[c] = rng.standard_normal()
            fr = rng.standard_normal() * 0.02
            r.update(time=t0 + pd.Timedelta(days=d), ticker=tk, close=100 + d * 0.1,
                     forward_return_5d=fr, label="Buy" if fr > 0.01 else ("Sell" if fr < -0.01 else "Hold"))
            rows.append(r)
    return pd.DataFrame(rows)


def _check(res, n):
    assert len(res.confidence) == n and len(res.signal) == n
    assert ((res.confidence >= 0) & (res.confidence <= 1)).all()
    assert set(res.signal.unique()) <= {"Buy", "Hold", "Sell"}


@pytest.mark.parametrize("name,market", [
    ("cross_sectional_ranker_news", "us"), ("news_surprise", "us"), ("regime_gated", "us"),
    ("cross_sectional_ranker_news_cn", "china"), ("news_surprise_cn", "china"),
    ("regime_gated_cn", "china"),
])
def test_fit_predict_valid(name, market):
    s = load_strategy(name)
    df = _df(market)
    s.fit(df)
    test = df[df["ticker"] == "A"].tail(21).reset_index(drop=True)
    _check(s.predict(test), 21)
    assert set(getattr(s, "extra_feature_cols", ())) <= set(overlay_cols(market))
    assert getattr(s, "extra_feature_cols", ())


def test_registry_markets():
    us, cn = set(list_strategies("us")), set(list_strategies("china"))
    for base in ("cross_sectional_ranker_news", "news_surprise", "regime_gated", "stacked_live"):
        assert base in us and base not in cn
        assert f"{base}_cn" in cn and f"{base}_cn" not in us


def test_news_surprise_uses_news_cols():
    s = load_strategy("news_surprise")
    assert set(NEWS_COLS) <= set(s.extra_feature_cols)


def test_regime_gated_routes_by_vol():
    s = load_strategy("regime_gated")
    df = _df()
    s.fit(df)
    assert s._threshold == pytest.approx(df["vix_level"].median())


def test_stacked_live_excludes_itself_and_ohlcv_bases(monkeypatch):
    from src.strategies.statistical import stacked_live as sl
    monkeypatch.setattr(sl, "live_set", lambda cache_dir, market: ["stacked_live", "bollinger_bounce", "random_forest", "regime_gated"])
    s = sl.StackedLiveStrategy(market="us")
    names = s.base_names()
    assert "stacked_live" not in names and "bollinger_bounce" not in names
    assert names == ["random_forest", "regime_gated"]


def test_stacked_live_fit_predict(monkeypatch):
    from src.strategies.statistical import stacked_live as sl
    monkeypatch.setattr(sl, "live_set", lambda cache_dir, market: ["regime_gated", "news_surprise"])
    s = sl.StackedLiveStrategy(market="us")
    df = _df()
    s.fit(df)
    _check(s.predict(df[df["ticker"] == "B"].tail(10).reset_index(drop=True)), 10)


def test_limit_days_china():
    from src.backtesting.limit_filter import limit_days
    t = [datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(days=i) for i in range(4)]
    df = pl.DataFrame({"ticker": ["X"] * 4 + ["Y"] * 4, "time": t + t,
                       "close": [10.0, 11.0, 11.5, 10.35, 10.0, 10.5, 10.5, 10.5]})
    assert limit_days(df) == {("X", t[1]), ("X", t[3])}


def test_walk_forward_blocks_limit_entries():
    from src.backtesting.strategy_runner import walk_forward_backtest_strategy
    from tests.unit.test_strategy_runner import _AlwaysBuyStrategy, _make_df, _OHLCV
    df = _make_df()
    blocked = set(zip(df["ticker"].to_list(), df["time"].to_list()))
    r = walk_forward_backtest_strategy(df, _AlwaysBuyStrategy(), _OHLCV, [],
                                       train_window_days=300, test_window_days=30, step_days=30,
                                       blocked_entries=blocked)
    assert sum(f.n_trades for f in r.folds) == 0


def test_blocked_entries_only_for_china():
    from src.backtesting.limit_filter import blocked_entries
    t = [datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(days=i) for i in range(2)]
    df = pl.DataFrame({"ticker": ["X", "X"], "time": t, "close": [10.0, 11.0]})
    assert blocked_entries("us", df) == set()
    assert blocked_entries("china", df) == {("X", t[1])}


def test_live_signal_hold_on_limit_day():
    from src.backtesting.limit_filter import is_limit_move
    assert is_limit_move(10.0, 11.0) and not is_limit_move(10.0, 10.5) and not is_limit_move(None, 11.0)


def test_limit_threshold_by_board():
    from src.backtesting.limit_filter import limit_days
    t = [datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(days=i) for i in range(2)]
    df = pl.DataFrame({"ticker": ["300001.SZ"] * 2 + ["688001.SS"] * 2 + ["600000.SS"] * 2,
                       "time": t * 3, "close": [10.0, 11.5, 10.0, 12.0, 10.0, 11.0]})
    # ChiNext +15% tradeable, STAR +20% at limit, main board +10% at limit
    assert limit_days(df) == {("688001.SS", t[1]), ("600000.SS", t[1])}
