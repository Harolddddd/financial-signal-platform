from scripts.precompute_dashboard import _trade_window


def test_trade_window_open_when_current_signal_is_buy():
    signals = ["Hold", "Buy", "Buy"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    result = _trade_window(signals, dates, pos=2)
    assert result == {"status": "open", "open_date": "2026-01-02", "close_date": None}


def test_trade_window_open_run_starts_at_series_start():
    signals = ["Buy", "Buy", "Buy"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    result = _trade_window(signals, dates, pos=2)
    assert result == {"status": "open", "open_date": "2026-01-01", "close_date": None}


def test_trade_window_open_run_resets_after_prior_sell():
    signals = ["Buy", "Sell", "Buy"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    result = _trade_window(signals, dates, pos=2)
    assert result == {"status": "open", "open_date": "2026-01-03", "close_date": None}


def test_trade_window_closed_finds_prior_buy_run():
    signals = ["Hold", "Buy", "Buy", "Sell"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    result = _trade_window(signals, dates, pos=3)
    assert result == {"status": "closed", "open_date": "2026-01-02", "close_date": "2026-01-04"}


def test_trade_window_closed_close_date_is_first_non_buy_after_run():
    signals = ["Buy", "Buy", "Hold", "Hold"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    result = _trade_window(signals, dates, pos=3)
    assert result == {"status": "closed", "open_date": "2026-01-01", "close_date": "2026-01-03"}


def test_trade_window_none_when_no_buy_ever_occurred():
    signals = ["Hold", "Hold"]
    dates = ["2026-01-01", "2026-01-02"]
    result = _trade_window(signals, dates, pos=1)
    assert result == {"status": "none", "open_date": None, "close_date": None}


def test_merge_signals_replaces_recomputed_tickers_and_keeps_others():
    from scripts.precompute_dashboard import _merge_signals
    existing = [
        {"ticker": "AAPL", "strategy": "momentum", "signal": "Hold"},
        {"ticker": "MSFT", "strategy": "momentum", "signal": "Buy"},
    ]
    fresh = [{"ticker": "MSFT", "strategy": "momentum", "signal": "Sell"}]
    merged = _merge_signals(existing, fresh, universe=["AAPL", "MSFT"])
    assert merged == [
        {"ticker": "AAPL", "strategy": "momentum", "signal": "Hold"},
        {"ticker": "MSFT", "strategy": "momentum", "signal": "Sell"},
    ]


def test_merge_signals_drops_tickers_removed_from_universe():
    from scripts.precompute_dashboard import _merge_signals
    existing = [
        {"ticker": "AAPL", "strategy": "momentum", "signal": "Hold"},
        {"ticker": "QRVO", "strategy": "momentum", "signal": "Buy"},
    ]
    fresh = [{"ticker": "HOOD", "strategy": "momentum", "signal": "Buy"}]
    merged = _merge_signals(existing, fresh, universe=["AAPL", "HOOD"])
    assert [s["ticker"] for s in merged] == ["AAPL", "HOOD"]


def test_fitted_strategy_cache_round_trip(tmp_path):
    from scripts.precompute_dashboard import _load_fitted, _save_fitted
    _save_fitted({"momentum": {"w": 1}}, tmp_path)
    assert _load_fitted(tmp_path, ["momentum"]) == {"momentum": {"w": 1}}


def test_load_fitted_returns_none_when_any_strategy_missing(tmp_path):
    from scripts.precompute_dashboard import _load_fitted, _save_fitted
    _save_fitted({"momentum": {"w": 1}}, tmp_path)
    assert _load_fitted(tmp_path, ["momentum", "knn_classifier"]) is None


# --- Task 6: US default strategies, overlaid live signals ------------------

import json
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import polars as pl

import scripts.precompute_dashboard as pc
from src.strategies.base import PredictionResult, Strategy

US_DEFAULTS = ["extra_trees", "lda_strategy", "random_forest", "conformal_lgbm"]


def test_us_default_strategies_used():
    with patch.object(pc, "step_data_summary"), patch.object(pc, "step_leaderboard"), \
            patch.object(pc, "step_signals"), patch.object(pc, "step_backtests") as bt:
        pc.main("us", signals_only=False)
    assert list(bt.call_args.kwargs["names"]) == US_DEFAULTS


def test_us_default_strategies_passed_to_signals():
    with patch.object(pc, "step_data_summary"), patch.object(pc, "step_leaderboard"), \
            patch.object(pc, "step_signals") as sg:
        pc.main("us", signals_only=True)
    assert list(sg.call_args.kwargs["names"]) == US_DEFAULTS


def test_china_runs_all_strategies():
    with patch.object(pc, "step_data_summary"), patch.object(pc, "step_leaderboard"), \
            patch.object(pc, "step_signals"), patch.object(pc, "step_backtests") as bt, \
            patch.object(pc, "list_strategies", return_value=["a", "b"]):
        pc.main("china", signals_only=False)
    assert list(bt.call_args.kwargs["names"]) == ["a", "b"]


class _AlwaysBuy(Strategy):
    data_source = "features"

    def fit(self, df):
        self.cols = list(df.columns)

    def predict(self, df):
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.9] * n, index=df.index),
            signal=pd.Series(["Buy"] * n, index=df.index),
        )


def _frame(ticker: str, n: int, z: float, count: int) -> pl.DataFrame:
    d0 = date(2026, 1, 1)
    return pl.DataFrame({
        "time": [d0 + timedelta(days=i) for i in range(n)],
        "ticker": [ticker] * n,
        "close": [10.0] * n,
        "news_count_3d": [count] * n,
        "news_net_z60": [z] * n,
    })


def _run_signals(market: str, tmp_path):
    train = _frame("AAA", 150, 0.0, 0)
    live = [("NEG", _frame("NEG", 5, -3.0, 5)), ("OK", _frame("OK", 5, 0.0, 0))]
    with patch.object(pc, "load_training_data", return_value=train), \
            patch.object(pc, "list_strategies", return_value=["fake"]), \
            patch.object(pc, "load_strategy", side_effect=lambda n: _AlwaysBuy()), \
            patch.object(pc, "iter_live_features", return_value=iter(live)):
        pc.step_signals(market=market, cache_dir=tmp_path)
    return {s["ticker"]: s for s in json.loads((tmp_path / "signals.json").read_text())["signals"]}


def test_us_signals_include_overlay_reason(tmp_path):
    sigs = _run_signals("us", tmp_path)
    assert sigs["NEG"]["signal"] == "Hold"
    assert sigs["NEG"]["overlay_reason"] == "BUY suppressed: negative news"
    assert sigs["NEG"]["raw_confidence"] == 0.9
    assert sigs["OK"]["signal"] == "Buy"
    assert sigs["OK"]["overlay_reason"] == ""


class _NewsAwareBuy(_AlwaysBuy):
    extra_feature_cols = ("news_count_3d", "news_net_z60")


def test_us_news_aware_strategy_is_not_overlay_wrapped(tmp_path):
    train = _frame("AAA", 150, 0.0, 0)
    live = [("NEG", _frame("NEG", 5, -3.0, 5))]
    made = []

    def _load(name):
        made.append(_NewsAwareBuy())
        return made[-1]

    with patch.object(pc, "load_training_data", return_value=train),             patch.object(pc, "list_strategies", return_value=["fake"]),             patch.object(pc, "load_strategy", side_effect=_load),             patch.object(pc, "iter_live_features", return_value=iter(live)):
        pc.step_signals(market="us", cache_dir=tmp_path)
    sig = json.loads((tmp_path / "signals.json").read_text())["signals"][0]
    # No veto: the model itself sees the news columns, and the overlay does not apply.
    assert sig["signal"] == "Buy"
    assert sig["overlay_reason"] == ""
    assert {"news_count_3d", "news_net_z60"} <= set(made[0].cols)


def test_loader_defaults_missing_overlay_reason(tmp_path, monkeypatch):
    # Lives here because test_dashboard_data_loader.py is module-skipped.
    import dashboard.data_loader as dl
    (tmp_path / "leaderboard.json").write_text(json.dumps(
        {"grades": [{"model_name": "s1", "composite_score": 1.0}]}))
    (tmp_path / "signals.json").write_text(json.dumps({"signals": [
        {"ticker": "AAA", "date": "2026-01-02", "signal": "Buy", "confidence": 0.8,
         "entry_price": 10.0, "strategy": "s1"},
        {"ticker": "BBB", "date": "2026-01-02", "signal": "Hold", "confidence": 0.7,
         "entry_price": 10.0, "strategy": "s1", "overlay_reason": "risk-off: credit spread"},
    ]}))
    monkeypatch.setattr(dl, "get_cache_dir", lambda market="us": tmp_path)
    _, detail = dl.get_combined_ratings("us")
    assert detail["AAA"][0]["overlay_reason"] == ""
    assert detail["BBB"][0]["overlay_reason"] == "risk-off: credit spread"


def test_china_signals_have_uniform_schema(tmp_path):
    sigs = _run_signals("china", tmp_path)
    assert sigs["NEG"]["signal"] == "Buy"
    assert sigs["NEG"]["overlay_reason"] == ""
    assert sigs["NEG"]["raw_confidence"] == sigs["NEG"]["confidence"]


def test_strategies_arg_overrides_backtest_names_not_signal_names():
    with patch.object(pc, "step_data_summary"), patch.object(pc, "step_leaderboard"), \
            patch.object(pc, "step_signals") as sg, patch.object(pc, "step_backtests") as bt:
        pc.main("us", strategies=["extra_trees"])
    assert list(bt.call_args.kwargs["names"]) == ["extra_trees"]
    assert list(sg.call_args.kwargs["names"]) == US_DEFAULTS


def test_signal_train_days_limits_fit_history(tmp_path, monkeypatch):
    seen = {}

    def _load(feature_dir, start=None, **kw):
        seen["start"] = start
        return _frame("AAA", 150, 0.0, 0)

    monkeypatch.setenv("SIGNAL_TRAIN_DAYS", "400")
    with patch.object(pc, "load_training_data", side_effect=_load),             patch.object(pc, "list_strategies", return_value=["fake"]),             patch.object(pc, "load_strategy", side_effect=lambda n: _AlwaysBuy()),             patch.object(pc, "iter_live_features", return_value=iter([])):
        pc.step_signals(market="china", cache_dir=tmp_path)
    from datetime import datetime, timezone
    age = (datetime.now(timezone.utc) - seen["start"]).days
    assert 399 <= age <= 400


def test_signal_fit_uses_full_history_by_default(tmp_path, monkeypatch):
    seen = {}

    def _load(feature_dir, start=None, **kw):
        seen["start"] = start
        return _frame("AAA", 150, 0.0, 0)

    monkeypatch.delenv("SIGNAL_TRAIN_DAYS", raising=False)
    with patch.object(pc, "load_training_data", side_effect=_load),             patch.object(pc, "list_strategies", return_value=["fake"]),             patch.object(pc, "load_strategy", side_effect=lambda n: _AlwaysBuy()),             patch.object(pc, "iter_live_features", return_value=iter([])):
        pc.step_signals(market="china", cache_dir=tmp_path)
    assert seen["start"] is None
