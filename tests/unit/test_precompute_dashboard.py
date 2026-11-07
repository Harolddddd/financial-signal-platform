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
