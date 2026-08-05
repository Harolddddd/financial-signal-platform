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
