from datetime import datetime, timezone

import polars as pl


def test_aux_tickers_by_market():
    from scripts.refresh_data import _AUX_TICKERS_BY_MARKET
    assert _AUX_TICKERS_BY_MARKET["us"] == ["SPY", "^VIX"]
    assert _AUX_TICKERS_BY_MARKET["china"] == ["510300.SS"]


def test_refresh_raw_new_ticker_writes_to_given_raw_dir(tmp_path, monkeypatch):
    from scripts.refresh_data import _refresh_raw

    sample = pl.DataFrame({
        "time":         [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker":       ["600519.SS"],
        "open":         [1700.0], "high": [1705.0], "low": [1690.0], "close": [1700.0],
        "volume":       [100_000],
        "adj_close":    [1700.0], "dividends": [0.0], "stock_splits": [0.0],
    })
    monkeypatch.setattr("scripts.refresh_data.fetch_ohlcv", lambda *a, **k: sample)

    _refresh_raw("600519.SS", datetime(2024, 1, 3, tzinfo=timezone.utc), tmp_path)

    assert (tmp_path / "600519.SS.parquet").exists()
    written = pl.read_parquet(tmp_path / "600519.SS.parquet")
    assert len(written) == 1


def _bar(day, close):
    return pl.DataFrame({
        "time": [datetime(2026, 11, day, tzinfo=timezone.utc)], "ticker": ["AAPL"],
        "open": [close], "high": [close], "low": [close], "close": [close], "volume": [1],
        "adj_close": [close], "dividends": [0.0], "stock_splits": [0.0],
    })


def test_refresh_raw_fetches_todays_bar_and_refreshes_last_stored_day(tmp_path, monkeypatch):
    """Yahoo's end date is exclusive: asking for end=today never returns today's bar.
    The last stored day is fetched again so a bar saved mid-session gets its final close."""
    from scripts.refresh_data import _refresh_raw

    _bar(6, 100.0).write_parquet(tmp_path / "AAPL.parquet")  # Friday, possibly partial
    calls = []

    def fake(ticker, start, end, *a, **k):
        calls.append((start.date().isoformat(), end.date().isoformat()))
        return pl.concat([_bar(6, 101.0), _bar(9, 102.0)])

    monkeypatch.setattr("scripts.refresh_data.fetch_ohlcv", fake)
    _refresh_raw("AAPL", datetime(2026, 11, 9, 23, 30, tzinfo=timezone.utc), tmp_path)

    assert calls == [("2026-11-06", "2026-11-10")]
    out = pl.read_parquet(tmp_path / "AAPL.parquet")
    assert out["close"].to_list() == [101.0, 102.0]


def test_refresh_raw_fetches_when_data_ends_yesterday(tmp_path, monkeypatch):
    from scripts.refresh_data import _refresh_raw

    _bar(9, 100.0).write_parquet(tmp_path / "AAPL.parquet")
    calls = []
    monkeypatch.setattr("scripts.refresh_data.fetch_ohlcv",
                        lambda t, s, e, *a, **k: calls.append(e.date().isoformat()) or _bar(10, 1.0))
    _refresh_raw("AAPL", datetime(2026, 11, 10, 23, 30, tzinfo=timezone.utc), tmp_path)
    assert calls == ["2026-11-11"]
    assert len(pl.read_parquet(tmp_path / "AAPL.parquet")) == 2


def test_last_complete_day_skips_a_session_still_trading():
    from scripts.refresh_data import _last_complete_day
    # China closes 15:00 Beijing = 07:00 UTC; US 16:00 New York = 21:00 UTC (winter).
    assert _last_complete_day("china", datetime(2026, 11, 10, 5, 0, tzinfo=timezone.utc)).date().isoformat() == "2026-11-09"
    assert _last_complete_day("china", datetime(2026, 11, 10, 10, 30, tzinfo=timezone.utc)).date().isoformat() == "2026-11-10"
    assert _last_complete_day("us", datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)).date().isoformat() == "2026-11-09"
    assert _last_complete_day("us", datetime(2026, 11, 10, 23, 30, tzinfo=timezone.utc)).date().isoformat() == "2026-11-10"
