from datetime import datetime, timezone

import polars as pl
import pytest


def test_aux_tickers_by_market():
    from scripts.refresh_data import _AUX_TICKERS_BY_MARKET
    assert _AUX_TICKERS_BY_MARKET["us"] == ["SPY", "^VIX"]
    assert _AUX_TICKERS_BY_MARKET["china"] == ["000300.SS"]


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
