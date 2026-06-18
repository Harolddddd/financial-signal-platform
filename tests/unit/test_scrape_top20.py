from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest


_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
_END   = datetime(2020, 12, 31, tzinfo=timezone.utc)

_SAMPLE_DF = pl.DataFrame({
    "time":         ["2020-01-02 00:00:00+0000", "2020-01-03 00:00:00+0000"],
    "ticker":       ["AAPL", "AAPL"],
    "open":         [295.0, 300.0],
    "high":         [300.0, 305.0],
    "low":          [290.0, 295.0],
    "close":        [298.0, 302.0],
    "volume":       [1_000_000, 1_100_000],
    "adj_close":    [298.0, 302.0],
    "dividends":    [0.0, 0.0],
    "stock_splits": [0.0, 0.0],
})


def test_tickers_list_has_20_stocks():
    from scripts.scrape_top20 import TICKERS
    assert len(TICKERS) == 20
    assert "AAPL" in TICKERS
    assert "MSFT" in TICKERS
    assert "NVDA" in TICKERS


def test_benchmark_tickers():
    from scripts.scrape_top20 import BENCHMARK_TICKERS
    assert "SPY" in BENCHMARK_TICKERS
    assert "^VIX" in BENCHMARK_TICKERS


def test_scrape_ticker_writes_parquet_and_returns_row_count(tmp_path):
    with patch("scripts.scrape_top20.fetch_ohlcv", return_value=_SAMPLE_DF):
        from scripts.scrape_top20 import scrape_ticker
        rows, err = scrape_ticker("AAPL", _START, _END, tmp_path)
    assert err is None
    assert rows == 2
    assert (tmp_path / "AAPL.parquet").exists()
    loaded = pl.read_parquet(tmp_path / "AAPL.parquet")
    assert len(loaded) == 2


def test_scrape_ticker_returns_error_on_fetch_failure(tmp_path):
    with patch("scripts.scrape_top20.fetch_ohlcv", side_effect=ValueError("no data")):
        from scripts.scrape_top20 import scrape_ticker
        rows, err = scrape_ticker("FAKE", _START, _END, tmp_path)
    assert rows == 0
    assert "no data" in err
    assert not (tmp_path / "FAKE.parquet").exists()
