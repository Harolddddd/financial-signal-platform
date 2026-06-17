from datetime import datetime, timezone
import polars as pl
import pytest
import responses as resp_mock

from src.ingestion.fmp_client import fetch_ohlcv_fmp

_FMP_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL"
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 5, tzinfo=timezone.utc)

_SAMPLE = {
    "symbol": "AAPL",
    "historical": [
        {
            "date": "2024-01-03",
            "open": 151.0, "high": 156.0, "low": 150.0,
            "close": 154.0, "adjClose": 154.0, "volume": 1_100_000,
        }
    ],
}


@resp_mock.activate
def test_fetch_ohlcv_fmp_returns_polars_dataframe():
    resp_mock.add(resp_mock.GET, _FMP_URL, json=_SAMPLE, status=200)
    df = fetch_ohlcv_fmp("AAPL", _START, _END)
    assert isinstance(df, pl.DataFrame)
    assert df["ticker"][0] == "AAPL"
    assert len(df) == 1
    # Verify all 10 columns are present
    expected_columns = {"time", "ticker", "open", "high", "low", "close", "volume", "adj_close", "dividends", "stock_splits"}
    assert set(df.columns) == expected_columns
    # Verify values
    assert df["open"][0] == 151.0
    assert df["high"][0] == 156.0
    assert df["low"][0] == 150.0
    assert df["close"][0] == 154.0
    assert df["adj_close"][0] == 154.0
    assert df["volume"][0] == 1_100_000
    assert df["stock_splits"][0] == 0.0


@resp_mock.activate
def test_fetch_ohlcv_fmp_raises_on_empty():
    resp_mock.add(resp_mock.GET, _FMP_URL, json={"historical": []}, status=200)
    with pytest.raises(ValueError, match="FMP returned no data"):
        fetch_ohlcv_fmp("AAPL", _START, _END)


@resp_mock.activate
def test_fetch_ohlcv_fmp_multiple_rows_sorted():
    """Verify that multiple rows are returned and sorted by time."""
    multi_sample = {
        "symbol": "AAPL",
        "historical": [
            {
                "date": "2024-01-03",
                "open": 151.0, "high": 156.0, "low": 150.0,
                "close": 154.0, "adjClose": 154.0, "volume": 1_100_000,
            },
            {
                "date": "2024-01-02",
                "open": 150.0, "high": 155.0, "low": 149.0,
                "close": 153.0, "adjClose": 153.0, "volume": 1_000_000,
            },
        ],
    }
    resp_mock.add(resp_mock.GET, _FMP_URL, json=multi_sample, status=200)
    df = fetch_ohlcv_fmp("AAPL", _START, _END)
    assert len(df) == 2
    # Verify sorted by time
    assert df["time"][0] < df["time"][1]
