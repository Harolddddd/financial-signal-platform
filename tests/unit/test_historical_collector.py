from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import polars as pl
import pandas as pd
import pytest

from src.ingestion.historical_collector import (
    fetch_ohlcv,
    fetch_corporate_actions,
    fetch_fundamentals,
)


def _make_yf_history_df():
    idx = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03"],
        name="Date",
        tz="America/New_York",
    )
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0],
            "High": [155.0, 156.0],
            "Low": [149.0, 150.0],
            "Close": [153.0, 154.0],
            "Adj Close": [152.5, 153.5],
            "Volume": [1_000_000, 1_100_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=idx,
    )


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_ohlcv_returns_polars_dataframe(mock_ticker):
    mock_ticker.return_value.history.return_value = _make_yf_history_df()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)

    df = fetch_ohlcv("AAPL", start, end)

    assert isinstance(df, pl.DataFrame)
    assert "ticker" in df.columns
    assert df["ticker"][0] == "AAPL"
    assert len(df) == 2


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_ohlcv_raises_on_empty_response(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No data returned"):
        fetch_ohlcv(
            "FAKE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_corporate_actions_returns_expected_keys(mock_ticker):
    inst = MagicMock()
    inst.dividends = pd.Series([], dtype=float)
    inst.splits = pd.Series([], dtype=float)
    inst.earnings_dates = pd.DataFrame()
    mock_ticker.return_value = inst

    result = fetch_corporate_actions("AAPL")
    assert set(result.keys()) == {"dividends", "splits", "earnings_dates"}


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_fundamentals_parses_info(mock_ticker):
    mock_ticker.return_value.info = {
        "trailingPE": 28.5,
        "trailingEps": 6.43,
        "bookValue": 4.5,
        "marketCap": 3_000_000_000_000,
    }
    result = fetch_fundamentals("AAPL")
    assert result["pe_ratio"] == 28.5
    assert result["eps"] == 6.43


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_fundamentals_handles_missing_keys(mock_ticker):
    mock_ticker.return_value.info = {}
    result = fetch_fundamentals("UNKNOWN")
    assert result["pe_ratio"] is None
    assert result["market_cap"] is None
