from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import polars as pl
import pytest
from src.ingestion.collector import collect_ohlcv

_START = datetime(2024, 1, 2, tzinfo=timezone.utc)
_END   = datetime(2024, 1, 10, tzinfo=timezone.utc)

def _df():
    return pl.DataFrame({"time": [_START], "ticker": ["AAPL"],
                         "open": [150.0], "high": [155.0], "low": [149.0],
                         "close": [153.0], "volume": [1000000],
                         "adj_close": [153.0], "dividends": [0.0], "stock_splits": [1.0]})

def test_collect_ohlcv_returns_yfinance_on_success():
    with patch("src.ingestion.collector.fetch_ohlcv", return_value=_df()) as mock_yf:
        result = collect_ohlcv("AAPL", _START, _END)
        mock_yf.assert_called_once()
        assert "close" in result.columns

def test_collect_ohlcv_falls_back_to_av():
    with patch("src.ingestion.collector.fetch_ohlcv", side_effect=Exception("yf down")), \
         patch("src.ingestion.collector.fetch_ohlcv_av", return_value=_df()) as mock_av:
        result = collect_ohlcv("AAPL", _START, _END)
        mock_av.assert_called_once()

def test_collect_ohlcv_falls_back_to_fmp():
    with patch("src.ingestion.collector.fetch_ohlcv",    side_effect=Exception("yf down")), \
         patch("src.ingestion.collector.fetch_ohlcv_av", side_effect=Exception("av down")), \
         patch("src.ingestion.collector.fetch_ohlcv_fmp", return_value=_df()) as mock_fmp:
        result = collect_ohlcv("AAPL", _START, _END)
        mock_fmp.assert_called_once()

def test_collect_ohlcv_raises_runtime_error_when_all_fail():
    with patch("src.ingestion.collector.fetch_ohlcv",    side_effect=Exception("yf down")), \
         patch("src.ingestion.collector.fetch_ohlcv_av", side_effect=Exception("av down")), \
         patch("src.ingestion.collector.fetch_ohlcv_fmp", side_effect=Exception("fmp down")):
        with pytest.raises(RuntimeError, match="All data sources failed"):
            collect_ohlcv("AAPL", _START, _END)
