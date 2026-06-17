from datetime import datetime, timezone
import polars as pl
import pytest
import responses as resp_mock

from src.ingestion.alpha_vantage_client import fetch_ohlcv_av

_AV_URL = "https://www.alphavantage.co/query"
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 5, tzinfo=timezone.utc)

_SAMPLE = {
    "Time Series (Daily)": {
        "2024-01-03": {
            "1. open": "151.00", "2. high": "156.00", "3. low": "150.00",
            "4. close": "154.00", "5. adjusted close": "154.00",
            "6. volume": "1100000", "7. dividend amount": "0.00",
            "8. split coefficient": "1.00",
        },
        "2024-01-02": {
            "1. open": "150.00", "2. high": "155.00", "3. low": "149.00",
            "4. close": "153.00", "5. adjusted close": "153.00",
            "6. volume": "1000000", "7. dividend amount": "0.00",
            "8. split coefficient": "1.00",
        },
    }
}


@resp_mock.activate
def test_fetch_ohlcv_av_returns_polars_dataframe():
    resp_mock.add(resp_mock.GET, _AV_URL, json=_SAMPLE, status=200)
    df = fetch_ohlcv_av("AAPL", _START, _END)
    assert isinstance(df, pl.DataFrame)
    assert "ticker" in df.columns
    assert len(df) == 2


@resp_mock.activate
def test_fetch_ohlcv_av_raises_on_rate_limit():
    resp_mock.add(resp_mock.GET, _AV_URL, json={"Note": "API rate limit"}, status=200)
    with pytest.raises(ValueError, match="Alpha Vantage returned no"):
        fetch_ohlcv_av("AAPL", _START, _END)
