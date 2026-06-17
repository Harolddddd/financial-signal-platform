from __future__ import annotations
from datetime import datetime
import requests
import polars as pl
from config.settings import settings

def fetch_ohlcv_av(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Fetch daily OHLCV from Alpha Vantage TIME_SERIES_DAILY_ADJUSTED."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": settings.ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    series = data.get("Time Series (Daily)", {})
    if not series:
        raise ValueError(f"Alpha Vantage returned no data for {ticker}")
    rows = []
    for date_str, vals in series.items():
        dt = datetime.fromisoformat(date_str).replace(tzinfo=None)
        if not (start.date() <= dt.date() <= end.date()):
            continue
        rows.append({
            "time": dt,
            "ticker": ticker,
            "open": float(vals["1. open"]),
            "high": float(vals["2. high"]),
            "low": float(vals["3. low"]),
            "close": float(vals["4. close"]),
            "volume": int(vals["6. volume"]),
            "adj_close": float(vals["5. adjusted close"]),
            "dividends": float(vals["7. dividend amount"]),
            "stock_splits": float(vals["8. split coefficient"]),
        })
    if not rows:
        raise ValueError(f"Alpha Vantage: no data for {ticker} in range {start}–{end}")
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        .sort("time")
    )
