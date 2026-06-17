from __future__ import annotations
from datetime import datetime
import requests
import polars as pl
from config.settings import settings

def fetch_ohlcv_fmp(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Fetch daily OHLCV from Financial Modeling Prep historical-price-full."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "apikey": settings.FMP_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    series = data.get("historical", [])
    if not series:
        raise ValueError(f"FMP returned no data for {ticker}")
    rows = [
        {
            "time": datetime.fromisoformat(item["date"]),
            "ticker": ticker,
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
            "volume": int(item["volume"]),
            "adj_close": float(item.get("adjClose", item["close"])),
            "dividends": float(item.get("dividend", 0.0)),
            "stock_splits": float(item.get("unadjustedVolume", 0.0)) / max(float(item.get("volume", 1)), 1),
        }
        for item in series
    ]
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        .sort("time")
    )
