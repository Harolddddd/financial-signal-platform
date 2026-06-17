from __future__ import annotations
from datetime import datetime
import polars as pl
from src.ingestion.historical_collector import fetch_ohlcv
from src.ingestion.alpha_vantage_client import fetch_ohlcv_av
from src.ingestion.fmp_client import fetch_ohlcv_fmp

def collect_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pl.DataFrame:
    """Fetch OHLCV, trying yfinance → Alpha Vantage → FMP; raises RuntimeError if all fail."""
    sources = [
        ("yfinance",       lambda: fetch_ohlcv(ticker, start, end, interval)),
        ("alpha_vantage",  lambda: fetch_ohlcv_av(ticker, start, end)),
        ("fmp",            lambda: fetch_ohlcv_fmp(ticker, start, end)),
    ]
    errors: list[str] = []
    for name, fn in sources:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"All data sources failed for {ticker}: {'; '.join(errors)}")
