from datetime import datetime, timezone
import logging

import pandas as pd
import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pl.DataFrame:
    raw = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        back_adjust=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    if "date" in raw.columns:
        raw = raw.rename(columns={"date": "time"})

    # Convert timezone-aware datetime to string for Polars compatibility (avoids PyArrow requirement)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Build data dict with simple numpy types to avoid PyArrow requirement
    data = {
        "time": raw["time"].to_numpy(),
        "ticker": [ticker] * len(raw),
        "open": raw["open"].astype("float64").to_numpy(),
        "high": raw["high"].astype("float64").to_numpy(),
        "low": raw["low"].astype("float64").to_numpy(),
        "close": raw["close"].astype("float64").to_numpy(),
        "volume": raw["volume"].astype("int64").to_numpy(),
        "dividends": raw.get("dividends", pd.Series([0.0] * len(raw))).astype("float64").to_numpy(),
        "stock_splits": raw.get("stock_splits", pd.Series([0.0] * len(raw))).astype("float64").to_numpy(),
    }

    df = pl.DataFrame(data)

    return df.select([
        pl.col("time").str.to_datetime("%Y-%m-%d %H:%M:%S%z").cast(pl.Datetime("us", "UTC")),
        pl.col("ticker"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("close").alias("adj_close").cast(pl.Float64),
        pl.col("dividends").cast(pl.Float64),
        pl.col("stock_splits").cast(pl.Float64),
    ])


def fetch_corporate_actions(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        "dividends": t.dividends,
        "splits": t.splits,
        "earnings_dates": t.earnings_dates if t.earnings_dates is not None else pd.DataFrame(),
    }


def fetch_fundamentals(ticker: str) -> dict[str, float | None]:
    info = yf.Ticker(ticker).info
    return {
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "book_value": info.get("bookValue"),
        "market_cap": info.get("marketCap"),
    }
