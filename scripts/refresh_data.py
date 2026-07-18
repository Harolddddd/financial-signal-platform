"""Incrementally refresh OHLCV data for all 151 tickers to today, then rebuild features."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from scripts.build_features import _STOCK_TICKERS, build_features_for_ticker
from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR = Path("data/raw/ohlcv")
_FEATURE_DIR = Path("data/features")
_AUX_TICKERS = ["SPY", "^VIX"]


def _refresh_raw(ticker: str, today: datetime) -> None:
    raw_path = _RAW_DIR / f"{ticker}.parquet"
    if not raw_path.exists():
        logger.warning("  %s — no raw file, skipping", ticker)
        return
    existing = pl.read_parquet(raw_path)
    max_ts = existing["time"].max()
    # NOTE: polars Datetime from .max() does not support .replace(tzinfo=...).
    # Reconstruct a timezone-aware datetime using the constructor instead.
    fetch_start = datetime(max_ts.year, max_ts.month, max_ts.day, tzinfo=timezone.utc) + timedelta(days=1)
    if fetch_start.date() >= today.date():
        logger.info("  %s already current (%s)", ticker, max_ts.date())
        return
    try:
        new_rows = fetch_ohlcv(ticker, fetch_start, today)
        if len(new_rows) == 0:
            logger.info("  %s — no new rows (market closed?)", ticker)
            return
        combined = (
            pl.concat([existing, new_rows])
            .unique(subset=["time"], keep="last")
            .sort("time")
        )
        combined.write_parquet(raw_path)
        logger.info("  %s +%d rows → %d total", ticker, len(new_rows), len(combined))
    except Exception as exc:
        logger.warning("  %s fetch failed: %s", ticker, exc)


def main() -> None:
    today = datetime.now(timezone.utc)
    logger.info("=== Data refresh → %s ===", today.date())

    logger.info("[1/2] Refreshing raw OHLCV (%d stock tickers + aux)", len(_STOCK_TICKERS))
    for ticker in list(_STOCK_TICKERS) + _AUX_TICKERS:
        _refresh_raw(ticker, today)

    spy_df = pl.read_parquet(_RAW_DIR / "SPY.parquet")
    vix_df = pl.read_parquet(_RAW_DIR / "^VIX.parquet")

    logger.info("[2/2] Rebuilding features for %d tickers", len(_STOCK_TICKERS))
    _FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker in _STOCK_TICKERS:
        if not (_RAW_DIR / f"{ticker}.parquet").exists():
            continue
        try:
            df = build_features_for_ticker(ticker, _RAW_DIR, spy_df, vix_df)
            df.write_parquet(_FEATURE_DIR / f"{ticker}.parquet")
            ok += 1
        except Exception as exc:
            logger.warning("  features FAIL %s: %s", ticker, exc)
            fail += 1

    logger.info("=== Done. features OK=%d fail=%d ===", ok, fail)


if __name__ == "__main__":
    main()
