"""Backfill US news sentiment from Finnhub. Run via scripts/run_job.py for long ranges."""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import polars as pl

from config.markets import get_market
from scripts.build_features import _TICKERS_BY_MARKET
from src.ingestion.news_ingest import ingest_news

logger = logging.getLogger(__name__)


def main(days: int = 365) -> dict:
    cfg = get_market("us")
    spy = pl.read_parquet(cfg.data_root / "raw" / "ohlcv" / "SPY.parquet")
    trading_days = spy["time"].cast(pl.Date).unique().sort()
    today = date.today()
    result = ingest_news(
        _TICKERS_BY_MARKET["us"],
        today - timedelta(days=days),
        today,
        cfg.data_root / "news",
        trading_days,
    )
    logger.info(
        "News backfill: ok=%d failed=%s new=%d",
        len(result["tickers_ok"]), result["tickers_failed"], result["articles_new"],
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    main(parser.parse_args().days)
