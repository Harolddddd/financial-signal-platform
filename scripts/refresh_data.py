"""Incrementally refresh OHLCV data for the given market's tickers to today, then rebuild features."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from scripts.build_features import _TICKERS_BY_MARKET, build_features_for_ticker
from src.features.cross_asset_features import synthetic_vol_index
from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR = get_market("us").data_root / "raw" / "ohlcv"
_FEATURE_DIR = get_market("us").data_root / "features"
_AUX_TICKERS_BY_MARKET: dict[str, list[str]] = {
    "us": ["SPY", "^VIX"],
    "china": ["510300.SS"],
}


_HISTORY_START = datetime(1990, 1, 1, tzinfo=timezone.utc)


def _refresh_raw(ticker: str, today: datetime, raw_dir: Path) -> None:
    raw_path = raw_dir / f"{ticker}.parquet"
    if not raw_path.exists():
        # New ticker — fetch full history
        logger.info("  %s — new ticker, fetching full history from %s", ticker, _HISTORY_START.date())
        try:
            rows = fetch_ohlcv(ticker, _HISTORY_START, today)
            if len(rows) == 0:
                logger.warning("  %s — no data returned (delisted / invalid?)", ticker)
                return
            rows.write_parquet(raw_path)
            logger.info("  %s fetched %d rows", ticker, len(rows))
        except Exception as exc:
            logger.warning("  %s full fetch failed: %s", ticker, exc)
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


def main(market: str = "us") -> None:
    market_cfg = get_market(market)
    raw_dir = market_cfg.data_root / "raw" / "ohlcv"
    feature_dir = market_cfg.data_root / "features"
    if market not in _TICKERS_BY_MARKET:
        raise KeyError(f"No ticker list configured for market {market!r} in _TICKERS_BY_MARKET")
    tickers = _TICKERS_BY_MARKET[market]
    if market not in _AUX_TICKERS_BY_MARKET:
        raise KeyError(f"No aux ticker list configured for market {market!r} in _AUX_TICKERS_BY_MARKET")
    aux_tickers = _AUX_TICKERS_BY_MARKET[market]

    today = datetime.now(timezone.utc)
    logger.info("=== Data refresh (%s) → %s ===", market, today.date())

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[1/2] Refreshing raw OHLCV (%d stock tickers + aux)", len(tickers))
    for ticker in list(tickers) + aux_tickers:
        _refresh_raw(ticker, today, raw_dir)

    benchmark_path = raw_dir / f"{market_cfg.benchmark_ticker}.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"{market_cfg.benchmark_ticker}.parquet missing from {raw_dir}/. "
            "Run scripts/refresh_data.py first."
        )
    benchmark_df = pl.read_parquet(benchmark_path)
    if market_cfg.vol_index_ticker:
        vix_df = pl.read_parquet(raw_dir / f"{market_cfg.vol_index_ticker}.parquet")
    else:
        vix_df = synthetic_vol_index(benchmark_df)

    logger.info("[2/2] Rebuilding features for %d tickers", len(tickers))
    feature_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker in tickers:
        if not (raw_dir / f"{ticker}.parquet").exists():
            continue
        try:
            df = build_features_for_ticker(ticker, raw_dir, benchmark_df, vix_df)
            df.write_parquet(feature_dir / f"{ticker}.parquet")
            ok += 1
        except Exception as exc:
            logger.warning("  features FAIL %s: %s", ticker, exc)
            fail += 1

    logger.info("=== Done. features OK=%d fail=%d ===", ok, fail)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
