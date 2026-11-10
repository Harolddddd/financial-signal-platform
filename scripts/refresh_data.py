"""Incrementally refresh OHLCV data for the given market's tickers to today, then rebuild features."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from scripts.build_features import _TICKERS_BY_MARKET, build_features_for_ticker
from src.features.news_macro_features import MACRO_COLS, load_news_macro, load_us_news_macro
from src.ingestion.cn_macro_ingest import CN_MACRO_COLS
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
# Hour (UTC) after which the day's bar is final: China closes 15:00 Beijing (07:00 UTC),
# US 16:00 New York (21:00 UTC in winter, 20:00 in summer; 21 is safe all year).
_CLOSE_UTC_HOUR = {"us": 21, "china": 7}


def _last_complete_day(market: str, now: datetime) -> datetime:
    """Today if the market has closed, else yesterday: never store a half-finished bar."""
    day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return day if now.hour >= _CLOSE_UTC_HOUR.get(market, 0) else day - timedelta(days=1)


def _refresh_raw(ticker: str, today: datetime, raw_dir: Path) -> None:
    raw_path = raw_dir / f"{ticker}.parquet"
    if not raw_path.exists():
        # New ticker — fetch full history
        logger.info("  %s — new ticker, fetching full history from %s", ticker, _HISTORY_START.date())
        try:
            rows = fetch_ohlcv(ticker, _HISTORY_START, today + timedelta(days=1))
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
    # Re-fetch the last stored day too: a bar saved during the session is replaced by
    # its final close (rows are de-duplicated on time, newest kept below).
    fetch_start = datetime(max_ts.year, max_ts.month, max_ts.day, tzinfo=timezone.utc)
    if fetch_start.date() > today.date():
        logger.info("  %s already current (%s)", ticker, max_ts.date())
        return
    # Yahoo's end date is exclusive: end=today would never return today's bar.
    fetch_end = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) + timedelta(days=1)
    try:
        new_rows = fetch_ohlcv(ticker, fetch_start, fetch_end)
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


def _ingest_us_news(tickers: list[str], benchmark_df: pl.DataFrame, news_dir: Path) -> None:
    """Best-effort news ingest; must never break the data refresh."""
    try:
        from src.ingestion.news_ingest import ingest_news
        from src.ingestion.news_store import load_daily

        today = datetime.now(timezone.utc).date()
        daily = load_daily(news_dir)
        last = daily["time"].max() if daily.height else None
        start = (last if last is not None else today) - timedelta(days=3)
        trading_days = benchmark_df["time"].cast(pl.Date).unique().sort()
        res = ingest_news(list(tickers), start, today, news_dir, trading_days)
        logger.info(
            "News ingest: ok=%d failed=%s new=%d",
            len(res["tickers_ok"]), res["tickers_failed"], res["articles_new"],
        )
    except Exception as exc:
        logger.warning("News ingest skipped: %s", exc)


def _refresh_us_macro() -> None:
    """Best-effort macro (FRED + FOMC) refresh; must never break the data refresh."""
    try:
        from scripts.refresh_macro import main as refresh_macro

        refresh_macro()
    except Exception as exc:
        logger.warning("Macro refresh skipped: %s", exc)


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

    today = _last_complete_day(market, datetime.now(timezone.utc))
    logger.info("=== Data refresh (%s) → %s (last finished session) ===", market, today.date())

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

    if market == "us":
        _ingest_us_news(tickers, benchmark_df, market_cfg.data_root / "news")
        _refresh_us_macro()
        news_daily, macro_daily = load_us_news_macro()
    else:
        # China news + macro arrive from the compute server (B) via the inbox.
        news_daily, macro_daily = load_news_macro(market)

    logger.info("[2/2] Rebuilding features for %d tickers", len(tickers))
    feature_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker in tickers:
        if not (raw_dir / f"{ticker}.parquet").exists():
            continue
        try:
            df = build_features_for_ticker(
                ticker, raw_dir, benchmark_df, vix_df,
                news_daily=news_daily, macro_daily=macro_daily,
                macro_cols=CN_MACRO_COLS if market == "china" else MACRO_COLS,
            )
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
