"""Refresh US macro features (FRED + FOMC tone) -> markets/us/data/macro/*.parquet."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from config.markets import get_market
from config.settings import settings
from src.ingestion.macro_ingest import (
    FRED_SERIES,
    build_macro_daily,
    fetch_fomc_statements,
    fetch_fred,
    score_fomc_statements,
)

logger = logging.getLogger(__name__)

_FIRST_RUN_START = date(2010, 1, 1)


def main() -> None:
    if not settings.FRED_KEY:
        logger.warning("FRED_KEY not set, skipping macro refresh")
        return

    cfg = get_market("us")
    macro_dir = cfg.data_root / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    daily_path = macro_dir / "macro_daily.parquet"
    scores_path = macro_dir / "fomc_scores.parquet"

    spy = pl.read_parquet(cfg.data_root / "raw" / "ohlcv" / "SPY.parquet")
    trading_days = spy["time"].cast(pl.Date).unique().sort()

    # FRED is 5 cheap calls and macro_daily is rebuilt whole (forward-fill, 5d change and z20 need
    # full history, and raw FRED is not cached), so always refetch from the first-run start.
    fred = {sid: fetch_fred(sid, _FIRST_RUN_START, settings.FRED_KEY) for sid in FRED_SERIES}

    cached = pl.read_parquet(scores_path) if scores_path.exists() else None
    since = _FIRST_RUN_START
    if cached is not None and cached.height:
        since = cached["release_ts"].dt.convert_time_zone("America/New_York").dt.date().max() + timedelta(days=1)
    new = score_fomc_statements(fetch_fomc_statements(since))
    scores = new if cached is None else pl.concat([cached, new]).unique("release_ts").sort("release_ts")
    scores.write_parquet(scores_path, compression="zstd")

    macro = build_macro_daily(fred, scores, trading_days)
    macro.write_parquet(daily_path, compression="zstd")
    logger.info("Macro refresh: %d daily rows, %d FOMC statements", macro.height, scores.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
