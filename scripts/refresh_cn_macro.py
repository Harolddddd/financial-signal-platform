"""China macro refresh (runs on the compute server B): akshare -> macro_daily.parquet."""
from __future__ import annotations

import logging
from datetime import date, datetime

import polars as pl

from config.markets import get_market
from scripts.refresh_cn_news import _trading_days
from src.ingestion.cn_macro_ingest import build_cn_macro_daily, fetch_cn_macro_raw
from src.ingestion.cn_news_ingest import SH

logger = logging.getLogger(__name__)

_START = date(2005, 1, 1)


def main() -> pl.DataFrame:
    today = datetime.now(SH).date()
    days = _trading_days(today)
    days = days.filter((days >= _START) & (days <= today))
    out = build_cn_macro_daily(fetch_cn_macro_raw(), days)
    path = get_market("china").data_root / "macro" / "macro_daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path, compression="zstd")
    logger.info("CN macro: %d rows through %s -> %s", out.height, out["date"].max(), path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
