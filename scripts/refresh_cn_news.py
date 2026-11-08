"""Daily China stock news refresh (runs on the compute server B, in mainland China)."""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

import polars as pl

from config.markets import get_market
from scripts.build_features import _TICKERS_BY_MARKET
from src.ingestion.cn_news_ingest import SH, ingest_cn_news

logger = logging.getLogger(__name__)

_NAMES_MAX_AGE_DAYS = 7


def _trading_days(today: date) -> pl.Series:
    """Benchmark parquet when present (A / dev PC), else the exchange calendar via akshare (B)."""
    bench = get_market("china").data_root / "raw" / "ohlcv" / "510300.SS.parquet"
    if bench.exists():
        days = pl.read_parquet(bench)["time"].cast(pl.Date).unique().sort()
        # Add the next weekdays so today's after-close news has a target day.
        extra = [today + timedelta(days=i) for i in range(0, 8) if (today + timedelta(days=i)).weekday() < 5]
        return pl.concat([days, pl.Series(extra, dtype=pl.Date)]).unique().sort()
    import akshare as ak

    cal = ak.tool_trade_date_hist_sina()
    return pl.Series(pl.Series(cal["trade_date"].astype(str)).str.to_date()).alias("d")


def _refresh_names() -> None:
    path = get_market("china").data_root / "index_compositions" / "names.parquet"
    if path.exists():
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age < _NAMES_MAX_AGE_DAYS * 86400:
            return
    try:
        import pandas as pd

        pd.set_option("future.infer_string", False)
        import akshare as ak

        df = ak.stock_info_a_code_name()
        suffix = lambda c: f"{c}.SS" if c.startswith(("6", "9")) else f"{c}.SZ"
        names = pl.DataFrame({
            "ticker": [suffix(str(c)) for c in df["code"]],
            "name": [str(n) for n in df["name"]],
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        names.write_parquet(path)
        logger.info("names: %d rows", names.height)
    except Exception as exc:
        logger.warning("names refresh skipped: %s", exc)


def main(tickers: list[str] | None = None) -> dict:
    today = datetime.now(SH).date()
    tickers = tickers or list(_TICKERS_BY_MARKET["china"])
    _refresh_names()
    res = ingest_cn_news(tickers, get_market("china").data_root / "news", _trading_days(today), today=today)
    logger.info(
        "CN news: ok=%d failed=%s new=%d",
        len(res["tickers_ok"]), res["tickers_failed"][:20], res["articles_new"],
    )
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=lambda s: [t.strip() for t in s.split(",") if t.strip()])
    main(parser.parse_args().tickers)
