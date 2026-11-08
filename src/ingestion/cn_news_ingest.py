"""China A-share stock news (East Money via akshare) -> Chinese FinBERT -> tiered news store.

Same storage contract as the US pipeline: only url hash, headline (recent tier) and
scores are written; body text lives in memory for scoring only.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.ingestion import news_store
from src.ingestion.news_collector import NewsArticle
from src.ingestion.news_ingest import _REBUILD_EVERY, _existing_hashes, _score_and_write

logger = logging.getLogger(__name__)

SH = ZoneInfo("Asia/Shanghai")
CN_TZ = "Asia/Shanghai"
CN_CLOSE = dtime(15, 0)
_PACE_SECONDS = 1.0
_RETRY_SLEEP = 30
_LOOKBACK_DAYS = 60  # dedup window = score-file retention


def _akshare_news(code: str):
    import pandas as pd

    # akshare's regex cleanup breaks on pandas 3 arrow-backed strings.
    pd.set_option("future.infer_string", False)
    import akshare as ak

    return ak.stock_news_em(symbol=code)


def fetch_cn_stock_news(ticker: str, get=_akshare_news) -> list[NewsArticle]:
    """Latest East Money articles for one A-share ticker like '000009.SZ'."""
    code = ticker.split(".")[0]
    df = get(code)
    out: list[NewsArticle] = []
    if df is None or len(df) == 0:
        return out
    for r in df.to_dict("records"):
        url, title = r.get("新闻链接"), r.get("新闻标题")
        if not (url and title):
            continue
        try:
            local = datetime.strptime(str(r.get("发布时间")), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        out.append(NewsArticle(
            ticker=ticker,
            headline=str(title),
            body=str(r.get("新闻内容") or "") or None,
            source=str(r.get("文章来源") or "eastmoney"),
            url=str(url),
            published_at=local.replace(tzinfo=SH).astimezone(timezone.utc),
        ))
    return out


def _fetch_with_retry(fetch, ticker: str, sleep) -> list | None:
    for attempt in range(2):
        try:
            return fetch(ticker)
        except Exception as exc:  # network / parse errors from akshare
            logger.warning("cn news %s attempt %d failed: %s", ticker, attempt + 1, exc)
            if attempt == 0:
                sleep(_RETRY_SLEEP)
    return None


def ingest_cn_news(
    tickers: list[str],
    news_dir: Path,
    trading_days: pl.Series,
    fetch=fetch_cn_stock_news,
    sleep=time.sleep,
    today: date | None = None,
    scorer=None,
) -> dict:
    if scorer is None:
        from src.ingestion.sentiment_processor import score_texts_cn as scorer
    if today is None:
        today = datetime.now(SH).date()
    start = today - timedelta(days=_LOOKBACK_DAYS)
    seen = _existing_hashes(news_dir, start, today)
    ok: list[str] = []
    failed: list[str] = []
    new_count = 0
    for i, ticker in enumerate(tickers, start=1):
        got = _fetch_with_retry(fetch, ticker, sleep)
        sleep(_PACE_SECONDS)
        if got is None:
            failed.append(ticker)
            continue
        ok.append(ticker)
        fresh: dict[int, NewsArticle] = {}
        for a in got:
            if a.published_at.date() < start:
                continue
            h = news_store.url_hash(a.url)
            if (a.ticker, h) not in seen and h not in fresh:
                fresh[h] = a
        if fresh:
            new_count += _score_and_write(news_dir, fresh, today, tz=SH, scorer=scorer)
            seen.update((a.ticker, h) for h, a in fresh.items())
        logger.info("news progress %d/%d %s new_total=%d", i, len(tickers), ticker, new_count)
        if i % _REBUILD_EVERY == 0 and i < len(tickers):
            news_store.rebuild_daily(news_dir, trading_days, since=start, tz=CN_TZ, cutoff=CN_CLOSE)

    news_store.rebuild_daily(news_dir, trading_days, since=start, tz=CN_TZ, cutoff=CN_CLOSE)
    news_store.prune(news_dir, today=today)
    return {"tickers_ok": ok, "tickers_failed": failed, "articles_new": new_count}
