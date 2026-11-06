"""Finnhub news -> in-memory scoring -> compact tiered store (no body text is persisted)."""

from __future__ import annotations

import calendar
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.settings import settings
from src.ingestion import news_store
from src.ingestion.news_collector import collect_finnhub
from src.ingestion.sentiment_processor import score_texts

logger = logging.getLogger(__name__)

_MAX_CALLS_PER_MIN = 55
_PACE_SECONDS = 60 / _MAX_CALLS_PER_MIN
_RETRY_SLEEP = 60


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur <= end:
        month_end = date(cur.year, cur.month, calendar.monthrange(cur.year, cur.month)[1])
        chunks.append((cur, min(month_end, end)))
        cur = month_end + timedelta(days=1)
    return chunks


def _as_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _fetch_with_retry(fetch, ticker: str, d0: date, d1: date, sleep) -> list | None:
    """Paced fetch; on a 429 sleep 60s and retry once, then give up (None)."""
    for attempt in range(2):
        try:
            result = fetch(ticker, _as_dt(d0), _as_dt(d1))
            sleep(_PACE_SECONDS)
            return result
        except Exception as exc:
            if "429" not in str(exc):
                logger.warning("News fetch failed for %s: %s", ticker, exc)
                return None
            if attempt == 0:
                logger.warning("429 for %s, sleeping %ds", ticker, _RETRY_SLEEP)
                sleep(_RETRY_SLEEP)
    logger.warning("429 persisted for %s, skipping", ticker)
    return None


def ingest_news(
    tickers: list[str],
    start: date,
    end: date,
    news_dir: Path,
    trading_days: pl.Series,
    fetch=collect_finnhub,
    sleep=time.sleep,
) -> dict:
    if fetch is collect_finnhub and not settings.FINNHUB_KEY:
        logger.warning("FINNHUB_KEY not set, skipping news ingest")
        return {"tickers_ok": [], "tickers_failed": [], "articles_new": 0}

    articles = []
    ok: list[str] = []
    failed: list[str] = []
    for ticker in tickers:
        ticker_ok = True
        for d0, d1 in _month_chunks(start, end):
            got = _fetch_with_retry(fetch, ticker, d0, d1, sleep)
            if got is None:
                ticker_ok = False
                break
            articles.extend(a for a in got if a.url and a.headline)
        (ok if ticker_ok else failed).append(ticker)

    new_count = 0
    if articles:
        texts = [f"{a.headline} {a.body or ''}".strip() for a in articles]
        results = score_texts(texts)
        base = pl.DataFrame(
            {
                "ticker": [a.ticker for a in articles],
                "published_at": [a.published_at for a in articles],
                "url_hash": [news_store.url_hash(a.url) for a in articles],
                "headline": [a.headline for a in articles],
                "sentiment_pos": [r.positive for r in results],
                "sentiment_neg": [r.negative for r in results],
                "sentiment_neu": [r.neutral for r in results],
                "scorer": [r.scorer for r in results],
            }
        ).with_columns(pl.col("published_at").dt.convert_time_zone("UTC"))
        scores = base.drop("headline")
        headlines = base.select("url_hash", "ticker", "published_at", "headline")
        new_count = news_store.write_scores(news_dir, scores)
        news_store.write_headlines(news_dir, headlines, today=date.today())

    # Rebuild before prune: rebuild needs score files covering `start`.
    news_store.rebuild_daily(news_dir, trading_days, since=start)
    news_store.prune(news_dir, today=date.today())
    return {"tickers_ok": ok, "tickers_failed": failed, "articles_new": new_count}
