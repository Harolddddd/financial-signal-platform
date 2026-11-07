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
# Rebuild the permanent daily file every N tickers, so a long backfill that is
# killed mid-run keeps what it finished (score files may be pruned later).
_REBUILD_EVERY = 25
# Finnhub returns at most ~250 articles per request, newest first. A response
# this large is assumed truncated and its range is re-fetched in halves.
_TRUNCATED_AT = 240


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


def _fetch_range(fetch, ticker: str, d0: date, d1: date, sleep) -> list | None:
    """All articles in [d0, d1]: split the range while responses look truncated."""
    got = _fetch_with_retry(fetch, ticker, d0, d1, sleep)
    if got is None or len(got) < _TRUNCATED_AT or d0 == d1:
        return got
    mid = d0 + (d1 - d0) // 2
    left = _fetch_range(fetch, ticker, d0, mid, sleep)
    right = _fetch_range(fetch, ticker, mid + timedelta(days=1), d1, sleep)
    if left is None or right is None:
        return None
    return left + right


def ingest_news(
    tickers: list[str],
    start: date,
    end: date,
    news_dir: Path,
    trading_days: pl.Series,
    fetch=collect_finnhub,
    sleep=time.sleep,
    today: date | None = None,
) -> dict:
    if fetch is collect_finnhub and not settings.FINNHUB_KEY:
        logger.warning("FINNHUB_KEY not set, skipping news ingest")
        return {"tickers_ok": [], "tickers_failed": [], "articles_new": 0}
    if today is None:
        today = datetime.now(timezone.utc).date()

    # Hashes already scored, read once per run; articles in it are never re-scored.
    seen = _existing_hashes(news_dir, start, end)
    ok: list[str] = []
    failed: list[str] = []
    new_count = 0
    # One ticker at a time: only that ticker's articles are ever held in
    # memory, and its scores/headlines hit disk before the next ticker starts.
    for i, ticker in enumerate(tickers, start=1):
        ticker_ok = True
        fresh: dict[int, object] = {}
        for d0, d1 in _month_chunks(start, end):
            got = _fetch_range(fetch, ticker, d0, d1, sleep)
            if got is None:
                ticker_ok = False
                break
            for a in got:
                if not (a.url and a.headline):
                    continue
                h = news_store.url_hash(a.url)
                if (a.ticker, h) not in seen and h not in fresh:
                    fresh[h] = a
        (ok if ticker_ok else failed).append(ticker)
        if fresh:
            new_count += _score_and_write(news_dir, fresh, today)
            seen.update((a.ticker, h) for h, a in fresh.items())
        if i % _REBUILD_EVERY == 0 and i < len(tickers):
            news_store.rebuild_daily(news_dir, trading_days, since=start)

    # Rebuild before prune: rebuild needs score files covering `start`.
    news_store.rebuild_daily(news_dir, trading_days, since=start)
    news_store.prune(news_dir, today=today)
    return {"tickers_ok": ok, "tickers_failed": failed, "articles_new": new_count}


def _existing_hashes(news_dir: Path, start: date, end: date) -> set[tuple[str, int]]:
    """(ticker, url_hash) pairs already stored in score files for months in range."""
    out: set[tuple[str, int]] = set()
    for d0, _ in _month_chunks(start, end):
        path = news_store._month_path(news_dir, d0.year, d0.month)
        if not path.exists():
            continue
        df = pl.read_parquet(path, columns=["ticker", "url_hash"])
        out.update(zip(df["ticker"].cast(pl.Utf8).to_list(), df["url_hash"].to_list()))
    return out


def _score_and_write(news_dir: Path, fresh: dict, today: date) -> int:
    hashes = list(fresh)
    articles = [fresh[h] for h in hashes]
    # Body text is used only in memory for scoring; it is never written.
    texts = [f"{a.headline} {a.body or ''}".strip() for a in articles]
    results = score_texts(texts)
    base = pl.DataFrame(
        {
            "ticker": [a.ticker for a in articles],
            "published_at": [a.published_at for a in articles],
            "url_hash": hashes,
            "headline": [a.headline for a in articles],
            "sentiment_pos": [r.positive for r in results],
            "sentiment_neg": [r.negative for r in results],
            "sentiment_neu": [r.neutral for r in results],
            "scorer": [r.scorer for r in results],
        },
        schema_overrides={"url_hash": pl.Int64},
    ).with_columns(pl.col("published_at").dt.convert_time_zone("UTC"))
    new_count = news_store.write_scores(news_dir, base.drop("headline"))
    news_store.write_headlines(
        news_dir, base.select("url_hash", "ticker", "published_at", "headline"), today=today
    )
    return new_count
