from datetime import datetime
import logging

import polars as pl

from src.ingestion.db import get_connection, release_connection, execute_many
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult

logger = logging.getLogger(__name__)

_OHLCV_SQL = """
INSERT INTO ohlcv (time, ticker, open, high, low, close, volume, adj_close, dividends, stock_splits)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, time) DO UPDATE SET
    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
    volume=EXCLUDED.volume, adj_close=EXCLUDED.adj_close,
    dividends=EXCLUDED.dividends, stock_splits=EXCLUDED.stock_splits
"""

_NEWS_SQL = """
INSERT INTO news_articles
    (published_at, ticker, headline, body, source, url, relevance,
     sentiment_pos, sentiment_neg, sentiment_neu, sentiment_label)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (url, published_at) DO NOTHING
"""

_FUND_SQL = """
INSERT INTO fundamentals (time, ticker, pe_ratio, eps, book_value, market_cap)
VALUES (%s, %s, %s, %s, %s, %s)
"""


def write_ohlcv(df: pl.DataFrame) -> int:
    rows = [
        (r["time"], r["ticker"], r["open"], r["high"], r["low"],
         r["close"], r["volume"], r["adj_close"], r["dividends"], r["stock_splits"])
        for r in df.iter_rows(named=True)
    ]
    conn = get_connection()
    try:
        execute_many(conn, _OHLCV_SQL, rows)
        conn.commit()
    finally:
        release_connection(conn)
    return len(rows)


def write_news(pairs: list[tuple[NewsArticle, SentimentResult]]) -> int:
    rows = [
        (a.published_at, a.ticker, a.headline, a.body, a.source, a.url,
         1.0, s.positive, s.negative, s.neutral, s.label)
        for a, s in pairs
    ]
    conn = get_connection()
    try:
        execute_many(conn, _NEWS_SQL, rows)
        conn.commit()
    finally:
        release_connection(conn)
    return len(rows)


def write_fundamentals(
    ticker: str,
    snapshot: dict[str, float | None],
    as_of: datetime,
) -> None:
    rows = [(
        as_of, ticker,
        snapshot.get("pe_ratio"), snapshot.get("eps"),
        snapshot.get("book_value"), snapshot.get("market_cap"),
    )]
    conn = get_connection()
    try:
        execute_many(conn, _FUND_SQL, rows)
        conn.commit()
    finally:
        release_connection(conn)
