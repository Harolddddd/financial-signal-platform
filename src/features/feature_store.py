from __future__ import annotations
from datetime import datetime
from pathlib import Path
import logging

import polars as pl
import psycopg2

from src.ingestion.db import get_connection, release_connection, execute_many
from src.features.technical_indicators import add_technical_indicators
from src.features.sentiment_features import add_sentiment_features
from src.features.cross_asset_features import add_cross_asset_features
from src.features.label_generator import add_labels

logger = logging.getLogger(__name__)

_FEATURES_SQL = """
INSERT INTO features (
    time, ticker,
    sma_10, sma_20, sma_50, sma_200, ema_12, ema_26,
    rsi_14, macd, macd_signal, macd_hist,
    bb_upper, bb_lower, bb_width, atr_14, hist_vol_21,
    sent_pos_avg_3d, sent_pos_avg_5d, sent_pos_avg_10d,
    sent_pos_mom_3d, news_vol_spike,
    rel_strength_spy, vix_level,
    forward_return_5d, label
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (ticker, time) DO UPDATE SET
    sma_10=EXCLUDED.sma_10, sma_20=EXCLUDED.sma_20,
    rsi_14=EXCLUDED.rsi_14, macd=EXCLUDED.macd,
    label=EXCLUDED.label, forward_return_5d=EXCLUDED.forward_return_5d
"""

_FEATURE_COLS = [
    "time", "ticker",
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike",
    "rel_strength_spy", "vix_level",
    "forward_return_5d", "label",
]


def build_features(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    stock_df = _load_ohlcv_from_db(ticker, start, end)
    spy_df   = _load_spy_from_db(start, end)
    vix_df   = _load_vix_from_db(start, end)
    sent_df  = _load_sentiment_from_db(ticker, start, end)

    df = add_technical_indicators(stock_df)
    df = add_cross_asset_features(df, spy_df, vix_df)

    sent_feats = add_sentiment_features(sent_df)
    sent_feats = sent_feats.with_columns(
        pl.col("time").cast(pl.Date).alias("join_date")
    )
    df = df.with_columns(pl.col("time").dt.date().alias("join_date"))
    df = df.join(
        sent_feats.select([
            "join_date",
            "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
            "sent_pos_mom_3d", "news_vol_spike",
        ]),
        on="join_date",
        how="left",
    ).drop("join_date")

    df = add_labels(df)
    return df.drop_nulls(subset=["label"])


def write_features(df: pl.DataFrame) -> int:
    rows = [
        tuple(row[col] for col in _FEATURE_COLS)
        for row in df.select(_FEATURE_COLS).iter_rows(named=True)
    ]
    conn = get_connection()
    try:
        execute_many(conn, _FEATURES_SQL, rows)
    finally:
        release_connection(conn)
    return len(rows)


def export_parquet(df: pl.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}.parquet"
    df.write_parquet(path)
    return path


def _load_ohlcv_from_db(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT time, ticker, open, high, low, close, volume,
                       adj_close, dividends, stock_splits
                FROM ohlcv
                WHERE ticker = %s AND time >= %s AND time <= %s
                ORDER BY time ASC
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()
    finally:
        release_connection(conn)
    cols = ["time", "ticker", "open", "high", "low", "close", "volume",
            "adj_close", "dividends", "stock_splits"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def _load_sentiment_from_db(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    from src.features.sentiment_features import aggregate_daily_sentiment
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT published_at, ticker, sentiment_pos, sentiment_neg, sentiment_neu
                FROM news_articles
                WHERE ticker = %s AND published_at >= %s AND published_at <= %s
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()
    finally:
        release_connection(conn)
    if not rows:
        from datetime import date, timedelta
        n = (end.date() - start.date()).days + 1
        return pl.DataFrame({
            "time": [start.date() + timedelta(days=i) for i in range(n)],
            "ticker": [ticker] * n,
            "avg_pos": [0.5] * n, "avg_neg": [0.25] * n, "avg_neu": [0.25] * n,
            "article_count": [0] * n,
        })
    cols = ["published_at", "ticker", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
    raw = pl.DataFrame(rows, schema=cols, orient="row")
    return aggregate_daily_sentiment(raw)


def _load_spy_from_db(start: datetime, end: datetime) -> pl.DataFrame:
    return _load_ohlcv_from_db("SPY", start, end)


def _load_vix_from_db(start: datetime, end: datetime) -> pl.DataFrame:
    return _load_ohlcv_from_db("^VIX", start, end)
