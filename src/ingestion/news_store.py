"""Tiered compact news store.

Layout under ``news_dir``:
  articles_scores_YYYY-MM.parquet   per-article sentiment scores (no text, only url_hash)
  headlines_recent.parquet        short rolling window of raw headlines
  sentiment_daily.parquet  per (ticker, trading day) aggregates
"""

from __future__ import annotations

import calendar
import hashlib
import logging
from datetime import date, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
CUTOFF = time(16, 0)
DAILY_FILE = "sentiment_daily.parquet"
HEADLINES_FILE = "headlines_recent.parquet"

_DAILY_SCHEMA = {
    "ticker": pl.Categorical,
    "time": pl.Date,
    "avg_pos": pl.Float32,
    "avg_neg": pl.Float32,
    "avg_neu": pl.Float32,
    "article_count": pl.UInt32,
}


def url_hash(url: str) -> int:
    digest = hashlib.blake2b(url.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def assign_trading_day(published_at: pl.Series, trading_days: pl.Series) -> pl.Series:
    """Map each timestamp to its trading day (>=16:00 ET rolls to the next day)."""
    if published_at.dtype.time_zone is None:  # type: ignore[union-attr]
        published_at = published_at.dt.replace_time_zone("UTC")
    et = published_at.dt.convert_time_zone("America/New_York")
    after_close = et.dt.time() >= CUTOFF
    base = et.dt.date()
    target = (
        pl.DataFrame({"base": base, "after": after_close})
        .select(
            pl.when(pl.col("after"))
            .then(pl.col("base") + pl.duration(days=1))
            .otherwise(pl.col("base"))
            .alias("target")
        )
        .to_series()
    )
    left = pl.DataFrame({"target": target}).with_row_index("_i")
    right = pl.DataFrame({"target": trading_days.cast(pl.Date)}).sort("target").with_columns(
        pl.col("target").alias("trading_day")
    )
    out = left.sort("target").join_asof(right, on="target", strategy="forward").sort("_i")
    return out["trading_day"].alias("trading_day")


def _month_path(news_dir: Path, year: int, month: int) -> Path:
    return news_dir / f"articles_scores_{year:04d}-{month:02d}.parquet"


def _read_plain(path: Path) -> pl.DataFrame:
    """Read a parquet file with categoricals as strings for safe concatenation."""
    return pl.read_parquet(path).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))


def _write_scores_file(path: Path, df: pl.DataFrame) -> None:
    df.with_columns(
        pl.col("ticker").cast(pl.Categorical),
        pl.col("scorer").cast(pl.Categorical),
        pl.col("sentiment_pos", "sentiment_neg", "sentiment_neu").cast(pl.Float32),
    ).write_parquet(path, compression="zstd")


def write_scores(news_dir: Path, scores: pl.DataFrame) -> int:
    news_dir.mkdir(parents=True, exist_ok=True)
    if scores.is_empty():
        return 0
    scores = scores.with_columns(
        pl.col("ticker").cast(pl.Utf8), pl.col("scorer").cast(pl.Utf8)
    ).with_columns(
        pl.col("published_at").dt.convert_time_zone("UTC").dt.year().alias("_y"),
        pl.col("published_at").dt.convert_time_zone("UTC").dt.month().alias("_m"),
    )
    new_total = 0
    for (year, month), part in scores.partition_by(["_y", "_m"], as_dict=True).items():
        part = part.drop("_y", "_m")
        path = _month_path(news_dir, year, month)
        if path.exists():
            existing = _read_plain(path)
            merged = pl.concat([existing, part.select(existing.columns)], how="vertical_relaxed")
            before = existing.height
        else:
            merged = part
            before = 0
        merged = merged.unique(subset=["ticker", "url_hash"], keep="first", maintain_order=True)
        new_total += merged.height - before
        _write_scores_file(path, merged)
    return new_total


def write_headlines(
    news_dir: Path, rows: pl.DataFrame, today: date, keep_days: int = 14
) -> None:
    news_dir.mkdir(parents=True, exist_ok=True)
    path = news_dir / HEADLINES_FILE
    frames = []
    if path.exists():
        frames.append(_read_plain(path))
    if not rows.is_empty():
        frames.append(rows.with_columns(pl.col("ticker").cast(pl.Utf8)))
    if not frames:
        return
    merged = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["ticker", "url_hash"], keep="last", maintain_order=True
    )
    cutoff = today - timedelta(days=keep_days)
    merged = merged.filter(
        pl.col("published_at").dt.convert_time_zone("UTC").dt.date() >= cutoff
    )
    merged.with_columns(pl.col("ticker").cast(pl.Categorical)).write_parquet(
        path, compression="zstd"
    )


def _empty_daily() -> pl.DataFrame:
    return pl.DataFrame(schema=_DAILY_SCHEMA)


def load_daily(news_dir: Path) -> pl.DataFrame:
    path = news_dir / DAILY_FILE
    if not path.exists():
        return _empty_daily()
    return pl.read_parquet(path)


def rebuild_daily(news_dir: Path, trading_days: pl.Series, since: date) -> None:
    news_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(news_dir.glob("articles_scores_*.parquet"))
    old = load_daily(news_dir).with_columns(pl.col("ticker").cast(pl.Utf8))
    old = old.filter(pl.col("time") < since)

    if files:
        scores = pl.concat([_read_plain(p) for p in files], how="vertical_relaxed")
        days = assign_trading_day(scores["published_at"], trading_days)
        scores = scores.with_columns(days.alias("time")).filter(
            pl.col("time").is_not_null() & (pl.col("time") >= since)
        )
        new = (
            scores.group_by(["ticker", "time"])
            .agg(
                pl.col("sentiment_pos").mean().alias("avg_pos"),
                pl.col("sentiment_neg").mean().alias("avg_neg"),
                pl.col("sentiment_neu").mean().alias("avg_neu"),
                pl.len().alias("article_count"),
            )
            .select(list(_DAILY_SCHEMA))
        )
    else:
        new = _empty_daily().with_columns(pl.col("ticker").cast(pl.Utf8))

    out = (
        pl.concat([old.select(list(_DAILY_SCHEMA)), new], how="vertical_relaxed")
        .with_columns(
            pl.col("ticker").cast(pl.Categorical),
            pl.col("avg_pos", "avg_neg", "avg_neu").cast(pl.Float32),
            pl.col("article_count").cast(pl.UInt32),
        )
        .sort(["ticker", "time"])
    )
    out.write_parquet(news_dir / DAILY_FILE, compression="zstd")


def prune(news_dir: Path, today: date, keep_days: int = 60) -> None:
    cutoff = today - timedelta(days=keep_days)
    for path in news_dir.glob("articles_scores_*.parquet"):
        try:
            year, month = (int(x) for x in path.stem.removeprefix("articles_scores_").split("-"))
        except ValueError:
            continue
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        if last_day < cutoff:
            path.unlink()
            logger.info("Pruned %s", path.name)
