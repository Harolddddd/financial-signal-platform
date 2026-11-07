"""US-only news and macro columns joined onto a single ticker's feature frame.

Every column produced here is null-free: missing news/macro days get neutral values so
downstream null-dropping never deletes history.
"""
import logging

import polars as pl

from config.markets import get_market

logger = logging.getLogger(__name__)

NEWS_COLS = [
    "news_count_1d", "news_count_3d", "news_has_3d",
    "news_net_3d", "news_net_z60", "news_vol_z20",
]
MACRO_COLS = [
    "macro_dff", "macro_dgs2", "macro_dgs10", "macro_t10y2y", "macro_hy_spread",
    "macro_dgs2_chg_5d", "macro_dgs10_chg_5d", "macro_hy_spread_chg_5d",
    "macro_hy_spread_z20",
    "fomc_tone", "fomc_tone_change", "days_since_fomc",
]
OVERLAY_COLS = NEWS_COLS + MACRO_COLS

_NO_FOMC_DAYS = 999


def _z(x: pl.Expr, window: int, min_samples: int) -> pl.Expr:
    """Trailing z-score (window includes the current row); 0.0 when undefined."""
    z = (x - x.rolling_mean(window, min_samples=min_samples)) / x.rolling_std(
        window, min_samples=min_samples
    )
    return pl.when(z.is_finite()).then(z).otherwise(0.0)


def add_news_features(df: pl.DataFrame, news_daily: pl.DataFrame | None) -> pl.DataFrame:
    """Join daily sentiment on (ticker, date) and add NEWS_COLS. `df` is one ticker in time order."""
    key = pl.col("time").dt.date().alias("_date")
    out = df.with_columns(key)
    if news_daily is None or news_daily.height == 0:
        out = out.with_columns(
            pl.lit(0).cast(pl.Int64).alias("_cnt"), pl.lit(0.0).alias("_net")
        )
    else:
        n = news_daily.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("time").alias("_date"),
            pl.col("article_count").cast(pl.Int64).alias("_cnt"),
            (pl.col("avg_pos") - pl.col("avg_neg")).alias("_net"),
        ).unique(subset=["ticker", "_date"], keep="last")
        out = out.join(n, on=["ticker", "_date"], how="left", maintain_order="left")
        out = out.with_columns(
            pl.col("_cnt").fill_null(0), pl.col("_net").fill_null(0.0)
        )
    out = out.with_columns(
        pl.col("_cnt").alias("news_count_1d"),
        pl.col("_cnt").rolling_sum(3, min_samples=1).alias("news_count_3d"),
        (pl.col("_net") * pl.col("_cnt")).rolling_sum(3, min_samples=1).alias("_wnet3"),
    ).with_columns(
        (pl.col("news_count_3d") > 0).alias("news_has_3d"),
        pl.when(pl.col("news_count_3d") > 0)
        .then(pl.col("_wnet3") / pl.col("news_count_3d"))
        .otherwise(0.0)
        .alias("news_net_3d"),
    ).with_columns(
        _z(pl.col("news_net_3d"), 60, 20).alias("news_net_z60"),
        _z(pl.col("news_count_1d").cast(pl.Float64), 20, 10).alias("news_vol_z20"),
    )
    return out.drop("_date", "_cnt", "_net", "_wnet3")


def add_macro_features(df: pl.DataFrame, macro_daily: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join macro_daily on date and add MACRO_COLS; missing values become 0.0 (days_since_fomc 999)."""
    neutral = {c: (_NO_FOMC_DAYS if c == "days_since_fomc" else 0.0) for c in MACRO_COLS}
    if macro_daily is None or macro_daily.height == 0:
        return df.with_columns([
            pl.lit(v).cast(pl.Int64 if c == "days_since_fomc" else pl.Float64).alias(c)
            for c, v in neutral.items()
        ])
    m = macro_daily.select(pl.col("date").alias("_date"), *MACRO_COLS)
    out = df.with_columns(pl.col("time").dt.date().alias("_date"))
    out = out.join(m, on="_date", how="left", maintain_order="left").drop("_date")
    return out.with_columns([
        pl.col(c).cast(pl.Int64 if c == "days_since_fomc" else pl.Float64).fill_null(v)
        for c, v in neutral.items()
    ])


def load_us_news_macro() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the US news-daily and macro-daily frames once per run.

    A missing file yields an empty frame (never None) so that US feature builds still
    attach neutral columns, while None stays the China "no overlay" signal.
    """
    from src.ingestion.news_store import DAILY_FILE, load_daily

    root = get_market("us").data_root
    news_path = root / "news" / DAILY_FILE
    if not news_path.exists():
        logger.warning("US news daily file missing (%s); news columns will be neutral", news_path)
    news = load_daily(root / "news")
    path = root / "macro" / "macro_daily.parquet"
    if path.exists():
        macro = pl.read_parquet(path)
    else:
        logger.warning("US macro file missing (%s); macro columns will be neutral", path)
        macro = pl.DataFrame()
    return news, macro
