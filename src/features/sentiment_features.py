import polars as pl


def aggregate_daily_sentiment(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates news_articles rows (one per article) into one row per (ticker, calendar date).
    raw_df must have columns: published_at (Datetime UTC), ticker, sentiment_pos, sentiment_neg, sentiment_neu.
    """
    return (
        raw_df
        .with_columns(pl.col("published_at").dt.date().alias("time"))
        .group_by(["ticker", "time"])
        .agg([
            pl.col("sentiment_pos").mean().alias("avg_pos"),
            pl.col("sentiment_neg").mean().alias("avg_neg"),
            pl.col("sentiment_neu").mean().alias("avg_neu"),
            pl.col("sentiment_pos").count().alias("article_count"),
        ])
        .sort("time")
    )


def add_sentiment_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds rolling sentiment features to a daily-aggregated sentiment DataFrame.
    Input must be sorted ascending by time, single ticker.
    """
    return (
        df
        .with_columns([
            pl.col("avg_pos").rolling_mean(window_size=3).alias("sent_pos_avg_3d"),
            pl.col("avg_pos").rolling_mean(window_size=5).alias("sent_pos_avg_5d"),
            pl.col("avg_pos").rolling_mean(window_size=10).alias("sent_pos_avg_10d"),
        ])
        .with_columns(
            (pl.col("sent_pos_avg_3d") - pl.col("sent_pos_avg_5d")).alias("sent_pos_mom_3d")
        )
        .with_columns([
            _zscore_rolling(pl.col("article_count").cast(pl.Float64), window=20).alias("news_vol_spike"),
        ])
    )


def _zscore_rolling(expr: pl.Expr, window: int) -> pl.Expr:
    mean = expr.rolling_mean(window_size=window)
    std = expr.rolling_std(window_size=window)
    return (expr - mean) / (std + 1e-9)
