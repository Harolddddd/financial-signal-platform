import polars as pl

_FORWARD_DAYS = 5
_RS_CLAMP = 5.0


def add_cross_asset_features(
    stock_df: pl.DataFrame,
    spy_df: pl.DataFrame,
    vix_df: pl.DataFrame,
) -> pl.DataFrame:
    stock_dates = stock_df.select(pl.col("time").dt.date().alias("date"))
    spy_ret = _five_day_return(spy_df).rename({"ret_5d": "spy_ret_5d", "date": "date"})
    stock_with_ret = stock_df.with_columns(
        pl.col("time").dt.date().alias("date")
    )
    stock_with_ret = _add_five_day_return(stock_with_ret)

    vix_daily = vix_df.with_columns(
        pl.col("time").dt.date().alias("date")
    ).select(["date", pl.col("close").alias("vix_level")])

    result = (
        stock_with_ret
        .join(spy_ret, on="date", how="left")
        .join(vix_daily, on="date", how="left")
    )

    rel_strength = (
        (pl.col("ret_5d") / (pl.col("spy_ret_5d").abs() + 1e-9))
        .clip(lower_bound=-_RS_CLAMP, upper_bound=_RS_CLAMP)
        .alias("rel_strength_spy")
    )

    return result.with_columns(rel_strength).drop(["date", "ret_5d", "spy_ret_5d"])


def synthetic_vol_index(benchmark_df: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    """Rolling realized volatility of the benchmark's close-to-close returns,
    annualized and scaled to a VIX-style index level, shaped like a real
    vix_df (time, close) so it flows through add_cross_asset_features()
    unchanged. Used when a market has no reliable direct vol-index ticker
    (MarketConfig.vol_index_ticker is None)."""
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    realized_vol = (
        log_ret.rolling_std(window_size=window) * (252 ** 0.5) * 100
    ).alias("close")
    return benchmark_df.select([
        pl.col("time"),
        realized_vol,
    ]).drop_nulls()


def _add_five_day_return(df: pl.DataFrame) -> pl.DataFrame:
    ret = (pl.col("close") / pl.col("close").shift(_FORWARD_DAYS) - 1).alias("ret_5d")
    return df.with_columns(ret)


def _five_day_return(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns(pl.col("time").dt.date().alias("date"))
        .with_columns(
            (pl.col("close") / pl.col("close").shift(_FORWARD_DAYS) - 1).alias("ret_5d")
        )
        .select(["date", "ret_5d"])
    )
