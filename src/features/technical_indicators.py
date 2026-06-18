import polars as pl


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = _add_moving_averages(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger_bands(df)
    df = _add_atr(df)
    df = _add_hist_vol(df)
    return df


def _add_moving_averages(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
        pl.col("close").ewm_mean(span=12, adjust=False).alias("ema_12"),
        pl.col("close").ewm_mean(span=26, adjust=False).alias("ema_26"),
    ])


def _add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    gain = (
        pl.col("close").diff()
        .clip(lower_bound=0)
        .ewm_mean(span=period, adjust=False)
    )
    loss = (
        (pl.col("close").diff() * -1)
        .clip(lower_bound=0)
        .ewm_mean(span=period, adjust=False)
    )
    rsi = (100 - 100 / (1 + gain / loss)).alias("rsi_14")
    return df.with_columns(rsi)


def _add_macd(df: pl.DataFrame) -> pl.DataFrame:
    ema12 = pl.col("close").ewm_mean(span=12, adjust=False)
    ema26 = pl.col("close").ewm_mean(span=26, adjust=False)
    return (
        df
        .with_columns((ema12 - ema26).alias("macd"))
        .with_columns(pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal"))
        .with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"))
    )


def _add_bollinger_bands(df: pl.DataFrame, period: int = 20, n_std: float = 2.0) -> pl.DataFrame:
    sma = pl.col("close").rolling_mean(window_size=period)
    std = pl.col("close").rolling_std(window_size=period)
    return df.with_columns([
        (sma + n_std * std).alias("bb_upper"),
        (sma - n_std * std).alias("bb_lower"),
        ((n_std * std * 2) / sma).alias("bb_width"),
    ])


def _add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        true_range.ewm_mean(span=period, adjust=False).alias("atr_14")
    )


def _add_hist_vol(df: pl.DataFrame, period: int = 21) -> pl.DataFrame:
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    hist_vol = log_ret.rolling_std(window_size=period) * (252 ** 0.5)
    return df.with_columns(hist_vol.alias("hist_vol_21"))
