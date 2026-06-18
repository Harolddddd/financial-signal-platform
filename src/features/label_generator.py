import polars as pl


def add_labels(
    df: pl.DataFrame,
    forward_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pl.DataFrame:
    future_close = pl.col("close").shift(-forward_days)
    forward_return = (future_close / pl.col("close") - 1).alias("forward_return_5d")

    label = (
        pl.when(pl.col("forward_return_5d") > buy_threshold).then(pl.lit("Buy"))
        .when(pl.col("forward_return_5d") < sell_threshold).then(pl.lit("Sell"))
        .otherwise(pl.lit("Hold"))
        .alias("label")
    )

    return (
        df
        .with_columns(forward_return)
        .with_columns(
            pl.when(pl.col("forward_return_5d").is_null())
            .then(None)
            .otherwise(label)
            .alias("label")
        )
    )
