"""China daily price limits: a stock that closes at its limit cannot be bought at the close."""
from __future__ import annotations

import polars as pl

# Main-board limit is 10% (ST stocks 5%, STAR/ChiNext 20%); 9.5% catches sealed
# main-board limits after rounding. ST flags are not in the data, so one threshold.
LIMIT_MOVE = 0.095


def limit_days(df: pl.DataFrame, threshold: float = LIMIT_MOVE) -> set[tuple]:
    """(ticker, time) rows whose close moved >= threshold vs the previous close."""
    d = (
        df.select("ticker", "time", "close").sort(["ticker", "time"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).abs().alias("_mv"))
        .filter(pl.col("_mv") >= threshold)
    )
    return set(zip(d["ticker"].to_list(), d["time"].to_list()))


def blocked_entries(market: str, df: pl.DataFrame) -> set[tuple]:
    """Rows where a Buy can't be filled: China limit days; nothing elsewhere."""
    return limit_days(df) if market == "china" else set()


def is_limit_move(prev_close: float | None, close: float | None, threshold: float = LIMIT_MOVE) -> bool:
    if not prev_close or close is None:
        return False
    return abs(close / prev_close - 1) >= threshold
