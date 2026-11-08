"""China daily price limits: a stock that closes at its limit cannot be bought at the close."""
from __future__ import annotations

import polars as pl

# Daily limit: 10% main board, 20% ChiNext (300/301) and STAR (688); thresholds sit
# just under the limit to catch sealed limits after price rounding. ST stocks (5%)
# are not flagged in the data and fall under the main-board threshold.
LIMIT_MOVE = 0.095
LIMIT_MOVE_20 = 0.195
_TWENTY_PCT_PREFIXES = ("300", "301", "688")


def limit_threshold(ticker: str) -> float:
    return LIMIT_MOVE_20 if str(ticker).startswith(_TWENTY_PCT_PREFIXES) else LIMIT_MOVE


def limit_days(df: pl.DataFrame) -> set[tuple]:
    """(ticker, time) rows whose close moved to the board's daily limit vs the previous close."""
    d = (
        df.select("ticker", "time", "close").sort(["ticker", "time"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).abs().alias("_mv"))
        .with_columns(
            pl.when(pl.col("ticker").str.slice(0, 3).is_in(list(_TWENTY_PCT_PREFIXES)))
            .then(LIMIT_MOVE_20).otherwise(LIMIT_MOVE).alias("_lim")
        )
        .filter(pl.col("_mv") >= pl.col("_lim"))
    )
    return set(zip(d["ticker"].to_list(), d["time"].to_list()))


def blocked_entries(market: str, df: pl.DataFrame) -> set[tuple]:
    """Rows where a Buy can't be filled: China limit days; nothing elsewhere."""
    return limit_days(df) if market == "china" else set()


def is_limit_move(prev_close: float | None, close: float | None, ticker: str = "") -> bool:
    if not prev_close or close is None:
        return False
    return abs(close / prev_close - 1) >= limit_threshold(ticker)
