from datetime import datetime
from pathlib import Path

import polars as pl

from config.markets import get_market

_CSV_PATH = get_market("us").data_root / "index_compositions" / "sp500_changes.csv"


def load_sp500_changes() -> pl.DataFrame:
    return pl.read_csv(
        _CSV_PATH,
        schema_overrides={"ticker": pl.Utf8, "added_date": pl.Date, "removed_date": pl.Date},
        try_parse_dates=True,
        null_values=[""],
    )


def get_sp500_tickers_at(date: datetime) -> list[str]:
    query_date = date.date() if hasattr(date, "date") else date
    df = load_sp500_changes()
    active = df.filter(
        (pl.col("added_date") <= pl.lit(query_date))
        & (
            pl.col("removed_date").is_null()
            | (pl.col("removed_date") > pl.lit(query_date))
        )
    )
    return active["ticker"].to_list()
