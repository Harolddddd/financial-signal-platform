from __future__ import annotations
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl


def load_training_data(
    parquet_dir: Path,
    tickers: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    pattern = str(parquet_dir / "*.parquet")
    conditions: list[str] = []

    if tickers:
        quoted = ", ".join(f"'{t}'" for t in tickers)
        conditions.append(f"ticker IN ({quoted})")
    if start:
        conditions.append(f"time >= TIMESTAMPTZ '{start.isoformat()}'")
    if end:
        conditions.append(f"time <= TIMESTAMPTZ '{end.isoformat()}'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM read_parquet('{pattern}') {where} ORDER BY ticker, time"

    conn = duckdb.connect()
    return conn.execute(sql).pl()
