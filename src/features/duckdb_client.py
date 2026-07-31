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
    # DuckDB auto-detects memory_limit as ~80% of system RAM (25GB on a
    # 27.6GB machine) with no cap otherwise — enough on its own to starve
    # the rest of the system (Windows, streamlit, joblib workers) and has
    # caused multiple crash-reboots. This dataset is ~1GB in memory; a few
    # GB of headroom is more than sufficient.
    conn.execute("SET memory_limit='4GB'")
    conn.execute("SET threads TO 4")
    return conn.execute(sql).pl()
