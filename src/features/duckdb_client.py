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
    # union_by_name: feature files may differ in columns (e.g. some US files
    # carry the news/macro overlay columns, stale ones don't); missing
    # columns come back as NULL instead of a schema-mismatch error.
    sql = (
        f"SELECT * FROM read_parquet('{pattern}', union_by_name=true) "
        f"{where} ORDER BY ticker, time"
    )

    conn = duckdb.connect()
    # DuckDB auto-detects memory_limit as ~80% of system RAM (25GB on a
    # 27.6GB machine) with no cap otherwise — enough on its own to starve
    # the rest of the system (Windows, streamlit, joblib workers) and has
    # caused multiple crash-reboots. This dataset is ~1GB in memory; a few
    # GB of headroom is more than sufficient.
    conn.execute("SET memory_limit='4GB'")
    conn.execute("SET threads TO 4")
    # TIMESTAMPTZ comes back in the session zone, which defaults to the machine's:
    # on a US Eastern server a bar stamped 00:00 UTC became 19:00 the day before.
    # Every date in the pipeline is the UTC date, so pin it.
    conn.execute("SET TimeZone='UTC'")
    return conn.execute(sql).pl()
