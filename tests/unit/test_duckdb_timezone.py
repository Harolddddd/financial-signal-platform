from datetime import date, datetime, timezone

import polars as pl

from src.features.duckdb_client import load_training_data


def test_loaded_times_stay_utc_whatever_the_server_timezone(tmp_path):
    """DuckDB returns TIMESTAMPTZ in the machine's zone: on server A (US Eastern) a bar
    stamped 2026-11-09 00:00 UTC came back as 11-08 19:00 EST, so .dt.date() gave the
    previous day and A scored B's predictions against outcomes one day off."""
    t = [datetime(2026, 11, d, tzinfo=timezone.utc) for d in (6, 9, 10)]
    pl.DataFrame({"time": t, "ticker": ["X"] * 3, "close": [1.0, 2.0, 3.0]}).write_parquet(tmp_path / "X.parquet")
    df = load_training_data(tmp_path, start=datetime(2026, 11, 9, tzinfo=timezone.utc))
    assert df.schema["time"] == pl.Datetime("us", "UTC")
    assert df["time"].dt.date().to_list() == [date(2026, 11, 9), date(2026, 11, 10)]
