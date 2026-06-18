from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import pytest
import tempfile

from src.features.duckdb_client import load_training_data


def _write_sample_parquet(tmp_dir: Path) -> None:
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc), datetime(2024, 1, 3, tzinfo=timezone.utc)],
        "ticker": ["AAPL", "AAPL"],
        "close": [153.0, 154.0],
        "rsi_14": [55.0, 57.0],
        "label": ["Buy", "Hold"],
    })
    df.write_parquet(tmp_dir / "AAPL.parquet")


def test_load_training_data_returns_polars_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        df = load_training_data(p)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2


def test_load_training_data_filters_by_ticker():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        df = load_training_data(p, tickers=["AAPL"])
        assert all(df["ticker"] == "AAPL")


def test_load_training_data_filters_by_date_range():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        start = datetime(2024, 1, 3, tzinfo=timezone.utc)
        df = load_training_data(p, start=start)
        assert len(df) == 1
        assert df["time"][0] == start
