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


def test_load_training_data_unions_files_with_differing_columns(tmp_path):
    t = datetime(2024, 1, 2, tzinfo=timezone.utc)
    # first file alphabetically lacks the extra column (like stale BK.parquet)
    pl.DataFrame({"time": [t], "ticker": ["AAA"], "close": [1.0]}).write_parquet(
        tmp_path / "AAA.parquet"
    )
    pl.DataFrame(
        {"time": [t, t], "ticker": ["ZZZ", "ZZZ"], "close": [2.0, 3.0],
         "news_count_1d": [4.0, 5.0]}
    ).write_parquet(tmp_path / "ZZZ.parquet")
    df = load_training_data(tmp_path)
    assert len(df) == 3
    assert "news_count_1d" in df.columns
    assert df.filter(pl.col("ticker") == "AAA")["news_count_1d"].is_null().all()
    assert df.filter(pl.col("ticker") == "ZZZ")["news_count_1d"].to_list() == [4.0, 5.0]


def test_load_training_data_tolerates_file_missing_later_column(tmp_path):
    t = datetime(2024, 1, 2, tzinfo=timezone.utc)
    pl.DataFrame(
        {"time": [t], "ticker": ["AAA"], "close": [1.0], "news_count_1d": [4.0]}
    ).write_parquet(tmp_path / "AAA.parquet")
    pl.DataFrame({"time": [t], "ticker": ["BK"], "close": [2.0]}).write_parquet(
        tmp_path / "BK.parquet"
    )
    df = load_training_data(tmp_path)
    assert len(df) == 2
    assert df.filter(pl.col("ticker") == "BK")["news_count_1d"].is_null().all()
