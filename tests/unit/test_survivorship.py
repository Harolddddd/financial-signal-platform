from datetime import datetime, timezone
import polars as pl

from src.ingestion.survivorship import get_sp500_tickers_at, load_sp500_changes


def test_load_sp500_changes_returns_dataframe():
    df = load_sp500_changes()
    assert isinstance(df, pl.DataFrame)
    assert {"ticker", "added_date", "removed_date"}.issubset(df.columns)


def test_excludes_tickers_added_after_query_date():
    date = datetime(2005, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "GOOGL" not in tickers   # added 2006-03-31
    assert "TSLA" not in tickers    # added 2020-12-21
    assert "AAPL" in tickers        # added 1982-11-30


def test_excludes_tickers_removed_before_query_date():
    date = datetime(2009, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "ENRN" not in tickers    # removed 2001-12-03
    assert "LEHM" not in tickers    # removed 2008-09-15


def test_includes_active_tickers_at_query_date():
    date = datetime(2015, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "META" in tickers        # added 2013-12-23
