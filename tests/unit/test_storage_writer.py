from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import polars as pl

from src.ingestion.storage_writer import write_ohlcv, write_news, write_fundamentals
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult

_DF = pl.DataFrame({
    "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
    "ticker": ["AAPL"],
    "open": [150.0], "high": [155.0], "low": [149.0], "close": [153.0],
    "volume": [1_000_000], "adj_close": [153.0], "dividends": [0.0], "stock_splits": [0.0],
})
_ARTICLE = NewsArticle(
    ticker="AAPL", headline="Apple surges", body=None, source="test",
    url="https://example.com/z", published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
)
_SENTIMENT = SentimentResult(label="positive", positive=0.9, negative=0.05, neutral=0.05)


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_ohlcv_returns_row_count(mock_rel, mock_exec, mock_conn):
    n = write_ohlcv(_DF)
    assert mock_exec.called
    assert n == 1


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_news_returns_row_count(mock_rel, mock_exec, mock_conn):
    n = write_news([(_ARTICLE, _SENTIMENT)])
    assert mock_exec.called
    assert n == 1


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_fundamentals_calls_execute_many(mock_rel, mock_exec, mock_conn):
    snapshot = {"pe_ratio": 28.5, "eps": 6.43, "book_value": 4.5, "market_cap": 3e12}
    write_fundamentals("AAPL", snapshot, datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert mock_exec.called
