from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest

from src.ingestion.news_collector import (
    NewsArticle,
    collect_rss,
    collect_newsapi,
    collect_finnhub,
    collect_all_news,
)

_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 7, tzinfo=timezone.utc)
_ARTICLE = NewsArticle(
    ticker="AAPL",
    headline="Apple hits new high",
    body=None,
    source="test",
    url="https://example.com/unique-url",
    published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
)


@patch("src.ingestion.news_collector.feedparser.parse")
def test_collect_rss_returns_list_of_articles(mock_parse):
    entry = MagicMock()
    entry.get.side_effect = lambda key, default="": {
        "title": "Apple hits new high",
        "link": "https://example.com/1",
        "summary": "Apple stock surged.",
    }.get(key, default)
    entry.published_parsed = (2024, 1, 2, 12, 0, 0, 0, 2, 0)

    mock_parse.return_value = MagicMock(entries=[entry])
    articles = collect_rss("AAPL")
    assert len(articles) == 1
    assert articles[0].headline == "Apple hits new high"
    assert articles[0].ticker == "AAPL"


@patch("src.ingestion.news_collector.settings")
@patch("src.ingestion.news_collector.NewsApiClient")
def test_collect_newsapi_returns_articles(mock_cls, mock_settings):
    mock_settings.NEWSAPI_KEY = "test-key"
    mock_cls.return_value.get_everything.return_value = {
        "articles": [{
            "title": "AAPL earnings beat",
            "url": "https://example.com/2",
            "publishedAt": "2024-01-03T10:00:00Z",
            "source": {"name": "Reuters"},
            "content": "Apple beat earnings.",
        }]
    }
    articles = collect_newsapi("AAPL", _START, _END)
    assert len(articles) == 1
    assert articles[0].ticker == "AAPL"


@patch("src.ingestion.news_collector.settings")
@patch("src.ingestion.news_collector.finnhub.Client")
def test_collect_finnhub_returns_articles(mock_cls, mock_settings):
    mock_settings.FINNHUB_KEY = "test-key"
    mock_cls.return_value.company_news.return_value = [{
        "headline": "AAPL buyback",
        "url": "https://example.com/3",
        "datetime": 1704153600,
        "source": "Finnhub",
        "summary": "Apple announces buyback.",
    }]
    articles = collect_finnhub("AAPL", _START, _END)
    assert len(articles) == 1


@patch("src.ingestion.news_collector.collect_rss", return_value=[_ARTICLE])
@patch("src.ingestion.news_collector.collect_newsapi", return_value=[_ARTICLE])
@patch("src.ingestion.news_collector.collect_finnhub", return_value=[])
def test_collect_all_news_deduplicates_by_url(mock_fh, mock_na, mock_rss):
    articles = collect_all_news("AAPL", _START, _END)
    assert len(articles) == 1
