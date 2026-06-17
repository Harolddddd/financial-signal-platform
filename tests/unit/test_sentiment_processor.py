from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.ingestion.sentiment_processor import SentimentResult, score_headline, score_articles
from src.ingestion.news_collector import NewsArticle


def _article(headline: str) -> NewsArticle:
    return NewsArticle(
        ticker="AAPL", headline=headline, body=None, source="test",
        url="https://example.com/x", published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


@patch("src.ingestion.sentiment_processor._get_finbert_pipeline")
def test_score_headline_returns_sentiment_result(mock_fn):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [[
        {"label": "positive", "score": 0.85},
        {"label": "neutral", "score": 0.10},
        {"label": "negative", "score": 0.05},
    ]]
    mock_fn.return_value = mock_pipeline

    result = score_headline("Apple hits record high on strong earnings beat")

    assert isinstance(result, SentimentResult)
    assert result.label == "positive"
    assert abs(result.positive + result.negative + result.neutral - 1.0) < 0.01


@patch("src.ingestion.sentiment_processor._get_finbert_pipeline")
def test_score_articles_returns_article_sentiment_pairs(mock_fn):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [[
        {"label": "neutral", "score": 0.70},
        {"label": "positive", "score": 0.20},
        {"label": "negative", "score": 0.10},
    ]]
    mock_fn.return_value = mock_pipeline

    articles = [_article("Apple launches new product line")]
    pairs = score_articles(articles)

    assert len(pairs) == 1
    assert pairs[0][0] is articles[0]
    assert isinstance(pairs[0][1], SentimentResult)


def test_score_headline_falls_back_to_vader_on_short_text():
    result = score_headline("up")
    assert isinstance(result, SentimentResult)
    assert result.label in {"positive", "negative", "neutral"}
