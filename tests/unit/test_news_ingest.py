from datetime import date, datetime, timezone

import polars as pl

from src.ingestion import news_ingest
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult


def _art(ticker, n):
    return NewsArticle(
        ticker=ticker, headline=f"{ticker} headline number {n}", body="secret body text",
        source="x", url=f"https://example.com/{ticker}/{n}",
        published_at=datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc),
    )


def _fake_score(texts, batch_size=32):
    return [SentimentResult("positive", 0.8, 0.1, 0.1, "finbert") for _ in texts]


def _days():
    return pl.Series("t", [date(2026, 10, d) for d in range(19, 24)])


def test_ingest_skips_ticker_after_429(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ingest, "score_texts", _fake_score)
    sleeps = []

    def fetch(ticker, f, t):
        if ticker == "MSFT":
            raise RuntimeError("429 Too Many Requests")
        return [_art(ticker, 1), _art(ticker, 2)]

    res = news_ingest.ingest_news(
        ["AAPL", "MSFT"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
        fetch=fetch, sleep=sleeps.append,
    )
    assert res["tickers_failed"] == ["MSFT"]
    assert res["tickers_ok"] == ["AAPL"]
    assert res["articles_new"] == 2
    assert 60 in sleeps


def test_ingest_never_writes_body_text(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ingest, "score_texts", _fake_score)
    news_ingest.ingest_news(
        ["AAPL"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
        fetch=lambda t, f, to: [_art(t, 1)], sleep=lambda s: None,
    )
    files = list(tmp_path.glob("*.parquet"))
    assert files
    for f in files:
        cols = set(pl.read_parquet(f).columns)
        assert not cols & {"body", "summary", "url"}
    assert (tmp_path / "sentiment_daily.parquet").exists()


def test_ingest_missing_key_returns_empty(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(news_ingest.settings, "FINNHUB_KEY", "")
    with caplog.at_level("WARNING"):
        res = news_ingest.ingest_news(
            ["AAPL"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days()
        )
    assert res == {"tickers_ok": [], "tickers_failed": [], "articles_new": 0}
    assert "FINNHUB_KEY" in caplog.text
