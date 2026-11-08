from datetime import date, datetime, timezone

import polars as pl
import pytest

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


def test_reingest_never_rescores(tmp_path, monkeypatch):
    calls = []

    def counting_score(texts, batch_size=32):
        calls.append(len(texts))
        return _fake_score(texts)

    monkeypatch.setattr(news_ingest, "score_texts", counting_score)
    kw = dict(fetch=lambda t, f, to: [_art(t, 1), _art(t, 2)], sleep=lambda s: None)
    args = (["AAPL", "MSFT"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days())
    first = news_ingest.ingest_news(*args, **kw)
    assert first["articles_new"] == 4
    assert sum(calls) == 4
    calls.clear()
    second = news_ingest.ingest_news(*args, **kw)
    assert calls == []
    assert second["articles_new"] == 0
    assert second["tickers_ok"] == ["AAPL", "MSFT"]


def test_ingest_dedupes_same_url_across_chunks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        news_ingest, "score_texts", lambda texts, batch_size=32: calls.append(len(texts)) or _fake_score(texts)
    )
    # range spans two month chunks; fetch returns the same article for each
    res = news_ingest.ingest_news(
        ["AAPL"], date(2026, 9, 29), date(2026, 10, 22), tmp_path, _days(),
        fetch=lambda t, f, to: [_art(t, 1)], sleep=lambda s: None,
    )
    assert calls == [1]
    assert res["articles_new"] == 1


def test_ingest_writes_per_ticker(tmp_path, monkeypatch):
    def score(texts, batch_size=32):
        if any(t.startswith("MSFT") for t in texts):
            raise RuntimeError("boom")
        return _fake_score(texts)

    monkeypatch.setattr(news_ingest, "score_texts", score)
    with pytest.raises(RuntimeError):
        news_ingest.ingest_news(
            ["AAPL", "MSFT"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
            fetch=lambda t, f, to: [_art(t, 1)], sleep=lambda s: None,
            today=date(2026, 10, 23),
        )
    files = list(tmp_path.glob("articles_scores_*.parquet"))
    assert files
    stored = pl.concat([pl.read_parquet(f) for f in files])
    assert stored["ticker"].cast(pl.Utf8).to_list() == ["AAPL"]
    heads = pl.read_parquet(tmp_path / "headlines_recent.parquet")
    assert heads["ticker"].cast(pl.Utf8).to_list() == ["AAPL"]


def test_ingest_uses_given_today_for_headlines_and_prune(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ingest, "score_texts", _fake_score)
    seen = {}
    real_wh, real_prune = news_ingest.news_store.write_headlines, news_ingest.news_store.prune

    def wh(news_dir, rows, today, keep_days=14):
        seen["headlines"] = today
        return real_wh(news_dir, rows, today=today, keep_days=keep_days)

    def pr(news_dir, today, keep_days=60):
        seen["prune"] = today
        return real_prune(news_dir, today=today, keep_days=keep_days)

    monkeypatch.setattr(news_ingest.news_store, "write_headlines", wh)
    monkeypatch.setattr(news_ingest.news_store, "prune", pr)
    news_ingest.ingest_news(
        ["AAPL"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
        fetch=lambda t, f, to: [_art(t, 1)], sleep=lambda s: None, today=date(2026, 10, 23),
    )
    assert seen == {"headlines": date(2026, 10, 23), "prune": date(2026, 10, 23)}


def test_ingest_missing_key_returns_empty(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(news_ingest.settings, "FINNHUB_KEY", "")
    with caplog.at_level("WARNING"):
        res = news_ingest.ingest_news(
            ["AAPL"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days()
        )
    assert res == {"tickers_ok": [], "tickers_failed": [], "articles_new": 0}
    assert "FINNHUB_KEY" in caplog.text


def test_interrupted_ingest_keeps_daily_rows_from_checkpoint(tmp_path, monkeypatch):
    # A long backfill killed mid-run must not lose the tickers it already finished.
    monkeypatch.setattr(news_ingest, "score_texts", _fake_score)
    monkeypatch.setattr(news_ingest, "_REBUILD_EVERY", 2)

    def fetch(ticker, f, t):
        if ticker == "CCC":
            raise KeyboardInterrupt  # stands in for the job being killed
        return [_art(ticker, 1)]

    with pytest.raises(KeyboardInterrupt):
        news_ingest.ingest_news(
            ["AAA", "BBB", "CCC"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
            fetch=fetch, sleep=lambda s: None,
        )
    daily = pl.read_parquet(tmp_path / "sentiment_daily.parquet")
    assert set(daily["ticker"].cast(pl.Utf8)) == {"AAA", "BBB"}


def test_truncated_responses_are_split_until_complete(tmp_path, monkeypatch):
    # Finnhub caps each response (~250, newest first); busy tickers must be re-fetched in smaller ranges.
    from datetime import timedelta
    monkeypatch.setattr(news_ingest, "score_texts", _fake_score)
    monkeypatch.setattr(news_ingest, "_TRUNCATED_AT", 5)
    per_day = 3
    start, end = date(2026, 10, 1), date(2026, 10, 10)
    calls = []

    def fetch(ticker, f, t):
        calls.append((f.date(), t.date()))
        out = []
        d = f.date()
        while d <= t.date():
            for n in range(per_day):
                a = _art(ticker, f"{d}-{n}")
                a.published_at = datetime(d.year, d.month, d.day, 15, tzinfo=timezone.utc)
                out.append(a)
            d += timedelta(days=1)
        return sorted(out, key=lambda a: a.published_at, reverse=True)[:5]  # the cap

    days = pl.Series("t", [date(2026, 10, d) for d in range(1, 15)])
    res = news_ingest.ingest_news(["AAPL"], start, end, tmp_path, days, fetch=fetch, sleep=lambda s: None)
    assert res["articles_new"] == 10 * per_day
    assert all(f == t for f, t in calls if (t - f).days == 0) and len(calls) > 1


def test_scores_at_most_cap_per_ticker_day_but_counts_all(tmp_path, monkeypatch):
    scored = []

    def counting_score(texts, batch_size=32):
        scored.extend(texts)
        return _fake_score(texts)

    monkeypatch.setattr(news_ingest, "score_texts", counting_score)
    monkeypatch.setattr(news_ingest, "_MAX_SCORED_PER_DAY", 4)
    arts = [_art("AAPL", n) for n in range(10)]  # all on one day
    news_ingest.ingest_news(
        ["AAPL"], date(2026, 10, 20), date(2026, 10, 22), tmp_path, _days(),
        fetch=lambda t, f, to: arts, sleep=lambda s: None,
    )
    assert len(scored) == 4
    daily = pl.read_parquet(tmp_path / "sentiment_daily.parquet")
    assert daily["article_count"].to_list() == [10]
    assert abs(daily["avg_pos"][0] - 0.8) < 1e-6  # mean over the scored sample only
