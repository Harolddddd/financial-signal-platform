from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.ingestion import news_store
from src.ingestion.cn_news_ingest import fetch_cn_stock_news, ingest_cn_news
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult

SH = ZoneInfo("Asia/Shanghai")
TODAY = date(2026, 11, 10)
DAYS = pl.Series([date(2026, 11, d) for d in (9, 10, 11)], dtype=pl.Date)


def art(ticker, url, hh=10, mm=0, day=10):
    return NewsArticle(ticker, f"标题 {url}", "内容", "em", url,
                       datetime(2026, 11, day, hh, mm, tzinfo=SH).astimezone(timezone.utc))


def scorer(texts):
    return [SentimentResult("positive", 0.8, 0.1, 0.1, "fake")] * len(texts)


def run(tickers, fetch, tmp_path):
    return ingest_cn_news(tickers, tmp_path, DAYS, fetch=fetch, sleep=lambda s: None,
                          today=TODAY, scorer=scorer)


def test_dedup_by_url_hash_across_runs(tmp_path):
    fetch = lambda t: [art(t, "https://x/1"), art(t, "https://x/2")]
    assert run(["000021.SZ"], fetch, tmp_path)["articles_new"] == 2
    assert run(["000021.SZ"], fetch, tmp_path)["articles_new"] == 0


def test_fetch_error_marks_failed_and_continues(tmp_path):
    def fetch(t):
        if t == "000009.SZ":
            raise RuntimeError("boom")
        return [art(t, "https://x/1")]
    res = run(["000009.SZ", "000021.SZ"], fetch, tmp_path)
    assert res["tickers_failed"] == ["000009.SZ"] and res["tickers_ok"] == ["000021.SZ"]


def test_zero_articles_is_ok(tmp_path):
    res = run(["000009.SZ"], lambda t: [], tmp_path)
    assert res["tickers_ok"] == ["000009.SZ"] and res["articles_new"] == 0


def test_cap_30_per_shanghai_day(tmp_path):
    fetch = lambda t: [art(t, f"https://x/{i}", hh=9, mm=i) for i in range(35)]
    run(["000021.SZ"], fetch, tmp_path)
    files = list(tmp_path.glob("articles_scores_*.parquet"))
    scores = pl.concat([pl.read_parquet(f) for f in files])
    assert (scores["scorer"] == "unscored").sum() == 5
    daily = news_store.load_daily(tmp_path)
    assert daily["article_count"].to_list() == [35]


def test_daily_file_uses_1500_cutoff(tmp_path):
    run(["000021.SZ"], lambda t: [art(t, "https://x/late", hh=15, mm=30)], tmp_path)
    assert news_store.load_daily(tmp_path)["time"].to_list() == [date(2026, 11, 11)]


def test_fetch_parses_akshare_frame():
    import pandas as pd
    df = pd.DataFrame([{"新闻标题": "t", "新闻内容": "c", "发布时间": "2026-11-05 21:13:00",
                        "文章来源": "证券时报网", "新闻链接": "http://e/1"}])
    out = fetch_cn_stock_news("000009.SZ", get=lambda code: df)
    assert out[0].ticker == "000009.SZ"
    assert out[0].published_at == datetime(2026, 11, 5, 13, 13, tzinfo=timezone.utc)
