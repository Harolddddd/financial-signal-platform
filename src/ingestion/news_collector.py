from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import feedparser
import finnhub
from newsapi import NewsApiClient

from config.settings import settings

logger = logging.getLogger(__name__)
_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


@dataclass
class NewsArticle:
    ticker: str | None
    headline: str
    body: str | None
    source: str
    url: str
    published_at: datetime


def collect_rss(
    ticker: str,
    max_items: int = 20,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> list[NewsArticle]:
    feed = feedparser.parse(_RSS_TEMPLATE.format(ticker=ticker))
    articles: list[NewsArticle] = []
    for entry in feed.entries[:max_items]:
        try:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if from_date is not None and ts < from_date:
                continue
            if to_date is not None and ts > to_date:
                continue
            articles.append(NewsArticle(
                ticker=ticker,
                headline=entry.get("title", ""),
                body=entry.get("summary"),
                source="yahoo_rss",
                url=entry.get("link", ""),
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("RSS parse error for %s: %s", ticker, e)
    return articles


def collect_newsapi(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    if not settings.NEWSAPI_KEY:
        return []
    client = NewsApiClient(api_key=settings.NEWSAPI_KEY)
    resp = client.get_everything(
        q=ticker,
        from_param=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )
    articles: list[NewsArticle] = []
    for a in resp.get("articles", []):
        try:
            ts = datetime.strptime(a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            articles.append(NewsArticle(
                ticker=ticker,
                headline=a["title"],
                body=a.get("content"),
                source=a.get("source", {}).get("name", "unknown"),
                url=a["url"],
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("NewsAPI parse error: %s", e)
    return articles


def collect_finnhub(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    if not settings.FINNHUB_KEY:
        return []
    client = finnhub.Client(api_key=settings.FINNHUB_KEY)
    raw = client.company_news(
        ticker,
        _from=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
    )
    articles: list[NewsArticle] = []
    for item in raw:
        try:
            ts = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
            articles.append(NewsArticle(
                ticker=ticker,
                headline=item["headline"],
                body=item.get("summary"),
                source=item.get("source", "finnhub"),
                url=item["url"],
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("Finnhub parse error: %s", e)
    return articles


def collect_all_news(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    seen: set[str] = set()
    result: list[NewsArticle] = []
    for article in (
        collect_rss(ticker, from_date=from_date, to_date=to_date)
        + collect_newsapi(ticker, from_date, to_date)
        + collect_finnhub(ticker, from_date, to_date)
    ):
        if article.url and article.url not in seen:
            seen.add(article.url)
            result.append(article)
    return result
