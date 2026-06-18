from datetime import date, datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.sentiment_features import aggregate_daily_sentiment, add_sentiment_features


def _make_raw_sentiment(n_days: int = 30, articles_per_day: int = 3) -> pl.DataFrame:
    rows = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for d in range(n_days):
        for _ in range(articles_per_day):
            rows.append({
                "published_at": base + timedelta(days=d),
                "ticker": "AAPL",
                "sentiment_pos": 0.6,
                "sentiment_neg": 0.2,
                "sentiment_neu": 0.2,
            })
    return pl.DataFrame(rows)


def _make_daily_sentiment(n: int = 30) -> pl.DataFrame:
    base = date(2024, 1, 1)
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "avg_pos": [0.6] * n,
        "avg_neg": [0.2] * n,
        "avg_neu": [0.2] * n,
        "article_count": [3] * n,
    })


def test_aggregate_daily_sentiment_one_row_per_day():
    raw = _make_raw_sentiment(n_days=5, articles_per_day=3)
    daily = aggregate_daily_sentiment(raw)
    assert len(daily) == 5


def test_aggregate_daily_sentiment_avg_pos_correct():
    raw = _make_raw_sentiment(n_days=1, articles_per_day=2)
    daily = aggregate_daily_sentiment(raw)
    assert abs(daily["avg_pos"][0] - 0.6) < 1e-6


def test_add_sentiment_features_adds_rolling_columns():
    df = add_sentiment_features(_make_daily_sentiment(30))
    for col in ["sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
                "sent_pos_mom_3d", "news_vol_spike"]:
        assert col in df.columns, f"Missing: {col}"


def test_rolling_averages_converge_on_constant_sentiment():
    df = add_sentiment_features(_make_daily_sentiment(30))
    last = df.row(-1, named=True)
    assert abs(last["sent_pos_avg_3d"] - 0.6) < 1e-6
    assert abs(last["sent_pos_avg_5d"] - 0.6) < 1e-6
    assert abs(last["sent_pos_avg_10d"] - 0.6) < 1e-6


def test_momentum_is_zero_on_constant_sentiment():
    df = add_sentiment_features(_make_daily_sentiment(30))
    last = df.row(-1, named=True)
    assert abs(last["sent_pos_mom_3d"]) < 1e-6
