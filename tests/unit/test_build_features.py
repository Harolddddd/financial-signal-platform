from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest


def _make_ohlcv(ticker: str, n: int = 300) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base + np.cumsum(rng.normal(0, 1, n))
    times = [
        datetime(2010, 1, 1, tzinfo=timezone.utc).replace(
            year=2010 + i // 252, month=1 + (i % 252) // 21, day=1 + i % 21
        )
        for i in range(n)
    ]
    return pl.DataFrame({
        "time":         times,
        "ticker":       [ticker] * n,
        "open":         closes * 0.99,
        "high":         closes * 1.01,
        "low":          closes * 0.98,
        "close":        closes,
        "volume":       [1_000_000] * n,
        "adj_close":    closes,
        "dividends":    [0.0] * n,
        "stock_splits": [0.0] * n,
    })


def test_add_neutral_sentiment_adds_all_columns():
    from scripts.build_features import add_neutral_sentiment
    df = pl.DataFrame({"time": [datetime(2010, 1, 4, tzinfo=timezone.utc)], "ticker": ["AAPL"]})
    out = add_neutral_sentiment(df)
    assert "sent_pos_avg_3d"  in out.columns
    assert "sent_pos_avg_5d"  in out.columns
    assert "sent_pos_avg_10d" in out.columns
    assert "sent_pos_mom_3d"  in out.columns
    assert "news_vol_spike"   in out.columns
    assert out["sent_pos_avg_5d"][0] == pytest.approx(0.5)
    assert out["sent_pos_mom_3d"][0] == pytest.approx(0.0)
    assert out["news_vol_spike"][0] == 0


def test_build_features_for_ticker_returns_required_cols(tmp_path):
    from scripts.build_features import build_features_for_ticker
    from dashboard.config import FEATURE_COLS

    aapl = _make_ohlcv("AAPL")
    spy  = _make_ohlcv("SPY")
    vix  = _make_ohlcv("^VIX")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    aapl.write_parquet(raw_dir / "AAPL.parquet")

    df = build_features_for_ticker("AAPL", raw_dir, spy, vix)

    assert "label" in df.columns
    assert "forward_return_5d" in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature col: {col}"
    assert df["label"].null_count() == 0


def test_build_features_for_ticker_has_no_null_labels(tmp_path):
    from scripts.build_features import build_features_for_ticker

    aapl = _make_ohlcv("AAPL")
    spy  = _make_ohlcv("SPY")
    vix  = _make_ohlcv("^VIX")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    aapl.write_parquet(raw_dir / "AAPL.parquet")

    df = build_features_for_ticker("AAPL", raw_dir, spy, vix)
    assert df["label"].null_count() == 0
    assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
