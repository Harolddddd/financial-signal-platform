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
    from dashboard.ui_config import FEATURE_COLS

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


def test_tickers_by_market_has_expected_markets():
    from scripts.build_features import _STOCK_TICKERS, _STOCK_TICKERS_CHINA, _TICKERS_BY_MARKET
    assert _TICKERS_BY_MARKET["us"] == _STOCK_TICKERS
    assert _TICKERS_BY_MARKET["china"] == _STOCK_TICKERS_CHINA
    assert len(_STOCK_TICKERS_CHINA) == 500
    assert all(t.endswith(".SS") or t.endswith(".SZ") for t in _STOCK_TICKERS_CHINA)
    assert len(set(_STOCK_TICKERS_CHINA)) == 500  # no duplicates


def test_build_live_features_default_market_is_us():
    import inspect
    from scripts.build_features import build_live_features
    sig = inspect.signature(build_live_features)
    assert sig.parameters["market"].default == "us"


def test_build_live_features_china_reads_from_china_raw_dir_and_tickers(tmp_path, monkeypatch):
    from scripts.build_features import build_live_features, _STOCK_TICKERS_CHINA
    from config.markets import MARKETS

    # Create synthetic OHLCV data for a few tickers in a temp directory
    # build_live_features looks for raw_dir at data_root / "raw" / "ohlcv"
    raw_dir = tmp_path / "raw" / "ohlcv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create data for the benchmark ticker and a couple of regular tickers
    benchmark_ticker = "510300.SS"
    test_tickers = [benchmark_ticker, "000009.SZ", "600000.SS"]

    for ticker in test_tickers:
        ohlcv_df = _make_ohlcv(ticker, n=100)
        ohlcv_df.write_parquet(raw_dir / f"{ticker}.parquet")

    # Monkeypatch get_market to return a modified config with our temp directory
    def mock_get_market(name):
        if name == "china":
            cfg = MARKETS["china"]
            # Return a new config with data_root pointing to our temp directory
            from config.markets import MarketConfig
            return MarketConfig(
                name=cfg.name,
                label=cfg.label,
                data_root=tmp_path,
                universe=cfg.universe,
                benchmark_ticker=cfg.benchmark_ticker,
                vol_index_ticker=cfg.vol_index_ticker,
                currency=cfg.currency,
            )
        return MARKETS[name]

    monkeypatch.setattr("scripts.build_features.get_market", mock_get_market)

    df = build_live_features(market="china")
    assert len(df) > 0
    assert set(df["ticker"].unique().to_list()).issubset(set(_STOCK_TICKERS_CHINA))
