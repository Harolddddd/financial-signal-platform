from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
import tempfile
import pytest

from src.features.feature_store import build_features, write_features, export_parquet


def _sample_ohlcv(n: int = 250) -> pl.DataFrame:
    import math
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    closes = [100.0 + 5 * math.sin(i / 10) + i * 0.1 for i in range(n)]
    return pl.DataFrame({
        "time":        [base + timedelta(days=i) for i in range(n)],
        "ticker":      ["AAPL"] * n,
        "open":        [c - 0.5 for c in closes],
        "high":        [c + 1.0 for c in closes],
        "low":         [c - 1.0 for c in closes],
        "close":       closes,
        "volume":      [1_000_000] * n,
        "adj_close":   closes,
        "dividends":   [0.0] * n,
        "stock_splits":[0.0] * n,
    })


def _sample_sentiment(n: int = 250) -> pl.DataFrame:
    from datetime import date
    base = date(2022, 1, 3)
    return pl.DataFrame({
        "time":          [base + timedelta(days=i) for i in range(n)],
        "ticker":        ["AAPL"] * n,
        "avg_pos":       [0.6] * n,
        "avg_neg":       [0.2] * n,
        "avg_neu":       [0.2] * n,
        "article_count": [3] * n,
    })


@patch("src.features.feature_store._load_ohlcv_from_db")
@patch("src.features.feature_store._load_sentiment_from_db")
@patch("src.features.feature_store._load_spy_from_db")
@patch("src.features.feature_store._load_vix_from_db")
def test_build_features_returns_dataframe_with_all_columns(
    mock_vix, mock_spy, mock_sent, mock_ohlcv
):
    ohlcv = _sample_ohlcv()
    mock_ohlcv.return_value = ohlcv
    mock_spy.return_value = ohlcv.with_columns(pl.lit("SPY").alias("ticker"))
    mock_vix.return_value = ohlcv.with_columns(pl.lit("^VIX").alias("ticker"))
    mock_sent.return_value = _sample_sentiment()

    start = datetime(2022, 1, 3, tzinfo=timezone.utc)
    end = datetime(2022, 12, 31, tzinfo=timezone.utc)
    df = build_features("AAPL", start, end)

    assert isinstance(df, pl.DataFrame)
    for col in ["sma_20", "rsi_14", "macd", "sent_pos_avg_5d", "label", "forward_return_5d"]:
        assert col in df.columns, f"Missing: {col}"


@patch("src.features.feature_store.get_connection", return_value=MagicMock())
@patch("src.features.feature_store.execute_many")
@patch("src.features.feature_store.release_connection")
def test_write_features_returns_row_count(mock_rel, mock_exec, mock_conn):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker": ["AAPL"],
        "sma_10": [150.0], "sma_20": [148.0], "sma_50": [145.0], "sma_200": [140.0],
        "ema_12": [151.0], "ema_26": [149.0],
        "rsi_14": [55.0], "macd": [1.2], "macd_signal": [1.0], "macd_hist": [0.2],
        "bb_upper": [160.0], "bb_lower": [140.0], "bb_width": [0.1],
        "atr_14": [2.5], "hist_vol_21": [0.15],
        "sent_pos_avg_3d": [0.6], "sent_pos_avg_5d": [0.58], "sent_pos_avg_10d": [0.57],
        "sent_pos_mom_3d": [0.02], "news_vol_spike": [0.5],
        "rel_strength_spy": [1.1], "vix_level": [15.0],
        "forward_return_5d": [0.03], "label": ["Buy"],
    })
    n = write_features(df)
    assert n == 1
    assert mock_exec.called


def test_export_parquet_writes_file():
    df = pl.DataFrame({"time": [datetime(2024, 1, 2, tzinfo=timezone.utc)], "ticker": ["AAPL"], "close": [153.0]})
    with tempfile.TemporaryDirectory() as tmp:
        path = export_parquet(df, "AAPL", Path(tmp))
        assert path.exists()
        assert path.name == "AAPL.parquet"
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
