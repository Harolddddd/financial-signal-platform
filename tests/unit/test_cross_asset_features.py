from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.cross_asset_features import add_cross_asset_features


def _df(closes: list[float], ticker: str = "AAPL") -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(len(closes))],
        "ticker": [ticker] * len(closes),
        "open":   closes,
        "high":   closes,
        "low":    closes,
        "close":  closes,
        "volume": [1_000_000] * len(closes),
        "adj_close": closes,
        "dividends": [0.0] * len(closes),
        "stock_splits": [0.0] * len(closes),
    })


def test_cross_asset_features_adds_rel_strength_and_vix():
    n = 20
    stock = _df([100.0 + i for i in range(n)])
    spy   = _df([400.0 + i * 0.5 for i in range(n)], ticker="SPY")
    vix   = _df([15.0 + i * 0.1 for i in range(n)], ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)

    assert "rel_strength_spy" in result.columns
    assert "vix_level" in result.columns
    assert len(result) == n


def test_rel_strength_is_null_for_first_five_rows():
    n = 20
    stock = _df([100.0 + i for i in range(n)])
    spy   = _df([400.0 + i * 0.5 for i in range(n)], ticker="SPY")
    vix   = _df([15.0] * n, ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)
    assert result["rel_strength_spy"][:5].is_null().all()


def test_rel_strength_clamped():
    n = 20
    stock = _df([100.0 + i * 5 for i in range(n)])       # fast mover
    spy   = _df([400.0 + i * 0.001 for i in range(n)], ticker="SPY")  # nearly flat
    vix   = _df([15.0] * n, ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)
    valid = result["rel_strength_spy"].drop_nulls()
    assert (valid <= 5.0).all()
    assert (valid >= -5.0).all()


def test_missing_vix_date_becomes_null():
    stock = _df([100.0, 101.0, 102.0])
    spy   = _df([400.0, 401.0, 402.0], ticker="SPY")
    vix   = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker": ["^VIX"], "open": [15.0], "high": [15.0],
        "low": [15.0], "close": [15.0], "volume": [0],
        "adj_close": [15.0], "dividends": [0.0], "stock_splits": [0.0],
    })
    result = add_cross_asset_features(stock, spy, vix)
    assert result["vix_level"][1] is None or result["vix_level"].is_null()[1]


def test_synthetic_vol_index_returns_time_and_close_columns():
    import numpy as np
    from src.features.cross_asset_features import synthetic_vol_index

    n = 60
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist()
    benchmark = _df(closes, ticker="000300.SS")

    result = synthetic_vol_index(benchmark, window=21)

    assert set(result.columns) == {"time", "close"}
    assert 0 < len(result) < n
    assert result["close"].null_count() == 0
    assert (result["close"] > 0).all()
    assert (result["close"] < 200).all()  # sane VIX-style upper bound
