from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.technical_indicators import add_technical_indicators

_INDICATOR_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14", "hist_vol_21",
]


def _make_ohlcv(n: int = 250) -> pl.DataFrame:
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    import math
    closes = [100.0 + 10 * math.sin(i / 20) + i * 0.05 for i in range(n)]
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "open":   [c - 0.5 for c in closes],
        "high":   [c + 1.0 for c in closes],
        "low":    [c - 1.0 for c in closes],
        "close":  closes,
        "volume": [1_000_000] * n,
    })


def test_all_indicator_columns_present():
    df = add_technical_indicators(_make_ohlcv())
    for col in _INDICATOR_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_row_count_unchanged():
    raw = _make_ohlcv()
    df = add_technical_indicators(raw)
    assert len(df) == len(raw)


def test_sma_20_is_rolling_mean_of_close():
    df = add_technical_indicators(_make_ohlcv())
    row_30 = df.row(30, named=True)
    manual_sma20 = df["close"][11:31].mean()
    assert abs(row_30["sma_20"] - manual_sma20) < 1e-6


def test_rsi_bounded_0_to_100():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["rsi_14"].drop_nulls()
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0


def test_bb_upper_above_bb_lower():
    df = add_technical_indicators(_make_ohlcv())
    valid = df.drop_nulls(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_atr_non_negative():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["atr_14"].drop_nulls()
    assert (valid >= 0).all()


def test_hist_vol_non_negative():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["hist_vol_21"].drop_nulls()
    assert (valid >= 0).all()
