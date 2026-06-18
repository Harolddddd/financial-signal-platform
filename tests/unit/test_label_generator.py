from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.label_generator import add_labels


def _make_df(closes: list[float]) -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame({
        "time":   [base + timedelta(days=i) for i in range(len(closes))],
        "ticker": ["AAPL"] * len(closes),
        "close":  closes,
    })


def test_forward_return_correct():
    df = add_labels(_make_df([100.0, 101.0, 102.0, 103.0, 104.0, 106.0]))
    assert abs(df["forward_return_5d"][0] - 0.06) < 1e-6


def test_last_five_rows_are_null():
    df = add_labels(_make_df([100.0] * 10))
    assert df["label"][-5:].is_null().all()
    assert df["forward_return_5d"][-5:].is_null().all()


def test_buy_label_when_return_above_threshold():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 103.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Buy"


def test_sell_label_when_return_below_threshold():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 97.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Sell"


def test_hold_label_when_return_within_thresholds():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 101.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Hold"


def test_custom_forward_days():
    closes = [100.0, 100.0, 100.0, 105.0]
    df = add_labels(_make_df(closes), forward_days=3, buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Buy"
    assert df["label"][-3:].is_null().all()
