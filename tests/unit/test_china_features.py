from pathlib import Path

import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS
from scripts.build_features import _STOCK_TICKERS_CHINA

_CHINA_FEATURES_DIR = get_market("china").data_root / "features"
_COVERAGE_FLOOR = 0.90
_NULL_RATE_CEILING = 0.65


def _existing_feature_paths() -> list[tuple[str, Path]]:
    return [
        (ticker, _CHINA_FEATURES_DIR / f"{ticker}.parquet")
        for ticker in _STOCK_TICKERS_CHINA
        if (_CHINA_FEATURES_DIR / f"{ticker}.parquet").exists()
    ]


def test_china_feature_coverage_meets_floor():
    existing = _existing_feature_paths()
    coverage = len(existing) / len(_STOCK_TICKERS_CHINA)
    assert coverage >= _COVERAGE_FLOOR, (
        f"only {len(existing)}/{len(_STOCK_TICKERS_CHINA)} "
        f"({coverage:.1%}) tickers have feature files, floor is {_COVERAGE_FLOOR:.0%}"
    )


def test_china_features_have_full_schema_and_no_null_labels():
    existing = _existing_feature_paths()
    assert len(existing) > 0
    for ticker, path in existing:
        df = pl.read_parquet(path)
        for col in FEATURE_COLS:
            assert col in df.columns, f"{ticker} missing feature col {col}"
        assert "label" in df.columns
        assert "forward_return_5d" in df.columns
        assert df["label"].null_count() == 0
        assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
        assert len(df) > 0


def test_china_features_have_no_weekend_timestamps():
    # Regression guard for the timezone bug (fixed in the prior plan) that
    # shifted every China date back one calendar day, landing bars on
    # Sunday. Chinese exchanges trade Monday-Friday only.
    existing = _existing_feature_paths()
    for ticker, path in existing:
        df = pl.read_parquet(path)
        weekdays = set(df.select(pl.col("time").dt.weekday()).to_series().unique().to_list())
        assert weekdays.issubset({1, 2, 3, 4, 5}), f"{ticker} has weekend timestamps: {weekdays}"


def test_china_benchmark_derived_columns_null_rate_is_bounded():
    # vix_level and rel_strength_spy both derive from the benchmark join;
    # tickers listed before the benchmark's own history start will have
    # leading nulls in both — expected and bounded, not a defect.
    existing = _existing_feature_paths()
    for ticker, path in existing:
        df = pl.read_parquet(path)
        for col in ("vix_level", "rel_strength_spy"):
            null_rate = df[col].null_count() / len(df)
            assert null_rate <= _NULL_RATE_CEILING, f"{ticker}.{col} null rate {null_rate:.1%} exceeds ceiling"
