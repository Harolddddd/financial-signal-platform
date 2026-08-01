import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS
from scripts.build_features import _STOCK_TICKERS_CHINA

_CHINA_FEATURES_DIR = get_market("china").data_root / "features"


def test_china_pilot_feature_files_exist_for_every_ticker():
    for ticker in _STOCK_TICKERS_CHINA:
        assert (_CHINA_FEATURES_DIR / f"{ticker}.parquet").exists(), f"missing {ticker}"


def test_china_pilot_features_have_full_schema_and_no_null_labels():
    for ticker in _STOCK_TICKERS_CHINA:
        df = pl.read_parquet(_CHINA_FEATURES_DIR / f"{ticker}.parquet")
        for col in FEATURE_COLS:
            assert col in df.columns, f"{ticker} missing feature col {col}"
        assert "label" in df.columns
        assert "forward_return_5d" in df.columns
        assert df["label"].null_count() == 0
        assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
        assert len(df) > 0


def test_china_pilot_features_have_no_weekend_timestamps():
    # Regression guard for the timezone bug that shifted every China date
    # back one calendar day, landing bars on Sunday. Chinese exchanges
    # trade Monday-Friday only.
    for ticker in _STOCK_TICKERS_CHINA:
        df = pl.read_parquet(_CHINA_FEATURES_DIR / f"{ticker}.parquet")
        weekdays = set(df.select(pl.col("time").dt.weekday()).to_series().unique().to_list())
        assert weekdays.issubset({1, 2, 3, 4, 5}), f"{ticker} has weekend timestamps: {weekdays}"


def test_china_pilot_benchmark_derived_columns_null_rate_is_bounded():
    # vix_level and rel_strength_spy both derive from the benchmark join;
    # tickers listed before the benchmark's own history start will have
    # leading nulls in both. This is expected and bounded, not a defect —
    # but a ceiling catches a regression back to a short-history benchmark.
    # Measured worst observed rate after switching to 510300.SS: 54.7%
    # (600887.SS, vix_level).
    _NULL_RATE_CEILING = 0.65
    for ticker in _STOCK_TICKERS_CHINA:
        df = pl.read_parquet(_CHINA_FEATURES_DIR / f"{ticker}.parquet")
        for col in ("vix_level", "rel_strength_spy"):
            null_rate = df[col].null_count() / len(df)
            assert null_rate <= _NULL_RATE_CEILING, f"{ticker}.{col} null rate {null_rate:.1%} exceeds ceiling"
