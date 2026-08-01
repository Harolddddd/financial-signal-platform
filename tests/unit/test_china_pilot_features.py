import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS

_CHINA_FEATURES_DIR = get_market("china").data_root / "features"

_PILOT_TICKERS = [
    "600519.SS", "601318.SS", "600036.SS", "601398.SS", "000858.SZ",
    "000333.SZ", "002594.SZ", "300750.SZ", "600887.SS", "601012.SS",
    "002415.SZ", "300059.SZ", "601888.SS", "600030.SS", "000651.SZ",
]


def test_china_pilot_feature_files_exist_for_every_ticker():
    for ticker in _PILOT_TICKERS:
        assert (_CHINA_FEATURES_DIR / f"{ticker}.parquet").exists(), f"missing {ticker}"


def test_china_pilot_features_have_full_schema_and_no_null_labels():
    for ticker in _PILOT_TICKERS:
        df = pl.read_parquet(_CHINA_FEATURES_DIR / f"{ticker}.parquet")
        for col in FEATURE_COLS:
            assert col in df.columns, f"{ticker} missing feature col {col}"
        assert "label" in df.columns
        assert "forward_return_5d" in df.columns
        assert df["label"].null_count() == 0
        assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
        assert len(df) > 0
