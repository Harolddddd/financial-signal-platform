from pathlib import Path
import logging

import polars as pl

from src.features.technical_indicators import add_technical_indicators
from src.features.cross_asset_features import add_cross_asset_features
from src.features.label_generator import add_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR     = Path("data/raw/ohlcv")
_FEATURE_DIR = Path("data/features")

_STOCK_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "AVGO", "TSM",
    "JPM", "LLY", "V", "WMT", "UNH",
    "XOM", "MA", "ORCL", "JNJ", "HD",
]


def add_neutral_sentiment(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.lit(0.5).alias("sent_pos_avg_3d"),
        pl.lit(0.5).alias("sent_pos_avg_5d"),
        pl.lit(0.5).alias("sent_pos_avg_10d"),
        pl.lit(0.0).alias("sent_pos_mom_3d"),
        pl.lit(0).cast(pl.Int64).alias("news_vol_spike"),
    ])


def build_features_for_ticker(
    ticker: str,
    raw_dir: Path,
    spy_df: pl.DataFrame,
    vix_df: pl.DataFrame,
) -> pl.DataFrame:
    df = pl.read_parquet(raw_dir / f"{ticker}.parquet")
    df = add_technical_indicators(df)
    df = add_cross_asset_features(df, spy_df, vix_df)
    df = add_neutral_sentiment(df)
    df = add_labels(df)
    return df.drop_nulls(subset=["label"])


def main() -> None:
    spy_path = _RAW_DIR / "SPY.parquet"
    vix_path = _RAW_DIR / "^VIX.parquet"

    if not spy_path.exists() or not vix_path.exists():
        raise FileNotFoundError(
            "SPY.parquet or ^VIX.parquet missing from data/raw/ohlcv/. "
            "Run scripts/scrape_top20.py first."
        )

    spy_df = pl.read_parquet(spy_path)
    vix_df = pl.read_parquet(vix_path)

    _FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    successes: list[tuple[str, int]] = []
    failures:  list[tuple[str, str]] = []

    for ticker in _STOCK_TICKERS:
        raw_path = _RAW_DIR / f"{ticker}.parquet"
        if not raw_path.exists():
            logger.warning("Skipping %s — raw parquet not found", ticker)
            failures.append((ticker, "raw parquet not found"))
            continue
        try:
            df = build_features_for_ticker(ticker, _RAW_DIR, spy_df, vix_df)
            out_path = _FEATURE_DIR / f"{ticker}.parquet"
            df.write_parquet(out_path)
            logger.info("OK    %s — %d rows → %s", ticker, len(df), out_path)
            successes.append((ticker, len(df)))
        except Exception as exc:
            logger.warning("FAILED %s: %s", ticker, exc)
            failures.append((ticker, str(exc)))

    print(f"\n{'='*50}")
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    for ticker, rows in successes:
        print(f"  OK    {ticker:10s}  {rows:>6,} rows")
    for ticker, err in failures:
        print(f"  FAIL  {ticker:10s}  {err}")


if __name__ == "__main__":
    main()
