from datetime import datetime, timezone
import logging
from pathlib import Path

import polars as pl

from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "AVGO", "TSM",
    "JPM", "LLY", "V", "WMT", "UNH",
    "XOM", "MA", "ORCL", "JNJ", "HD",
]
BENCHMARK_TICKERS: list[str] = ["SPY", "^VIX"]

START = datetime(2000, 1, 1, tzinfo=timezone.utc)
END   = datetime(2020, 12, 31, tzinfo=timezone.utc)

_OUTPUT_DIR = Path("data/raw/ohlcv")


def scrape_ticker(
    ticker: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> tuple[int, str | None]:
    try:
        df = fetch_ohlcv(ticker, start, end)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_dir / f"{ticker}.parquet")
        return len(df), None
    except Exception as exc:
        return 0, str(exc)


def main() -> None:
    all_tickers = TICKERS + BENCHMARK_TICKERS
    successes: list[tuple[str, int]] = []
    failures:  list[tuple[str, str]] = []

    for ticker in all_tickers:
        logger.info("Fetching %s ...", ticker)
        rows, err = scrape_ticker(ticker, START, END, _OUTPUT_DIR)
        if err:
            logger.warning("FAILED %s: %s", ticker, err)
            failures.append((ticker, err))
        else:
            logger.info("OK     %s — %d rows", ticker, rows)
            successes.append((ticker, rows))

    print(f"\n{'='*50}")
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    for ticker, rows in successes:
        print(f"  OK    {ticker:10s}  {rows:>6,} rows")
    for ticker, err in failures:
        print(f"  FAIL  {ticker:10s}  {err}")


if __name__ == "__main__":
    main()
