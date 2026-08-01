from datetime import datetime, timezone
import logging
from pathlib import Path

import polars as pl

from config.markets import get_market
from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "TSM", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM",
    "TXN", "NOW", "AMAT", "MU", "LRCX",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC",
    "GS", "MS", "BLK", "AXP", "C",
    # Healthcare
    "UNH", "LLY", "JNJ", "PFE", "ABT",
    "MRK", "AMGN", "TMO", "ISRG", "GILD",
    # Consumer & Retail
    "WMT", "HD", "COST", "PG", "KO",
    "PEP", "NKE", "MCD", "SBUX", "TGT",
    # Energy & Industrials
    "XOM", "CVX", "CAT", "HON", "RTX",
    "NEE", "LIN", "DE", "UPS", "GE",
    # Communication & Media
    "DIS", "NFLX", "T", "VZ", "CMCSA",
    # Other large-caps
    "BRK-B", "BX",
    # Cybersecurity & Cloud
    "PANW", "CRWD", "SNOW", "PLTR", "ZS",
    # Financials (extended)
    "SCHW", "COF", "USB", "ICE", "SPGI",
    # Healthcare (extended)
    "REGN", "VRTX", "CVS", "CI", "ZBH",
    # Consumer (extended)
    "LOW", "LULU", "ROST",
    # Energy (extended)
    "SLB", "EOG",
    # Industrials / Defense / Transport
    "LMT", "NOC", "FDX",
    # Semiconductors
    "MRVL", "ARM",
    # Sector & Index ETFs
    "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "GLD", "TLT",
    # Healthcare (new)
    "ABBV", "MDT", "BSX", "HUM", "SYK",
    # Financials (new)
    "CME", "PRU", "ADP", "TROW", "TRV",
    # Energy / Industrials (new)
    "OXY", "MPC", "PSX", "HAL", "NSC",
    # Industrials (new)
    "MMM", "ITW", "SWK", "GWW", "PH",
    # Consumer Discretionary (new)
    "BKNG", "TJX", "TDG", "DAL", "HLT",
    # Consumer Staples (new)
    "KMB", "PM", "KR", "SYY", "YUM",
    # Real Estate / Utilities (new)
    "PLD", "SPG", "SO", "DUK", "XEL",
    # Tech / Communication (new)
    "CSCO", "ACN", "MSCI", "F", "TMUS",
    # Diversified (new)
    "BA", "WM", "ECL", "DHR", "AMT",
    "MMC", "PGR", "SHW", "UNP", "PSA",
]
BENCHMARK_TICKERS: list[str] = ["SPY", "^VIX"]

START = datetime(1980, 1, 1, tzinfo=timezone.utc)
END   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

_OUTPUT_DIR = get_market("us").data_root / "raw" / "ohlcv"


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
