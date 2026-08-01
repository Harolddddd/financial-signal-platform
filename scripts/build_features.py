from pathlib import Path
import logging

import polars as pl

from config.markets import MARKETS, get_market
from src.features.technical_indicators import add_technical_indicators
from src.features.cross_asset_features import add_cross_asset_features, synthetic_vol_index
from src.features.label_generator import add_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR     = get_market("us").data_root / "raw" / "ohlcv"
_FEATURE_DIR = get_market("us").data_root / "features"

_STOCK_TICKERS: list[str] = [
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
    # ── S&P 500 expansion ──────────────────────────────────────────────────
    # Technology
    "ADSK", "AKAM", "ANSS", "APP", "CDNS", "CTSH", "DXCM", "EPAM",
    "FFIV", "FISV", "FTNT", "GEN", "HPE", "HPQ", "IBM", "JNPR", "KEYS",
    "KLAC", "LDOS", "MPWR", "NET", "NTAP", "NXPI", "ON", "PAYC", "PYPL",
    "SWKS", "TRMB", "TYL", "VRSN", "WDAY", "ZBRA", "ZM",
    # Healthcare
    "A", "ABC", "BAX", "BDX", "BIO", "BIIB", "CAH", "CNC", "CTLT",
    "DVA", "EW", "HCA", "HOLX", "HSIC", "IDXX", "IQV",
    "LH", "MCK", "MOH", "MRNA", "MTD", "PODD", "RVTY", "STE",
    "TFX", "VTRS", "ZTS", "COO", "CRL", "DGX", "INCY", "VEEV",
    # Financials
    "ACGL", "AFL", "AIG", "AJG", "AIZ", "ALL", "ALLY", "AMP", "AON",
    "BEN", "BK", "CBOE", "CFG", "CMA", "DFS", "FITB", "GL", "HBAN",
    "KEY", "MCO", "MET", "MKTX", "NTRS", "PFG", "PNC", "RE", "RJF",
    "STT", "UNM", "WRB", "WTW", "ZION", "CINF", "FIS", "FNF", "LNC",
    "NDAQ", "RGA", "SEIC", "SSNC", "VOYA",
    # Consumer Discretionary
    "AZO", "BBY", "CMG", "DECK", "DG", "DLTR", "DPZ", "ETSY",
    "FIVE", "GPC", "HAS", "LKQ", "LVS", "LYV", "MAR", "MGM",
    "MKC", "MNST", "MO", "NCLH", "NVR", "PHM", "PVH", "RCL",
    "SJM", "STZ", "TAP", "TPR", "ULTA", "VFC", "WHR", "WSM", "YUMC",
    "BWA", "CDW", "EL", "HRL", "HSY", "KHC", "KVUE", "SKX", "TXRH",
    # Energy
    "APA", "BKR", "CTRA", "DVN", "FANG", "HES", "MRO", "NOV",
    "PBF", "RRC", "TRGP", "VLO", "WMB",
    # Industrials
    "AME", "APTV", "AXON", "BR", "CARR", "DOV", "EMN", "EMR",
    "ETN", "FAST", "FLS", "GEV", "GNRC", "HII", "HWM", "IEX",
    "IR", "JCI", "KNX", "LII", "LSTR", "MAS", "MHK", "MLM",
    "NUE", "OC", "ODFL", "OTIS", "PCAR", "PPG", "PWR",
    "ROK", "RSG", "SNA", "TER", "TXT", "UBER", "URI",
    "VRSK", "WAB", "XYL", "ALB", "ALK",
    # REITs
    "AVB", "BXP", "CPT", "EQR", "ESS", "EXR", "FRT", "HST",
    "IRM", "KIM", "MAA", "NNN", "O", "REG", "UDR", "VTR", "WY",
    # Utilities
    "AEE", "AES", "ATO", "CNP", "D", "ETR", "EXC", "FE",
    "LNT", "NI", "NRG", "PCG", "SRE", "VST", "WEC",
    # Materials
    "APD", "CE", "CF", "CTVA", "DD", "FCX", "IP", "IFF",
    "LYB", "MOS", "NEM", "PKG", "RS", "STLD",
    # Communication Services
    "CHTR", "EA", "MTCH", "PARA", "PINS", "SNAP", "TTWO", "WBD",
    # ── Russell 1000 mid-caps ──────────────────────────────────────────────
    "AGCO", "AOS", "APO", "ARCC", "AWK", "AXS", "BAH", "BALL",
    "BBWI", "BG", "BURL", "CPB", "CSL", "CUBE", "EFX",
    "EQT", "EXPE", "FMC", "GLOB", "HIG", "INGR", "ITT",
    "JBHT", "K", "LAMR", "LEA", "LEN", "LPX", "LW", "MANH",
    "MELI", "MORN", "MTH", "MTZ", "NLY", "NTRA", "NVT", "NXST",
    "OKE", "POOL", "PRI", "QRVO", "REXR", "RMD", "RPRX", "RRX",
    "SBAC", "SEE", "SFM", "SKY", "SLGN", "SM", "SMAR",
    "SNX", "THC", "TOST", "TPL", "TPX", "TREX", "TRNO",
    "TWO", "USFD", "UTHR", "VICI", "VMI", "WING", "WPC",
    "CLX", "CMS", "CW", "DT", "ELAN", "ESAB", "EVR",
    "EXPD", "FWONA", "GGG", "JEF",
    "LAZ", "LEVI", "MMS", "ORI",
    "PSN", "SF", "SPSC", "STAG",
    "TRN", "UMBF", "VNT", "WFRD", "WMS",
    "ANET", "BRO", "CSGP", "DDOG", "ENPH",
    "GDDY", "LNTH", "OGN", "RBLX", "SAIA",
]

_STOCK_TICKERS_CHINA: list[str] = [
    "600519.SS", "601318.SS", "600036.SS", "601398.SS", "000858.SZ",
    "000333.SZ", "002594.SZ", "300750.SZ", "600887.SS", "601012.SS",
    "002415.SZ", "300059.SZ", "601888.SS", "600030.SS", "000651.SZ",
]

_TICKERS_BY_MARKET: dict[str, list[str]] = {
    "us": _STOCK_TICKERS,
    "china": _STOCK_TICKERS_CHINA,
}


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
    drop_label_nulls: bool = True,
) -> pl.DataFrame:
    df = pl.read_parquet(raw_dir / f"{ticker}.parquet")
    df = add_technical_indicators(df)
    df = add_cross_asset_features(df, spy_df, vix_df)
    df = add_neutral_sentiment(df)
    df = add_labels(df)
    if drop_label_nulls:
        # Training/backtesting need a real label — the last forward_days
        # rows can't have one yet, so they're dropped for that use case.
        return df.drop_nulls(subset=["label"])
    # Live inference doesn't need a label at all, so keep every row —
    # including the most recent trading day, which is what makes "today's"
    # signal actually be today instead of trailing by forward_days.
    return df


def build_live_features(raw_dir: Path = _RAW_DIR) -> pl.DataFrame:
    """Full per-ticker feature history through the latest raw trading day,
    with no label-driven trim on the tail. Used only for live signals —
    training/backtesting must keep using the labeled markets/us/data/features/*.parquet
    (via load_training_data) so their results stay unaffected."""
    spy_df = pl.read_parquet(raw_dir / "SPY.parquet")
    vix_df = pl.read_parquet(raw_dir / "^VIX.parquet")

    frames: list[pl.DataFrame] = []
    for ticker in _STOCK_TICKERS:
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            continue
        try:
            frames.append(build_features_for_ticker(
                ticker, raw_dir, spy_df, vix_df, drop_label_nulls=False,
            ))
        except Exception as exc:
            logger.warning("  live features FAILED %s: %s", ticker, exc)
    return pl.concat(frames, how="vertical_relaxed")


def main(market: str = "us") -> None:
    market_cfg = get_market(market)
    raw_dir = market_cfg.data_root / "raw" / "ohlcv"
    feature_dir = market_cfg.data_root / "features"
    if market not in _TICKERS_BY_MARKET:
        raise KeyError(f"No ticker list configured for market {market!r} in _TICKERS_BY_MARKET")
    tickers = _TICKERS_BY_MARKET[market]

    benchmark_path = raw_dir / f"{market_cfg.benchmark_ticker}.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"{market_cfg.benchmark_ticker}.parquet missing from {raw_dir}/. "
            "Run scripts/refresh_data.py first."
        )
    benchmark_df = pl.read_parquet(benchmark_path)

    if market_cfg.vol_index_ticker:
        vol_path = raw_dir / f"{market_cfg.vol_index_ticker}.parquet"
        if not vol_path.exists():
            raise FileNotFoundError(
                f"{market_cfg.vol_index_ticker}.parquet missing from {raw_dir}/. "
                "Run scripts/refresh_data.py first."
            )
        vix_df = pl.read_parquet(vol_path)
    else:
        vix_df = synthetic_vol_index(benchmark_df)

    feature_dir.mkdir(parents=True, exist_ok=True)

    successes: list[tuple[str, int]] = []
    failures:  list[tuple[str, str]] = []

    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            logger.warning("Skipping %s — raw parquet not found", ticker)
            failures.append((ticker, "raw parquet not found"))
            continue
        try:
            df = build_features_for_ticker(ticker, raw_dir, benchmark_df, vix_df)
            out_path = feature_dir / f"{ticker}.parquet"
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
