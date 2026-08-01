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
    "000009.SZ", "000021.SZ", "000027.SZ", "000032.SZ", "000034.SZ", "000039.SZ", "000050.SZ", "000060.SZ",
    "000062.SZ", "000088.SZ", "000155.SZ", "000400.SZ", "000415.SZ", "000423.SZ", "000429.SZ", "000513.SZ",
    "000519.SZ", "000528.SZ", "000537.SZ", "000539.SZ", "000559.SZ", "000582.SZ", "000591.SZ", "000598.SZ",
    "000623.SZ", "000629.SZ", "000661.SZ", "000683.SZ", "000703.SZ", "000709.SZ", "000723.SZ", "000728.SZ",
    "000729.SZ", "000733.SZ", "000737.SZ", "000738.SZ", "000739.SZ", "000750.SZ", "000783.SZ", "000785.SZ",
    "000786.SZ", "000800.SZ", "000825.SZ", "000830.SZ", "000831.SZ", "000878.SZ", "000883.SZ", "000887.SZ",
    "000893.SZ", "000898.SZ", "000921.SZ", "000932.SZ", "000937.SZ", "000951.SZ", "000959.SZ", "000960.SZ",
    "000967.SZ", "000983.SZ", "000987.SZ", "000997.SZ", "001203.SZ", "001221.SZ", "001286.SZ", "001309.SZ",
    "001386.SZ", "001389.SZ", "001696.SZ", "002007.SZ", "002008.SZ", "002025.SZ", "002032.SZ", "002044.SZ",
    "002056.SZ", "002064.SZ", "002065.SZ", "002078.SZ", "002085.SZ", "002120.SZ", "002126.SZ", "002130.SZ",
    "002131.SZ", "002138.SZ", "002152.SZ", "002153.SZ", "002155.SZ", "002157.SZ", "002185.SZ", "002195.SZ",
    "002203.SZ", "002223.SZ", "002244.SZ", "002252.SZ", "002261.SZ", "002262.SZ", "002265.SZ", "002266.SZ",
    "002271.SZ", "002273.SZ", "002281.SZ", "002299.SZ", "002312.SZ", "002318.SZ", "002335.SZ", "002340.SZ",
    "002402.SZ", "002407.SZ", "002409.SZ", "002410.SZ", "002414.SZ", "002423.SZ", "002429.SZ", "002430.SZ",
    "002432.SZ", "002436.SZ", "002444.SZ", "002461.SZ", "002465.SZ", "002472.SZ", "002487.SZ", "002500.SZ",
    "002508.SZ", "002517.SZ", "002568.SZ", "002583.SZ", "002601.SZ", "002603.SZ", "002608.SZ", "002624.SZ",
    "002670.SZ", "002673.SZ", "002683.SZ", "002738.SZ", "002739.SZ", "002756.SZ", "002773.SZ", "002797.SZ",
    "002812.SZ", "002821.SZ", "002831.SZ", "002841.SZ", "002850.SZ", "002851.SZ", "002926.SZ", "002939.SZ",
    "002945.SZ", "002966.SZ", "002984.SZ", "003021.SZ", "003022.SZ", "003031.SZ", "003035.SZ", "300001.SZ",
    "300002.SZ", "300003.SZ", "300012.SZ", "300017.SZ", "300024.SZ", "300037.SZ", "300054.SZ", "300058.SZ",
    "300073.SZ", "300100.SZ", "300115.SZ", "300136.SZ", "300140.SZ", "300142.SZ", "300144.SZ", "300146.SZ",
    "300207.SZ", "300223.SZ", "300285.SZ", "300339.SZ", "300346.SZ", "300373.SZ", "300383.SZ", "300390.SZ",
    "300395.SZ", "300432.SZ", "300454.SZ", "300458.SZ", "300474.SZ", "300475.SZ", "300487.SZ", "300496.SZ",
    "300548.SZ", "300558.SZ", "300567.SZ", "300570.SZ", "300604.SZ", "300620.SZ", "300623.SZ", "300627.SZ",
    "300666.SZ", "300676.SZ", "300677.SZ", "300679.SZ", "300699.SZ", "300718.SZ", "300724.SZ", "300735.SZ",
    "300748.SZ", "300751.SZ", "300757.SZ", "300759.SZ", "300763.SZ", "300857.SZ", "300888.SZ", "300919.SZ",
    "300953.SZ", "300957.SZ", "300972.SZ", "301200.SZ", "301301.SZ", "301358.SZ", "301377.SZ", "301498.SZ",
    "301526.SZ", "301536.SZ", "301606.SZ", "301611.SZ", "600004.SS", "600008.SS", "600021.SS", "600032.SS",
    "600038.SS", "600060.SS", "600062.SS", "600095.SS", "600098.SS", "600100.SS", "600105.SS", "600109.SS",
    "600126.SS", "600131.SS", "600132.SS", "600141.SS", "600143.SS", "600153.SS", "600157.SS", "600161.SS",
    "600166.SS", "600170.SS", "600171.SS", "600177.SS", "600208.SS", "600256.SS", "600282.SS", "600292.SS",
    "600295.SS", "600298.SS", "600299.SS", "600312.SS", "600316.SS", "600329.SS", "600332.SS", "600339.SS",
    "600348.SS", "600350.SS", "600352.SS", "600363.SS", "600369.SS", "600377.SS", "600378.SS", "600380.SS",
    "600390.SS", "600392.SS", "600398.SS", "600435.SS", "600483.SS", "600486.SS", "600497.SS", "600498.SS",
    "600499.SS", "600511.SS", "600516.SS", "600517.SS", "600521.SS", "600535.SS", "600536.SS", "600546.SS",
    "600562.SS", "600563.SS", "600566.SS", "600578.SS", "600582.SS", "600583.SS", "600595.SS", "600598.SS",
    "600601.SS", "600602.SS", "600606.SS", "600637.SS", "600642.SS", "600655.SS", "600663.SS", "600685.SS",
    "600688.SS", "600699.SS", "600704.SS", "600707.SS", "600711.SS", "600737.SS", "600754.SS", "600763.SS",
    "600764.SS", "600765.SS", "600801.SS", "600808.SS", "600816.SS", "600820.SS", "600848.SS", "600862.SS",
    "600863.SS", "600871.SS", "600873.SS", "600879.SS", "600884.SS", "600885.SS", "600901.SS", "600906.SS",
    "600909.SS", "600927.SS", "600967.SS", "600968.SS", "600970.SS", "600977.SS", "600985.SS", "600988.SS",
    "600995.SS", "600998.SS", "601000.SS", "601001.SS", "601016.SS", "601019.SS", "601098.SS", "601099.SS",
    "601106.SS", "601108.SS", "601112.SS", "601118.SS", "601128.SS", "601139.SS", "601155.SS", "601156.SS",
    "601162.SS", "601179.SS", "601198.SS", "601212.SS", "601216.SS", "601228.SS", "601233.SS", "601236.SS",
    "601298.SS", "601399.SS", "601555.SS", "601567.SS", "601577.SS", "601598.SS", "601608.SS", "601611.SS",
    "601615.SS", "601665.SS", "601666.SS", "601696.SS", "601699.SS", "601717.SS", "601799.SS", "601808.SS",
    "601865.SS", "601866.SS", "601869.SS", "601880.SS", "601928.SS", "601958.SS", "601966.SS", "601990.SS",
    "601991.SS", "601997.SS", "603000.SS", "603049.SS", "603077.SS", "603087.SS", "603092.SS", "603119.SS",
    "603129.SS", "603156.SS", "603160.SS", "603175.SS", "603179.SS", "603225.SS", "603233.SS", "603256.SS",
    "603290.SS", "603298.SS", "603308.SS", "603338.SS", "603341.SS", "603345.SS", "603379.SS", "603444.SS",
    "603486.SS", "603529.SS", "603565.SS", "603568.SS", "603589.SS", "603596.SS", "603605.SS", "603606.SS",
    "603650.SS", "603658.SS", "603659.SS", "603688.SS", "603699.SS", "603728.SS", "603737.SS", "603766.SS",
    "603786.SS", "603806.SS", "603816.SS", "603833.SS", "603858.SS", "603885.SS", "603899.SS", "603920.SS",
    "603939.SS", "603979.SS", "605358.SS", "605589.SS", "688002.SS", "688017.SS", "688018.SS", "688019.SS",
    "688027.SS", "688037.SS", "688052.SS", "688065.SS", "688099.SS", "688114.SS", "688120.SS", "688122.SS",
    "688166.SS", "688169.SS", "688172.SS", "688180.SS", "688187.SS", "688188.SS", "688192.SS", "688200.SS",
    "688213.SS", "688220.SS", "688234.SS", "688235.SS", "688248.SS", "688266.SS", "688278.SS", "688281.SS",
    "688295.SS", "688297.SS", "688301.SS", "688313.SS", "688318.SS", "688322.SS", "688331.SS", "688336.SS",
    "688343.SS", "688347.SS", "688349.SS", "688361.SS", "688363.SS", "688375.SS", "688385.SS", "688387.SS",
    "688411.SS", "688425.SS", "688469.SS", "688475.SS", "688498.SS", "688520.SS", "688538.SS", "688561.SS",
    "688563.SS", "688568.SS", "688578.SS", "688582.SS", "688599.SS", "688608.SS", "688615.SS", "688617.SS",
    "688629.SS", "688676.SS", "688692.SS", "688702.SS", "688708.SS", "688709.SS", "688728.SS", "688772.SS",
    "688777.SS", "688778.SS", "688819.SS", "689009.SS",
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
