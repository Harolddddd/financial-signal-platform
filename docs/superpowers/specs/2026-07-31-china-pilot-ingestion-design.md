# China Pilot Ingestion — Design

## Goal

Prove the China side of the US/China market split actually works end-to-end —
real OHLCV ingestion for a small, well-known set of CSI 300 constituents,
real feature-building (including a synthetic volatility-index substitute,
since no reliable CN VIX-equivalent exists), validated output — without yet
committing to sourcing/maintaining the full 300-ticker CSI 300 universe or
generalizing training/backtesting. Training and backtesting on China data
follow as a fast follow-up plan once features exist and are proven correct.

## Non-goals

- No full CSI 300 constituent list (300 real tickers) — a 15-ticker pilot
  only. Expanding to the full index is a later plan.
- No training or backtesting on China data in this plan.
- No dashboard changes (still a separate, later plan).
- No changes to `scripts/scrape_top20.py` — `refresh_data.py`'s existing
  "new ticker → fetch full history" path already covers first-time China
  ingestion, so there's no need to touch the legacy bootstrap script.

## Validated assumptions

Before designing further, two assumptions the whole China plan depends on
were checked directly against live yfinance, not assumed:

- `fetch_ohlcv("600519.SS", ...)`, `fetch_ohlcv("000001.SZ", ...)`,
  `fetch_ohlcv("300750.SZ", ...)` all returned clean 22-row OHLCV for
  January 2024 — Shanghai (`.SS`) and Shenzhen (`.SZ`) suffixed tickers work
  through the existing `src/ingestion/historical_collector.fetch_ohlcv`
  with zero code changes.
- `fetch_ohlcv("000300.SS", ...)` (the CSI 300 index itself, used as the
  benchmark ticker) also returned clean data the same way.

## Architecture

### Pilot ticker universe

15 CSI 300 constituents spanning sectors, plus the benchmark index:

```
600519.SS  Kweichow Moutai          (consumer staples / liquor)
601318.SS  Ping An Insurance        (financials / insurance)
600036.SS  China Merchants Bank     (financials / banking)
601398.SS  ICBC                     (financials / banking)
000858.SZ  Wuliangye Yibin          (consumer staples / liquor)
000333.SZ  Midea Group              (consumer discretionary / appliances)
002594.SZ  BYD Company              (EV / auto)
300750.SZ  CATL                     (EV / battery)
600887.SS  Yili Group               (consumer staples / dairy)
601012.SS  LONGi Green Energy       (renewables / solar)
002415.SZ  Hikvision                (tech / surveillance)
300059.SZ  East Money Information   (fintech / brokerage)
601888.SS  China Tourism Duty Free  (consumer / travel retail)
600030.SS  CITIC Securities         (financials / brokerage)
000651.SZ  Gree Electric Appliances (consumer discretionary / appliances)

Benchmark: 000300.SS (CSI 300 index) — MarketConfig.benchmark_ticker, already set
```

Recorded in two places:
- `config/stocks.yaml`'s existing `csi300` universe entry (documentation —
  no code reads this file today, confirmed via repo-wide grep; left as-is).
- A new Python constant in `scripts/build_features.py`, mirroring the
  existing `_STOCK_TICKERS` (US) constant:

```python
_STOCK_TICKERS_CHINA: list[str] = [
    "600519.SS", "601318.SS", "600036.SS", "601398.SS", "000858.SZ",
    "000333.SZ", "002594.SZ", "300750.SZ", "600887.SS", "601012.SS",
    "002415.SZ", "300059.SZ", "601888.SS", "600030.SS", "000651.SZ",
]

_TICKERS_BY_MARKET: dict[str, list[str]] = {
    "us": _STOCK_TICKERS,
    "china": _STOCK_TICKERS_CHINA,
}
```

`refresh_data.py` imports `_TICKERS_BY_MARKET` (and drops its current direct
`_STOCK_TICKERS` import) so both scripts resolve the ticker list the same way.

### Synthetic volatility substitution

New function in `src/features/cross_asset_features.py`:

```python
def synthetic_vol_index(benchmark_df: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    """Rolling realized volatility of the benchmark's close-to-close returns,
    annualized and scaled to a VIX-style index level, shaped like a real
    vix_df (time, close) so it flows through add_cross_asset_features()
    unchanged. Used when a market has no reliable direct vol-index ticker
    (MarketConfig.vol_index_ticker is None)."""
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    realized_vol = (
        log_ret.rolling_std(window_size=window) * (252 ** 0.5) * 100
    ).alias("close")
    return benchmark_df.select([
        pl.col("time"),
        realized_vol,
    ]).drop_nulls()
```

This scales to roughly the same numeric range as a real VIX level (typically
10-40), so `vix_level` stays on a comparable scale across markets without
changing anything downstream — `add_cross_asset_features`, the label
generator, and the model code never need to know which market produced the
number.

### Generalizing `build_features.py` and `refresh_data.py`

Both scripts gain a `--market us|china` flag, default `us` (existing
invocations with no flag are unchanged). Internally, each resolves its
US-hardcoded pieces from `config.markets.get_market(market)` instead:

- Ticker list: `_TICKERS_BY_MARKET[market]` instead of the bare `_STOCK_TICKERS` loop.
- Raw/feature dirs: `get_market(market).data_root / "raw" / "ohlcv"` and
  `.../ "features"` (already the pattern from the prior plan, just now
  parameterized instead of hardcoded to `"us"`).
- Benchmark/vol-index loading in `build_features.py`'s `main()` and
  `build_live_features()`: replace the hardcoded `SPY.parquet`/`^VIX.parquet`
  reads with `get_market(market).benchmark_ticker` /
  `.vol_index_ticker`-driven reads; when `vol_index_ticker is None`, call
  `synthetic_vol_index(benchmark_df)` instead of reading a second file.
- `refresh_data.py`'s `_AUX_TICKERS` becomes market-derived too: `["SPY", "^VIX"]`
  for `us`, `["000300.SS"]` only for `china` (no separate vol-index ticker to fetch).

`build_features_for_ticker()` itself (the per-ticker function `test_build_features.py`
already covers) keeps its exact current signature — it already takes
`spy_df`/`vix_df` as plain parameters, market-agnostic by construction. Only
the two scripts' `main()`/orchestration functions change.

## Data flow

```
python scripts/refresh_data.py --market china
  → fetch OHLCV for 15 pilot tickers + 000300.SS benchmark
  → write markets/china/data/raw/ohlcv/{ticker}.parquet

python scripts/build_features.py --market china
  → load 000300.SS benchmark parquet
  → synthetic_vol_index(benchmark_df)  [since vol_index_ticker is None]
  → build_features_for_ticker(...) per pilot ticker, same as US path
  → write markets/china/data/features/{ticker}.parquet
```

Both commands actually run as part of this plan — this isn't just code, it's
a real one-time population of `markets/china/data/`, validated against the
same bar as existing US feature output (schema, row counts, no null labels).

## Error handling

- Unknown `--market` value: already raises a clear `KeyError` via the
  existing `config.markets.get_market` registry (no new handling needed).
- Missing benchmark parquet: raises `FileNotFoundError` naming the expected
  path, matching the existing pattern for missing `SPY.parquet`/`^VIX.parquet`.
- No new failure modes: `synthetic_vol_index` only depends on the
  benchmark's own OHLCV, which the same run fetches first — no external
  dependency beyond what `refresh_data.py` already ingests.

## Testing

- `synthetic_vol_index`: unit test with a small synthetic price series with
  known volatility, asserting the output has `(time, close)` columns and a
  realized-vol value in a sane range (not NaN/negative/absurdly large).
- `build_features.py`/`refresh_data.py` market resolution: unit tests in the
  style of `tests/unit/test_data_paths.py`, asserting `--market china`
  resolves to `_STOCK_TICKERS_CHINA`, `markets/china/data/raw/ohlcv`,
  `markets/china/data/features`, and benchmark ticker `000300.SS` with no
  vol-index ticker.
- Real China feature output: after actually running the two commands above,
  a test (or a validation step, mirroring `test_build_features.py`'s
  existing assertions) confirms every one of the 15 output parquet files has
  the full 22-column feature schema, `label` has zero nulls, and labels are
  a subset of `{"Buy", "Hold", "Sell"}`.
