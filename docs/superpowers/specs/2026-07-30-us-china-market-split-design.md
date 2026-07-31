# US / China Market Split — Design

## Goal

Split the platform into two markets — existing US universe (S&P 500) and a new
China A-share universe (CSI 300) — sharing one codebase (`src/`, `scripts/`,
`dashboard/`) but keeping training data, caches, and model registries fully
separate per market.

## Non-goals

- No survivorship-bias-free CSI 300 history (static current-constituents list only).
- No new database schema — PostgreSQL tables already key on `ticker`, which is
  globally unique once suffixed (`600519.SS` vs `AAPL`).
- No change to the 22-feature schema or model architectures — "same methods."

## Architecture

### Folder layout

```
markets/
  us/data/{raw,cache,features,registry,index_compositions}     ← current data/ moved here verbatim
  china/data/{raw,cache,features,registry,index_compositions}  ← new, empty until first ingestion run
```

`src/`, `scripts/`, `dashboard/`, `config/` stay a single shared tree. No code
duplication between markets.

### Market registry — `config/markets.py`

```python
@dataclass(frozen=True)
class MarketConfig:
    name: str
    label: str
    data_root: Path
    universe: str            # key into config/stocks.yaml `universes`
    benchmark_ticker: str
    vol_index_ticker: str | None
    currency: str

MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        name="us", label="United States (S&P 500)",
        data_root=Path("markets/us/data"), universe="sp500",
        benchmark_ticker="SPY", vol_index_ticker="^VIX", currency="USD",
    ),
    "china": MarketConfig(
        name="china", label="China A-Share (CSI 300)",
        data_root=Path("markets/china/data"), universe="csi300",
        benchmark_ticker="000300.SS", vol_index_ticker=None, currency="CNY",
    ),
}

def get_market(name: str) -> MarketConfig:
    return MARKETS[name]
```

This is the single source of truth for "which data root / which benchmark /
which universe" a given run uses. Every path that is currently a hardcoded
`Path("data/cache")`-style literal is replaced with
`get_market(market).data_root / "cache"` (etc).

### Files requiring the hardcoded-path → market-aware change

Found via grep for `data/cache|data/features|data/registry|data/raw|data/index_compositions`:

- `dags/feature_engineering_dag.py`, `dags/model_retrain_dag.py`
- `dashboard/config.py`, `dashboard/data_loader.py`, `dashboard/pages/4_Live_Signals.py`
- `scripts/build_features.py`, `scripts/incremental_train.py`,
  `scripts/precompute_dashboard.py`, `scripts/precompute_full.py`,
  `scripts/precompute_new_strategies.py`, `scripts/refresh_data.py`,
  `scripts/run_signals_isolated.py`, `scripts/scrape_top20.py`,
  `scripts/signal_one_strategy.py`, `scripts/train_lstm_only.py`,
  `scripts/train_models.py`, `scripts/train_new_models.py`

Scripts gain a `--market us|china` CLI flag (argparse), default `us` so
existing invocations/automation keep working unchanged.

`dashboard/config.py` module-level constants (`PARQUET_DIR`, `REGISTRY_DIR`)
become a function `get_market_paths(market: str)` since the dashboard needs to
serve both markets in one running process (switcher, not restart).

### China data ingestion

No new ingestion code path needed:

- `collect_ohlcv("600519.SS", start, end)` already works today — yfinance
  natively resolves `.SS` (Shanghai) and `.SZ` (Shenzhen) suffixed tickers.
  The existing yfinance → Alpha Vantage → FMP fallback chain
  (`src/ingestion/collector.py`) is reused as-is; the AV/FMP legs will likely
  fail for CN tickers and that's fine, since yfinance is the primary source.
- No exchange-calendar dependency exists in `src/` (confirmed via grep for
  `market_calendar|NYSE|trading_days`) — features are computed off each
  ticker's own row series, so SSE/SZSE trading-day differences from NYSE need
  no code change.

### Benchmark / volatility-index substitution

`src/features/feature_store.py` currently hardcodes `SPY` and `^VIX` as the
benchmark and vol-index tickers (`_load_spy_from_db`, `_load_vix_from_db`).
These become market-parameterized lookups against `MarketConfig.benchmark_ticker`
/ `vol_index_ticker`.

China has no reliable direct VIX-equivalent on yfinance. When
`vol_index_ticker is None`, `vix_level` is computed instead as rolling
realized volatility of the benchmark ticker (`000300.SS`) itself. This keeps
the 22-column feature schema identical across markets, so model training code
needs zero changes.

### CSI 300 universe file

New static file `markets/china/data/index_compositions/csi300_constituents.csv`
(ticker, name, sector — no add/remove history, per the static-list decision).
Populated during implementation from a current public CSI 300 constituents
listing. `config/stocks.yaml` gains a `csi300` universe entry alongside the
existing `sp500`/`watchlist` ones.

### Migration of existing data

`git mv data/ markets/us/data/` preserves history. All ~16 files above get
their path literals updated in the same change.

### Dashboard

Single Streamlit app (`dashboard/app.py`), not two. A market selectbox
(US / China) near the top of the app drives which `MarketConfig` the rest of
the page tree resolves paths against. Existing pages under `dashboard/pages/`
are unchanged in structure — they call `get_market_paths(selected_market)`
instead of importing module-level constants.

## Data flow (per market)

```
collect_ohlcv(ticker, start, end)         [unchanged, ticker string carries market]
  → PostgreSQL ohlcv table                 [unchanged schema]
  → build_features(..., market=...)        [benchmark/vix now market-parameterized]
  → export_parquet(df, ticker, market_root / "features")
  → train_models(..., market=...)          [registry written to market_root / "registry"]
  → backtest/signals(..., market=...)      [reads/writes market_root / "cache"]
  → dashboard, market selectbox            [reads whichever market_root is selected]
```

## Error handling

- Unknown `--market` value → `KeyError` from `MARKETS[name]` surfaces
  immediately with a clear message (no silent fallback to US).
- If `markets/china/data/` subfolders don't exist yet when a script runs,
  create them (`Path.mkdir(parents=True, exist_ok=True)`) rather than erroring
  — matches how `data/cache` etc. are already handled today for a fresh clone.

## Testing

- Existing 162 tests are US-path and stay green untouched (they exercise the
  default `market="us"` behavior).
- New unit tests:
  - `MARKETS` registry resolves both `"us"` and `"china"` to distinct,
    correctly-shaped `MarketConfig`.
  - CSI 300 universe loader returns the static constituent list.
  - `vix_level` feature computation falls back to realized-volatility-of-benchmark
    when `vol_index_ticker is None`, and still produces the same column name/dtype.
  - At least one script (e.g. `build_features.py`) round-trips with
    `--market china` against a small synthetic CN ticker fixture.
