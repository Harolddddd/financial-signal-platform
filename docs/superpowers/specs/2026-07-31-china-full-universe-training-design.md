# China Full-Universe Training & Backtesting — Design

## Goal

Expand China from the 15-ticker pilot (Plan 3) to the real, live CSI 500
(500 tickers), then generalize the training and backtesting entrypoints to
actually run on it — producing real trained models and real
backtest/leaderboard/signal output for China, mirroring what already exists
for the US.

## Non-goals

- No dashboard market-switcher UI — separate, later plan.
- No `incremental_train.py` generalization — nothing to incrementally
  update immediately after an initial training run; add `--market` to it
  in a later plan once there's new China data to incrementally train on
  (same YAGNI call made for `build_features.py`'s `--market` flag in
  Plan 2).
- No full 8-model zoo — `train_new_models.py` (logistic_regression,
  naive_bayes, mlp, svm, lstm) stays US-only. Only the 3 core models
  (`random_forest`, `xgboost`, `lightgbm`) train on China via
  `train_models.py`, matching the scale already proven in Plan 3 before
  committing heavier models — `lstm`'s 420-day window and `svm`'s
  30,000-row sample were tuned against the full US universe and haven't
  been validated at any China scale yet.
- No Airflow DAG changes (`dags/*.py`) — those use their own hardcoded
  `/opt/airflow/...` container paths, entirely separate from the
  `config.markets` registry; an unrelated deployment concern.
- No CSI 300 — superseded by this plan's choice of CSI 500 as the single
  China universe (not combined, not kept alongside it).
- No survivorship-adjusted CSI 500 history — like the US `sp500` universe,
  this is the current constituent list as of today, not a
  point-in-time-correct historical membership record.

## Validated assumptions

Before designing further, the assumptions this plan depends on were
checked directly, not assumed:

- **CSI 500 constituents are fetchable from a real, public, official
  source.** `https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls`
  (index code `000905` = CSI 500; the same endpoint pattern China
  Securities Index Co. serves official constituent files from, and what
  the open-source `akshare` library's `index_stock_cons_csindex` function
  calls under the hood) returned a real 500-row Excel file — 500 unique
  constituent codes, English/Chinese names, and Shenzhen/Shanghai exchange
  tags only (no Beijing Stock Exchange constituents to special-case).
  Codes were zero-padded to 6 digits and mapped to yfinance suffixes
  (`.SZ` for Shenzhen, `.SS` for Shanghai, including STAR Market and
  Beijing-registered-but-Shanghai-listed codes) — 500 unique tickers, no
  collisions.
- **The mapped tickers resolve on yfinance.** Spot-checked 3 tickers
  (`000009.SZ`, `689009.SS`, `688819.SS`) via `yfinance.download` for a
  Jan 2024 date range — all 3 returned clean 6-row OHLCV.

The full mapped list (ticker, Chinese name, English name) is saved at
`docs/reference/csi500-constituents.csv` for reference.

## Architecture

### CSI 500 ticker expansion

`_STOCK_TICKERS_CHINA` in `scripts/build_features.py` grows from the
15-ticker pilot to the real 500 CSI 500 tickers fetched above — a
hardcoded Python list, the same pattern as the existing US
`_STOCK_TICKERS` constant (not yaml-driven, not re-fetched at runtime — a
static snapshot, consistent with the "static current list, no
survivorship tracking" call already made for the US universe).
`config/stocks.yaml`'s China universe entry is renamed from `csi300` to
`csi500`, its `tickers` list and description updated to match (still
documentation only — no code reads this file, reconfirmed unchanged from
Plan 3). `_AUX_TICKERS_BY_MARKET["china"]` and the benchmark ticker
(`510300.SS`) are unchanged from Plan 3 — the benchmark tracks CSI 300,
which remains a reasonable China-market proxy regardless of which
constituent universe the pilot trains on.

### Generalizing `train_models.py`

Gains a `--market us|china` flag, default `us` (existing invocations
unchanged), matching the convention already established in
`build_features.py`/`refresh_data.py`. Internally:

- `_FEATURE_DIR` resolves via `get_market(market).data_root / "features"`
  instead of the hardcoded `get_market("us")`.
- `REGISTRY_DIR` resolves the same way, via
  `get_market(market).data_root / "registry"`, instead of importing the
  US-hardcoded `dashboard.ui_config.REGISTRY_DIR`.
- `main()` takes `market: str = "us"` and threads it through; the
  `train_and_save()` helper and model list are unchanged — already
  market-agnostic.

This creates `markets/china/data/registry/` for the first time.

### Generalizing `precompute_dashboard.py`

Same treatment:

- `CACHE_DIR` resolves via `get_market(market).data_root / "cache"`.
- The feature dir it reads (currently `dashboard.ui_config.PARQUET_DIR`)
  resolves via `get_market(market).data_root / "features"`.
- `step_data_summary()`, `step_backtests()`, `step_leaderboard()`,
  `step_signals()` each take the resolved `cache_dir`/`feature_dir` as
  parameters instead of reading the module-level constants directly (the
  same "explicit parameter instead of module global" pattern already used
  for `refresh_data.py`'s `_refresh_raw(ticker, today, raw_dir)` in Plan 3).
- `step_backtests()` already defaults `step_days=21` (monthly) — unchanged,
  confirmed this stays the default for China too.
- `main()` gains the same `--market` flag and threads it through all four
  steps.

This creates `markets/china/data/cache/{data_summary,leaderboard,signals,
backtest_*}.json`.

## Data flow

```
scripts/refresh_data.py --market china         → raw OHLCV for 500 tickers + 510300.SS benchmark
scripts/build_features.py --market china       → markets/china/data/features/{ticker}.parquet (~500 files)
scripts/train_models.py --market china         → markets/china/data/registry/{random_forest,xgboost,lightgbm}/...
scripts/precompute_dashboard.py --market china → markets/china/data/cache/*.json
```

All four commands actually run for real as part of this plan, same as
Plan 3 — this is real data population, not just code.

## Error handling

- Unknown `--market` value: existing `KeyError` from `config.markets.get_market`,
  no new handling needed.
- Per-ticker ingestion/feature-build failures: already tolerated end to end
  (`refresh_data.py`'s `_refresh_raw` catches and logs per-ticker
  exceptions; its feature-rebuild loop tracks `ok`/`fail` counts and
  continues). At 500 tickers, some yfinance misses are expected (very
  recent listings, temporary data gaps) — the validation step checks a
  high-but-not-100% coverage threshold, not "all 500 or the run failed."
- Missing feature/registry directory when running `train_models.py` or
  `precompute_dashboard.py` for a market with no data yet: existing
  `FileNotFoundError` pattern, unchanged.

## Testing

- `--market` resolution unit tests for `train_models.py` and
  `precompute_dashboard.py`, in the style of `tests/unit/test_data_paths.py`
  — asserting `--market china` resolves to `markets/china/data/registry`
  and `markets/china/data/cache` respectively.
- `_STOCK_TICKERS_CHINA` test: exactly 500 entries, all unique, each ending
  in `.SS` or `.SZ` — mirrors Plan 3's
  `test_tickers_by_market_has_expected_markets`, scaled up.
- Real-data validation after actually running the pipeline:
  - Feature-file coverage: at least 90% of the 500 tickers have a valid
    feature parquet (tolerating a handful of yfinance misses, per Error
    Handling above).
  - Schema/null-label checks on every produced feature file, scaled up
    from Plan 3's per-ticker `test_china_pilot_features.py` pattern (full
    22-column schema, zero null labels, labels subset of
    `{"Buy", "Hold", "Sell"}`, no weekend timestamps).
  - China's registry has 3 saved model JSONs (`random_forest`, `xgboost`,
    `lightgbm`).
  - China's cache has non-empty `leaderboard.json` and `signals.json`, and
    at least one `backtest_*.json` per registered strategy.
