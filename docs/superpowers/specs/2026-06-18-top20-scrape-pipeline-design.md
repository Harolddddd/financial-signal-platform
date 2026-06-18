# Top-20 Stock Scrape & End-to-End Pipeline Design

**Date:** 2026-06-18
**Scope:** Scrape OHLCV data for the top 20 S&P 500 stocks (by current market cap) for 2000–2020, build features, train models, and surface the data in the existing Streamlit dashboard — all without requiring TimescaleDB.

---

## Goal

Populate the full pipeline from raw price data through to a live dashboard using three standalone scripts. The existing dashboard, model zoo, and feature engineering code require no changes.

---

## Stock Universe

Top 20 by current market cap (2025):

```
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, AVGO, TSM,
JPM, LLY, V, WMT, UNH, XOM, MA, ORCL, JNJ, HD
```

Plus two benchmark instruments required for cross-asset features:
```
SPY, ^VIX
```

**Note:** Tickers with IPO after 2000 (GOOGL 2004, TSLA 2010, META 2012, AVGO 2009) will return partial history from their listing date — this is expected and handled gracefully.

---

## Data Flow

```
yfinance API
    ↓  scripts/scrape_top20.py
data/raw/ohlcv/<TICKER>.parquet       (22 files: 20 stocks + SPY + ^VIX)
    ↓  scripts/build_features.py
data/features/<TICKER>.parquet        (20 files — SPY and VIX are inputs, not outputs)
    ↓  scripts/train_models.py
data/registry/<model>.pkl             (3 models: RandomForest, XGBoost, LightGBM)
    ↓
dashboard/app.py  (Streamlit — existing, unchanged)
```

---

## Script 1: `scripts/scrape_top20.py`

**Purpose:** Fetch raw OHLCV for all 22 tickers (20 + SPY + ^VIX) via yfinance and write one Parquet file per ticker.

**Inputs:** None (yfinance requires no API key for daily OHLCV)

**Outputs:** `data/raw/ohlcv/<TICKER>.parquet`

**Columns:** `time (Datetime[us, UTC]), ticker (Utf8), open, high, low, close, volume (Int64), adj_close, dividends, stock_splits (Float64)`

**Behaviour:**
- Date range: `2000-01-01` → `2020-12-31` (UTC)
- Uses existing `fetch_ohlcv()` from `src/ingestion/historical_collector.py`
- Creates `data/raw/ohlcv/` if it does not exist
- On per-ticker failure: logs warning, appends to `failed[]`, continues to next ticker
- Prints per-ticker row count and a summary table at the end

**Error handling:** Any ticker that fails (network, delisted, no data in range) is skipped; summary lists successes and failures.

---

## Script 2: `scripts/build_features.py`

**Purpose:** Read raw OHLCV parquet files and produce feature-enriched parquet files compatible with `duckdb_client.load_training_data()`.

**Inputs:** `data/raw/ohlcv/<TICKER>.parquet` (one per ticker)

**Outputs:** `data/features/<TICKER>.parquet` (20 files — SPY and ^VIX are inputs only)

**Feature columns produced** (matches `dashboard/config.py::FEATURE_COLS` exactly):
```
sma_10, sma_20, sma_50, sma_200, ema_12, ema_26,
rsi_14, macd, macd_signal, macd_hist,
bb_upper, bb_lower, bb_width, atr_14, hist_vol_21,
sent_pos_avg_3d, sent_pos_avg_5d, sent_pos_avg_10d,
sent_pos_mom_3d, news_vol_spike,
rel_strength_spy, vix_level,
forward_return_5d, label
```

**Sentiment handling:** No news data available for 2000–2020. All sentiment columns filled with neutral defaults: `sent_pos_avg_* = 0.5`, `sent_pos_mom_3d = 0.0`, `news_vol_spike = 0`. This avoids DB dependency while keeping schema compatible.

**Per-ticker pipeline:**
1. Load `data/raw/ohlcv/<TICKER>.parquet`
2. Load `data/raw/ohlcv/SPY.parquet` and `data/raw/ohlcv/^VIX.parquet`
3. `add_technical_indicators(df)` — uses `src/features/technical_indicators.py`
4. `add_cross_asset_features(df, spy_df, vix_df)` — uses `src/features/cross_asset_features.py`
5. Add neutral sentinel columns for all sentiment features
6. `add_labels(df)` — uses `src/features/label_generator.py`
7. `drop_nulls(subset=["label"])` — drops warm-up rows (first 200 days) without labels
8. Write to `data/features/<TICKER>.parquet`

**Error handling:** Per-ticker failures are logged and skipped.

---

## Script 3: `scripts/train_models.py`

**Purpose:** Train three classifiers on the full feature dataset and save them to the model registry.

**Inputs:** `data/features/*.parquet` via `src/features/duckdb_client.load_training_data()`

**Outputs:** `data/registry/<model_name>.pkl` + metadata JSON

**Models trained:**
| Name | Class |
|---|---|
| `random_forest` | `src/models/zoo/random_forest.RandomForestModel` |
| `xgboost` | `src/models/zoo/xgboost_model.XGBoostModel` |
| `lightgbm` | `src/models/zoo/lightgbm_model.LightGBMModel` |

**Training approach:** Single fit on full dataset (no walk-forward here — walk-forward happens in the dashboard's leaderboard/backtest views at runtime). Uses `FEATURE_COLS` from `dashboard/config.py`.

**Registry:** Uses existing `src/models/registry.save_model()`.

---

## Dashboard — No Changes Required

The existing dashboard reads from `data/features/` and `data/registry/`. Once the three scripts run, all four pages work:

| Page | What it shows |
|---|---|
| 1 Data Overview | 20 tickers, date range 2000–2020, price + SMA charts |
| 2 Model Leaderboard | RandomForest / XGBoost / LightGBM graded A–D |
| 3 Backtest Results | Walk-forward metrics per model |
| 4 Live Signals | Buy/sell signals with confidence from top-graded model |

---

## Constraints

- No TimescaleDB required — all I/O is local Parquet
- No API keys required — yfinance daily OHLCV is free
- Scripts are standalone and re-runnable; idempotent (overwrite existing files)
- Follows project conventions: Polars DataFrames, type hints, no Pandas except where yfinance forces it
- `data/raw/` and `data/features/` paths relative to project root (same convention as `dashboard/config.py`)

---

## Run Order

```bash
python scripts/scrape_top20.py       # ~2 min
python scripts/build_features.py     # ~30 sec
python scripts/train_models.py       # ~1-2 min
streamlit run dashboard/app.py
```
