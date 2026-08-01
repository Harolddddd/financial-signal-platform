# China Pilot Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the China side of the market split actually works — real OHLCV ingestion for 15 well-known CSI 300 pilot tickers, real feature-building (with a synthetic volatility-index substitute, since no reliable CN VIX-equivalent exists), validated against the same bar as existing US feature output.

**Architecture:** `scripts/build_features.py` and `scripts/refresh_data.py` both gain a `--market us|china` flag (default `us`, existing invocations unchanged) and resolve their ticker list, raw/feature dirs, and benchmark/vol-index tickers from `config.markets.get_market(market)` instead of hardcoded US constants. A new `synthetic_vol_index()` function in `src/features/cross_asset_features.py` computes rolling realized volatility of the benchmark, shaped identically to a real VIX parquet, so it flows through the existing `add_cross_asset_features()` unchanged. The final task actually runs the pipeline for real and commits real China feature output, validated the same way `test_build_features.py` already validates US output.

**Tech Stack:** Python 3.11, polars, yfinance (via existing `src.ingestion.historical_collector.fetch_ohlcv`), pytest.

## Global Constraints

- No training or backtesting on China data in this plan — stops once features are built and validated. A follow-up plan generalizes training/backtesting.
- No changes to `scripts/scrape_top20.py`, the dashboard, or DAGs.
- `build_features_for_ticker()` and `build_live_features()` in `scripts/build_features.py` keep their exact current signatures — only each script's `main()` becomes market-parameterized. Module-level `_RAW_DIR`, `_FEATURE_DIR`, `_STOCK_TICKERS` constants in both `scripts/build_features.py` and `scripts/refresh_data.py` are directly imported by `tests/unit/test_data_paths.py` (`test_build_features_dirs_resolve_under_markets_us_data`, `test_refresh_data_dirs_resolve_under_markets_us_data`) — do not remove or rename them; they must keep resolving to the US path exactly as today.
- Zero behavior change to any existing test. Measured baseline on current main (commit `825646b`): **349 passed, 15 failed, 2 skipped**. All 15 failures are pre-existing and unrelated to this plan (Airflow version drift, stale ticker-count assertion, flaky Sharpe-ratio test, 3 `ma_crossover`-registry-staleness tests — see the prior plan for details). Do not fix these. The bar for every task: same 15 failures, same 2 skipped, zero new failures, plus whatever new tests each task adds.
- `--market` CLI flags use `choices=sorted(MARKETS)` (imported from `config.markets`), not a hardcoded `["us", "china"]` list, so a future third market doesn't need this list updated in two places.
- Follow the approved design: `docs/superpowers/specs/2026-07-31-china-pilot-ingestion-design.md`.

---

### Task 1: Synthetic volatility-index function

**Files:**
- Modify: `src/features/cross_asset_features.py`
- Test: `tests/unit/test_cross_asset_features.py` (append)

**Interfaces:**
- Produces: `synthetic_vol_index(benchmark_df: pl.DataFrame, window: int = 21) -> pl.DataFrame`, returning a `(time, close)`-shaped DataFrame — consumed by Task 2 and Task 3 as the `vix_df` argument to `add_cross_asset_features()` / `build_features_for_ticker()` when a market has no real vol-index ticker.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cross_asset_features.py` (the file already has a `_df(closes, ticker)` helper at the top — reuse it, don't redefine):

```python
def test_synthetic_vol_index_returns_time_and_close_columns():
    import numpy as np
    from src.features.cross_asset_features import synthetic_vol_index

    n = 60
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist()
    benchmark = _df(closes, ticker="000300.SS")

    result = synthetic_vol_index(benchmark, window=21)

    assert set(result.columns) == {"time", "close"}
    assert 0 < len(result) < n
    assert result["close"].null_count() == 0
    assert (result["close"] > 0).all()
    assert (result["close"] < 200).all()  # sane VIX-style upper bound
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_cross_asset_features.py::test_synthetic_vol_index_returns_time_and_close_columns -v`
Expected: FAIL with `ImportError: cannot import name 'synthetic_vol_index'`

- [ ] **Step 3: Write the implementation**

Modify `src/features/cross_asset_features.py` — add this function anywhere after the existing imports (e.g. right after `add_cross_asset_features`, before `_add_five_day_return`):

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

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_cross_asset_features.py -v`
Expected: 5 passed (the 4 existing tests in this file plus the new one)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `350 passed, 15 failed, 2 skipped` — the measured baseline (349) plus this task's 1 new test. Same 15 pre-existing failures.

- [ ] **Step 6: Commit**

```bash
git add src/features/cross_asset_features.py tests/unit/test_cross_asset_features.py
git commit -m "feat: add synthetic volatility-index function for markets with no VIX-equivalent"
```

---

### Task 2: Generalize `scripts/build_features.py` with `--market`

**Files:**
- Modify: `scripts/build_features.py`
- Modify: `config/stocks.yaml`
- Test: `tests/unit/test_build_features.py` (append)

**Interfaces:**
- Consumes: `synthetic_vol_index` (Task 1), `config.markets.get_market` / `MARKETS` (existing).
- Produces: `scripts.build_features._STOCK_TICKERS_CHINA: list[str]`, `scripts.build_features._TICKERS_BY_MARKET: dict[str, list[str]]` — consumed by Task 3 (`refresh_data.py` imports `_TICKERS_BY_MARKET` instead of the old `_STOCK_TICKERS`). `main(market: str = "us") -> None` now takes a market parameter; the CLI entry point parses `--market`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_build_features.py`:

```python
def test_tickers_by_market_has_expected_markets():
    from scripts.build_features import _STOCK_TICKERS, _STOCK_TICKERS_CHINA, _TICKERS_BY_MARKET
    assert _TICKERS_BY_MARKET["us"] == _STOCK_TICKERS
    assert _TICKERS_BY_MARKET["china"] == _STOCK_TICKERS_CHINA
    assert len(_STOCK_TICKERS_CHINA) == 15
    assert all(t.endswith(".SS") or t.endswith(".SZ") for t in _STOCK_TICKERS_CHINA)
    assert len(set(_STOCK_TICKERS_CHINA)) == 15  # no duplicates
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_build_features.py::test_tickers_by_market_has_expected_markets -v`
Expected: FAIL with `ImportError: cannot import name '_STOCK_TICKERS_CHINA'`

- [ ] **Step 3: Add the China ticker list and the market-keyed dict**

Modify `scripts/build_features.py` — insert immediately after the closing `]` of the existing `_STOCK_TICKERS` list (currently ending at line 135) and before `def add_neutral_sentiment`:

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

- [ ] **Step 4: Add the `synthetic_vol_index` import**

Modify `scripts/build_features.py:1-9` (the import block):

```diff
 from pathlib import Path
 import logging

 import polars as pl

-from config.markets import get_market
+from config.markets import MARKETS, get_market
 from src.features.technical_indicators import add_technical_indicators
-from src.features.cross_asset_features import add_cross_asset_features
+from src.features.cross_asset_features import add_cross_asset_features, synthetic_vol_index
 from src.features.label_generator import add_labels
```

- [ ] **Step 5: Generalize `main()`**

Replace `scripts/build_features.py`'s `main()` function (currently lines 192-235) with:

```python
def main(market: str = "us") -> None:
    market_cfg = get_market(market)
    raw_dir = market_cfg.data_root / "raw" / "ohlcv"
    feature_dir = market_cfg.data_root / "features"
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
```

This does NOT touch `_RAW_DIR`, `_FEATURE_DIR`, `_STOCK_TICKERS`, `build_features_for_ticker()`, or `build_live_features()` — those stay exactly as they are today (still US-only, still relied on by `test_data_paths.py` and the live-signal scripts).

- [ ] **Step 6: Update the CSI 300 documentation placeholder**

Modify `config/stocks.yaml` — replace the empty `csi300.tickers: []` with the pilot list (this file has no Python reader, confirmed in the prior plan — this is documentation only):

```diff
   csi300:
-    description: "CSI 300 constituents (China A-share) — static list, populated by the China ingestion plan"
-    tickers: []
+    description: "CSI 300 pilot subset (15 tickers, China A-share) — full 300-constituent list is a follow-up plan"
+    tickers:
+      - "600519.SS"
+      - "601318.SS"
+      - "600036.SS"
+      - "601398.SS"
+      - "000858.SZ"
+      - "000333.SZ"
+      - "002594.SZ"
+      - "300750.SZ"
+      - "600887.SS"
+      - "601012.SS"
+      - "002415.SZ"
+      - "300059.SZ"
+      - "601888.SS"
+      - "600030.SS"
+      - "000651.SZ"
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/unit/test_build_features.py -v`
Expected: all tests in this file pass, including the new one (5 total: the 4 existing plus this task's 1).

- [ ] **Step 8: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `351 passed, 15 failed, 2 skipped`. Same 15 pre-existing failures. Also specifically re-run `pytest tests/unit/test_data_paths.py -v` — 17 passed, confirming `_RAW_DIR`/`_FEATURE_DIR` still resolve correctly (untouched by this task).

- [ ] **Step 9: Commit**

```bash
git add scripts/build_features.py config/stocks.yaml tests/unit/test_build_features.py
git commit -m "feat: add --market flag to build_features.py, add China pilot ticker list"
```

---

### Task 3: Generalize `scripts/refresh_data.py` with `--market`

**Files:**
- Modify: `scripts/refresh_data.py`
- Test: Create `tests/unit/test_refresh_data.py`

**Interfaces:**
- Consumes: `_TICKERS_BY_MARKET` (Task 2), `synthetic_vol_index` (Task 1), `config.markets.get_market` / `MARKETS`.
- Produces: `main(market: str = "us") -> None`; `_refresh_raw(ticker: str, today: datetime, raw_dir: Path) -> None` (now takes `raw_dir` as an explicit parameter instead of reading the module-level `_RAW_DIR` global); `_AUX_TICKERS_BY_MARKET: dict[str, list[str]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_refresh_data.py`:

```python
from datetime import datetime, timezone

import polars as pl
import pytest


def test_aux_tickers_by_market():
    from scripts.refresh_data import _AUX_TICKERS_BY_MARKET
    assert _AUX_TICKERS_BY_MARKET["us"] == ["SPY", "^VIX"]
    assert _AUX_TICKERS_BY_MARKET["china"] == ["000300.SS"]


def test_refresh_raw_new_ticker_writes_to_given_raw_dir(tmp_path, monkeypatch):
    from scripts.refresh_data import _refresh_raw

    sample = pl.DataFrame({
        "time":         [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker":       ["600519.SS"],
        "open":         [1700.0], "high": [1705.0], "low": [1690.0], "close": [1700.0],
        "volume":       [100_000],
        "adj_close":    [1700.0], "dividends": [0.0], "stock_splits": [0.0],
    })
    monkeypatch.setattr("scripts.refresh_data.fetch_ohlcv", lambda *a, **k: sample)

    _refresh_raw("600519.SS", datetime(2024, 1, 3, tzinfo=timezone.utc), tmp_path)

    assert (tmp_path / "600519.SS.parquet").exists()
    written = pl.read_parquet(tmp_path / "600519.SS.parquet")
    assert len(written) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_refresh_data.py -v`
Expected: FAIL — `test_aux_tickers_by_market` with `ImportError: cannot import name '_AUX_TICKERS_BY_MARKET'`; `test_refresh_raw_new_ticker_writes_to_given_raw_dir` with a `TypeError` (current `_refresh_raw` only takes 2 positional args).

- [ ] **Step 3: Rewrite `scripts/refresh_data.py`**

Modify the imports (`scripts/refresh_data.py:1-17`):

```diff
 """Incrementally refresh OHLCV data for all 151 tickers to today, then rebuild features."""
 from __future__ import annotations
 import logging
 from datetime import datetime, timedelta, timezone
+from pathlib import Path

 import polars as pl

-from config.markets import get_market
-from scripts.build_features import _STOCK_TICKERS, build_features_for_ticker
+from config.markets import MARKETS, get_market
+from scripts.build_features import _TICKERS_BY_MARKET, build_features_for_ticker
+from src.features.cross_asset_features import synthetic_vol_index
 from src.ingestion.historical_collector import fetch_ohlcv

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

 _RAW_DIR = get_market("us").data_root / "raw" / "ohlcv"
 _FEATURE_DIR = get_market("us").data_root / "features"
-_AUX_TICKERS = ["SPY", "^VIX"]
+_AUX_TICKERS_BY_MARKET: dict[str, list[str]] = {
+    "us": ["SPY", "^VIX"],
+    "china": ["000300.SS"],
+}


 _HISTORY_START = datetime(1990, 1, 1, tzinfo=timezone.utc)
```

(`_RAW_DIR`/`_FEATURE_DIR` stay exactly as-is — still tested directly by `test_data_paths.py`, still resolve to the US path.)

Modify `_refresh_raw` to take `raw_dir` as a parameter instead of reading the module global — change its signature and every internal reference to `_RAW_DIR`:

```diff
-def _refresh_raw(ticker: str, today: datetime) -> None:
-    raw_path = _RAW_DIR / f"{ticker}.parquet"
+def _refresh_raw(ticker: str, today: datetime, raw_dir: Path) -> None:
+    raw_path = raw_dir / f"{ticker}.parquet"
```

(The rest of `_refresh_raw`'s body is unchanged — it only ever referenced `_RAW_DIR` once, in that first line, to build `raw_path`.)

Replace `main()` (currently lines 62-88) with:

```python
def main(market: str = "us") -> None:
    market_cfg = get_market(market)
    raw_dir = market_cfg.data_root / "raw" / "ohlcv"
    feature_dir = market_cfg.data_root / "features"
    tickers = _TICKERS_BY_MARKET[market]
    aux_tickers = _AUX_TICKERS_BY_MARKET[market]

    today = datetime.now(timezone.utc)
    logger.info("=== Data refresh (%s) → %s ===", market, today.date())

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[1/2] Refreshing raw OHLCV (%d stock tickers + aux)", len(tickers))
    for ticker in list(tickers) + aux_tickers:
        _refresh_raw(ticker, today, raw_dir)

    benchmark_df = pl.read_parquet(raw_dir / f"{market_cfg.benchmark_ticker}.parquet")
    if market_cfg.vol_index_ticker:
        vix_df = pl.read_parquet(raw_dir / f"{market_cfg.vol_index_ticker}.parquet")
    else:
        vix_df = synthetic_vol_index(benchmark_df)

    logger.info("[2/2] Rebuilding features for %d tickers", len(tickers))
    feature_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker in tickers:
        if not (raw_dir / f"{ticker}.parquet").exists():
            continue
        try:
            df = build_features_for_ticker(ticker, raw_dir, benchmark_df, vix_df)
            df.write_parquet(feature_dir / f"{ticker}.parquet")
            ok += 1
        except Exception as exc:
            logger.warning("  features FAIL %s: %s", ticker, exc)
            fail += 1

    logger.info("=== Done. features OK=%d fail=%d ===", ok, fail)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_refresh_data.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `353 passed, 15 failed, 2 skipped` — the prior 351 plus this task's 2 new tests. Same 15 pre-existing failures. Also specifically re-run `pytest tests/unit/test_data_paths.py -v` — 17 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/refresh_data.py tests/unit/test_refresh_data.py
git commit -m "feat: add --market flag to refresh_data.py"
```

---

### Task 4: Run the China pilot ingestion for real and validate the output

**Files:**
- No source changes — this task executes Tasks 1-3's code for real and commits the resulting data.
- Test: Create `tests/unit/test_china_pilot_features.py`

**Interfaces:**
- Consumes: `scripts/refresh_data.py --market china`, `scripts/build_features.py --market china` (Tasks 2-3), `dashboard.ui_config.FEATURE_COLS`, `config.markets.get_market`.

- [ ] **Step 1: Run the real ingestion**

From the repo root, with the venv active:

```bash
python scripts/refresh_data.py --market china
```

This fetches OHLCV for the 15 pilot tickers plus the `000300.SS` benchmark from yfinance (already confirmed reachable — see the design doc's "Validated assumptions" section) and writes to `markets/china/data/raw/ohlcv/`, then builds features into `markets/china/data/features/` using the synthetic vol index (Task 1) since China has no `vol_index_ticker`.

Expected: log line `=== Done. features OK=15 fail=0 ===`. If any ticker fails, investigate before proceeding — do not commit partial/broken output. (`markets/china/data/raw/ohlcv/` is gitignored per `markets/*/data/raw/` in `.gitignore` — this is expected, matches the US raw data convention.)

- [ ] **Step 2: Spot-check the real output**

```bash
python -c "
import polars as pl
from config.markets import get_market
df = pl.read_parquet(get_market('china').data_root / 'features' / '600519.SS.parquet')
print(df.shape)
print(df.columns)
print(df['label'].value_counts())
"
```

Expected: a non-trivial row count (hundreds to low thousands of daily rows depending on how much history yfinance returns for this ticker), all 22 feature columns present, `label` containing only `Buy`/`Hold`/`Sell`.

- [ ] **Step 3: Write the validation test**

Create `tests/unit/test_china_pilot_features.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_china_pilot_features.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `355 passed, 15 failed, 2 skipped` — the prior 353 plus this task's 2 new tests. Same 15 pre-existing failures.

- [ ] **Step 6: Commit the real China feature data and the validation test**

`markets/china/data/features/` is git-tracked (matches the US convention — `features/` is not in `.gitignore`, only `raw/` is):

```bash
git add markets/china/data/features/ markets/china/data/index_compositions/ tests/unit/test_china_pilot_features.py
git status
```

Confirm the status shows 15 new `.parquet` files under `markets/china/data/features/` (the `index_compositions/` add should be a no-op — nothing changed there in this plan, only `.gitkeep` already tracked from the prior plan).

```bash
git commit -m "data: China pilot ingestion — 15 CSI 300 tickers, features validated"
```

---

## What this plan does NOT do (deliberately deferred)

- No full CSI 300 constituent list (300 real tickers) — a follow-up plan.
- No training or backtesting on China data — a fast follow-up plan once these features are proven, reusing `train_models.py`/backtesting code generalized the same way `build_features.py`/`refresh_data.py` were here.
- No dashboard changes.
