# Market Registry & Data Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a single `MarketConfig`/`MARKETS` registry as the source of truth for per-market data roots and benchmark tickers, and migrate the existing `data/` tree to `markets/us/data/` behind it — with zero behavior change to any currently-passing test.

**Architecture:** One new module, `config/markets.py`, defines `MarketConfig` (a frozen dataclass) and a `MARKETS: dict[str, MarketConfig]` registry with `"us"` and `"china"` entries. The existing `data/` directory is physically moved to `markets/us/data/`; `markets/china/data/` is created as an empty skeleton. The one piece of `src/` code that hardcodes a `data/...` path today — `src/ingestion/survivorship.py` — is repointed through the registry. Every other hardcoded-path file (19 scripts/dags/dashboard files) is **out of scope for this plan** and is handled by the next plan in the sequence; those files still point at the old `data/` location and will not run correctly until then — this plan only proves the registry and the data move are sound, verified by the existing test suite staying green.

**Tech Stack:** Python 3.11, dataclasses, pathlib, pytest (existing suite).

## Global Constraints

- Zero behavior change to any existing test. `docs/DEVELOPER_GUIDE.md` claims `162 passed`, but that's stale — the measured baseline on this branch (before any change in this plan) is **331 passed, 12 failed, 2 skipped**. The 12 failures are pre-existing and unrelated to this plan: Airflow version drift (`ImportError: cannot import name 'ObjectStoragePath'` in `test_dags.py`, `test_feature_dag.py`, `test_model_retrain_dag.py`), a stale ticker-count assertion (`test_scrape_top20.py::test_tickers_list_has_20_stocks` expects 20, the real list has grown to 152), and a flaky Sharpe-ratio edge case (`test_backtest_metrics.py::test_sharpe_ratio_positive_for_consistent_gains`). Do not fix these — they are out of scope. The bar for this plan is: same 12 failures, same 2 skipped, zero new failures, plus whatever new tests each task adds.
- No new dependencies.
- `data/raw/` is gitignored (630MB, untracked) — the migration must not accidentally bring it under git tracking at its new path.
- Follow the approved design: `docs/superpowers/specs/2026-07-30-us-china-market-split-design.md`.

---

### Task 1: Market registry module

**Files:**
- Create: `config/markets.py`
- Test: `tests/unit/test_markets.py`

**Interfaces:**
- Produces: `MarketConfig` (frozen dataclass with fields `name: str`, `label: str`, `data_root: Path`, `universe: str`, `benchmark_ticker: str`, `vol_index_ticker: str | None`, `currency: str`); `MARKETS: dict[str, MarketConfig]`; `get_market(name: str) -> MarketConfig`. These are consumed by Task 2 (survivorship) and by the next plan (scripts/dags/dashboard).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_markets.py`:

```python
from pathlib import Path

import pytest

from config.markets import MARKETS, MarketConfig, get_market


def test_us_market_config():
    us = get_market("us")
    assert isinstance(us, MarketConfig)
    assert us.name == "us"
    assert us.data_root == Path("markets/us/data")
    assert us.universe == "sp500"
    assert us.benchmark_ticker == "SPY"
    assert us.vol_index_ticker == "^VIX"
    assert us.currency == "USD"


def test_china_market_config():
    china = get_market("china")
    assert china.name == "china"
    assert china.data_root == Path("markets/china/data")
    assert china.universe == "csi300"
    assert china.benchmark_ticker == "000300.SS"
    assert china.vol_index_ticker is None
    assert china.currency == "CNY"


def test_markets_have_distinct_data_roots():
    roots = {m.data_root for m in MARKETS.values()}
    assert len(roots) == len(MARKETS)


def test_get_market_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown market 'atlantis'"):
        get_market("atlantis")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_markets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config.markets'`

- [ ] **Step 3: Write the implementation**

Create `config/markets.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketConfig:
    name: str
    label: str
    data_root: Path
    universe: str
    benchmark_ticker: str
    vol_index_ticker: str | None
    currency: str


MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        name="us",
        label="United States (S&P 500)",
        data_root=Path("markets/us/data"),
        universe="sp500",
        benchmark_ticker="SPY",
        vol_index_ticker="^VIX",
        currency="USD",
    ),
    "china": MarketConfig(
        name="china",
        label="China A-Share (CSI 300)",
        data_root=Path("markets/china/data"),
        universe="csi300",
        benchmark_ticker="000300.SS",
        vol_index_ticker=None,
        currency="CNY",
    ),
}


def get_market(name: str) -> MarketConfig:
    try:
        return MARKETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown market {name!r}; valid markets: {sorted(MARKETS)}"
        ) from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_markets.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add config/markets.py tests/unit/test_markets.py
git commit -m "feat: add market registry (config/markets.py)"
```

---

### Task 2: Migrate `data/` to `markets/us/data/`, create `markets/china/data/` skeleton

**Files:**
- Move: `data/` → `markets/us/data/` (filesystem move, not `git mv` — see Step 1 rationale)
- Create: `markets/china/data/{cache,features,registry,index_compositions}/.gitkeep`
- Modify: `.gitignore:5`
- Test: none new (existing suite is the regression check)

**Interfaces:**
- Consumes: nothing new.
- Produces: `markets/us/data/` on disk containing everything that was in `data/` (raw, cache, features, registry, index_compositions). Later tasks/plans read paths via `get_market("us").data_root`.

- [ ] **Step 1: Move the directory**

`data/raw/` is gitignored and untracked (630MB) while `data/{cache,features,registry,index_compositions}` are git-tracked (558 files). A plain filesystem move followed by `git add -A` handles both correctly — git's rename detection pairs the tracked old/new paths automatically, and the untracked `raw/` content simply moves with it on disk without ever touching git.

```bash
mkdir -p markets/us
mv data markets/us/data
```

- [ ] **Step 2: Fix the now-stale gitignore rule**

`data/raw/` was anchored to the repo root; after the move the real path is `markets/us/data/raw/`, which is no longer covered. Without this fix, the next `git add -A` would attempt to stage 630MB of raw OHLCV data.

In `.gitignore`, change line 5:

```diff
-data/raw/
+markets/*/data/raw/
```

- [ ] **Step 3: Create the China data skeleton**

```bash
mkdir -p markets/china/data/{cache,features,registry,index_compositions}
touch markets/china/data/cache/.gitkeep
touch markets/china/data/features/.gitkeep
touch markets/china/data/registry/.gitkeep
touch markets/china/data/index_compositions/.gitkeep
```

(`markets/china/data/raw/` is intentionally not created — it's gitignored and every ingestion entry point already does `mkdir(parents=True, exist_ok=True)` before writing, per the design doc's error-handling section.)

- [ ] **Step 4: Repoint the one `src/` hardcoded path — `src/ingestion/survivorship.py`**

This is the only file under `src/` that hardcodes a `data/...` path (confirmed via `grep -rn "parents\[" src/`). Everything else stays untouched in this plan.

Modify `src/ingestion/survivorship.py:1-6`:

```diff
 from datetime import datetime
 from pathlib import Path

 import polars as pl

-_CSV_PATH = Path(__file__).parents[2] / "data" / "index_compositions" / "sp500_changes.csv"
+from config.markets import get_market
+
+_CSV_PATH = get_market("us").data_root / "index_compositions" / "sp500_changes.csv"
```

No other line in the file changes — `load_sp500_changes()` and `get_sp500_tickers_at()` keep their exact existing signatures, so the four DAG call sites (`dags/news_sentiment_dag.py`, `dags/feature_engineering_dag.py` ×2, `dags/historical_data_dag.py`) and `tests/unit/test_survivorship.py` need no changes.

One caveat: `_CSV_PATH` is now a path relative to the process's current working directory (`markets/us/data/...`) rather than one derived from `__file__`. This matches how every other data path in the codebase already works (`Path("data/cache")` etc. in scripts/dashboard) — all of them assume the process runs from the repo root, which `pytest` and `streamlit run dashboard/app.py` both do.

- [ ] **Step 5: Run the full existing test suite to confirm nothing broke**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `335 passed, 12 failed, 2 skipped` — the measured baseline (331 passed, 12 failed, 2 skipped) plus Task 1's 4 new `test_markets.py` tests. The 12 failures must be the *same* pre-existing ones named in Global Constraints — if any different test fails, that's a real regression from this task.

- [ ] **Step 6: Verify the gitignore fix actually excludes the moved raw data**

```bash
git status --porcelain=v1 markets/us/data/raw | head -5
```

Expected: no output (nothing untracked-but-not-ignored shows up; if raw files appear here, Step 2's pattern is wrong — fix and re-check before proceeding).

- [ ] **Step 7: Stage and commit**

```bash
git add -A -- data markets .gitignore src/ingestion/survivorship.py
git status
```

Confirm the status shows renames (`data/cache/... -> markets/us/data/cache/...` etc.) for the 558 previously-tracked files, additions for the four new `.gitkeep` files, and the `.gitignore`/`survivorship.py` modifications — and nothing under `markets/us/data/raw/`.

```bash
git commit -m "refactor: migrate data/ to markets/us/data/, add markets/china/data/ skeleton"
```

---

### Task 3: Register the `csi300` universe placeholder in `config/stocks.yaml`

**Files:**
- Modify: `config/stocks.yaml`

**Interfaces:**
- Consumes: nothing (this file currently has no Python reader anywhere in the codebase — confirmed via `grep -rn "stocks.yaml" --include=*.py`).
- Produces: a documented placeholder that the CSI 300 ingestion plan will populate with real tickers.

- [ ] **Step 1: Add the `csi300` universe entry**

Modify `config/stocks.yaml`:

```diff
 universes:
   sp500:
     description: "S&P 500 constituents — populated at runtime from survivorship module"
     tickers: []
+  csi300:
+    description: "CSI 300 constituents (China A-share) — static list, populated by the China ingestion plan"
+    tickers: []
   watchlist:
     description: "Small test watchlist"
     tickers:
       - AAPL
       - MSFT
       - GOOGL
       - AMZN
       - NVDA
```

- [ ] **Step 2: Commit**

```bash
git add config/stocks.yaml
git commit -m "docs: add csi300 universe placeholder to stocks.yaml"
```

---

## What this plan does NOT do (deliberately deferred to later plans)

- Does not update any of the 19 files that hardcode `data/cache`, `data/features`, `data/registry`, or `data/raw` paths (`scripts/*.py`, `dags/feature_engineering_dag.py`, `dags/model_retrain_dag.py`, `dashboard/config.py`, `dashboard/data_loader.py`, `dashboard/pages/4_Live_Signals.py`). **After this plan, none of those scripts will run correctly** — they still point at the now-nonexistent `data/` path. This is expected and acceptable because none of them are exercised by the automated test suite today (confirmed via `grep` — no test hardcodes a `data/` literal), so the "162 tests green" bar is unaffected. The next plan threads `--market`/`MARKETS` through those files.
- Does not touch `scripts/build_features.py`'s hardcoded 300+-ticker `_STOCK_TICKERS` list or `cross_asset_features.py`'s SPY/VIX loading — that's the China-ingestion plan.
- Does not add a dashboard market switcher — that's the last plan in the sequence.
