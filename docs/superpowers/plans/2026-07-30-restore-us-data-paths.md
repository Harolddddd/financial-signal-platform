# Restore US Data Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every script, dashboard module, and dashboard page that still hardcodes a `data/...` literal path — broken since the prior plan physically moved `data/` to `markets/us/data/` — so the US pipeline (ingest → build features → train → precompute → live signals → dashboard) works end-to-end again.

**Architecture:** No `--market` CLI flag in this plan — nothing can target China yet (no China data exists until a future ingestion plan), so adding CLI plumbing now would be speculative. Instead, every hardcoded `Path("data/...")` module-level constant is replaced with the equivalent path resolved through the existing `config.markets.get_market("us")` registry (added in the prior plan). This is a pure "fix the default", not the dashboard market-switcher feature (that stays a separate future plan). A new test file, `tests/unit/test_data_paths.py`, asserts every one of these constants resolves under `markets/us/data/` — no such check existed before, which is exactly how the prior plan's data move broke 15 files silently with zero test failures.

**Tech Stack:** Python 3.11, pathlib, pytest (existing suite + new `test_data_paths.py`).

## Global Constraints

- No `--market` flag, no new CLI arguments, no function-signature changes in this plan — every fix is a one-line (or few-line) change to a module-level constant's right-hand side, plus one new `import`. Defer CLI/multi-market plumbing to whichever future plan actually adds China data.
- Every changed constant must resolve to a path under `markets/us/data/` (via `get_market("us").data_root`), never a hardcoded `"markets/us/data"` string literal — always go through `config.markets.get_market`.
- Zero behavior change to any existing test. Measured baseline on current `main` (commit `ed5f95c`, after the raw-data move to `markets/us/data/raw/`): **332 passed, 15 failed, 2 skipped**. All 15 failures are pre-existing and unrelated to this plan (12 documented in the prior plan: Airflow version drift, stale `scrape_top20` ticker-count assertion, flaky Sharpe-ratio test; plus 3 more surfaced when the prior plan's merge committed pending work that had already dropped the `ma_crossover` strategy: `test_strategies_registry.py::test_list_strategies_contains_expected`, `::test_load_strategy_returns_strategy_instance`, `::test_load_strategy_injects_params`). Do not fix any of these 15 — out of scope. The bar for every task in this plan is: same 15 failures, same 2 skipped, zero new failures, plus whatever new tests each task adds.
- No new dependencies.
- Follow the approved design: `docs/superpowers/specs/2026-07-30-us-china-market-split-design.md`.

---

### Task 1: Fix the dashboard path hub

**Files:**
- Modify: `dashboard/config.py`
- Modify: `dashboard/data_loader.py:19`
- Modify: `dashboard/pages/4_Live_Signals.py:11`
- Create: `tests/unit/test_data_paths.py`

**Interfaces:**
- Consumes: `config.markets.get_market(name: str) -> MarketConfig` (existing, from the prior plan), `MarketConfig.data_root: Path`.
- Produces: `dashboard.config.PARQUET_DIR`, `dashboard.config.REGISTRY_DIR`, `dashboard.data_loader.CACHE_DIR` now resolve to real, existing paths under `markets/us/data/`. `tests/unit/test_data_paths.py` is created here and extended by every later task in this plan — later tasks append new test functions to this same file, they do not create a new one.

`dashboard/config.py` is imported directly by 10 other files (4 dashboard pages, 6 scripts) for `PARQUET_DIR`/`REGISTRY_DIR` — fixing it here means none of those 10 files need any change themselves for those two constants.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_data_paths.py`:

```python
from pathlib import Path

from config.markets import get_market

_US_ROOT = get_market("us").data_root


def test_dashboard_config_paths_resolve_under_markets_us_data():
    from dashboard.config import PARQUET_DIR, REGISTRY_DIR
    assert PARQUET_DIR == _US_ROOT / "features"
    assert REGISTRY_DIR == _US_ROOT / "registry"
    assert PARQUET_DIR.exists()
    assert REGISTRY_DIR.exists()


def test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data():
    from dashboard.data_loader import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
    assert CACHE_DIR.exists()


def test_live_signals_page_no_longer_hardcodes_old_data_path():
    # dashboard/pages/4_Live_Signals.py runs Streamlit UI calls
    # (st.set_page_config, st.slider, ...) at module level, and "4_Live_Signals"
    # isn't a valid Python identifier — it can't be imported directly in a
    # test. Verify the source text instead: the stale literal must be gone
    # and the fix must route through get_market.
    source = Path("dashboard/pages/4_Live_Signals.py").read_text()
    assert 'Path("data/cache")' not in source
    assert "get_market(" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: FAIL — `test_dashboard_config_paths_resolve_under_markets_us_data` and `test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data` fail with `AssertionError` (constants still equal `Path("data/features")` etc, which don't exist); `test_live_signals_page_no_longer_hardcodes_old_data_path` fails because the stale literal is still present.

- [ ] **Step 3: Fix `dashboard/config.py`**

```diff
 # dashboard/config.py
 from pathlib import Path

-PARQUET_DIR  = Path("data/features")
-REGISTRY_DIR = Path("data/registry")
+from config.markets import get_market
+
+PARQUET_DIR  = get_market("us").data_root / "features"
+REGISTRY_DIR = get_market("us").data_root / "registry"
```

- [ ] **Step 4: Fix `dashboard/data_loader.py`**

Modify `dashboard/data_loader.py:1-19` (the import block and `CACHE_DIR` line):

```diff
 # dashboard/data_loader.py
 from __future__ import annotations
 import json
 import logging
 from pathlib import Path

 import polars as pl

+from config.markets import get_market
 from src.backtesting.grader import Grade, ModelGrade, grade_model, build_leaderboard
 from src.backtesting.metrics import BacktestMetrics
 from src.backtesting.walk_forward import FoldBacktestResult, WalkForwardBacktestResult
 from src.backtesting.strategy_runner import walk_forward_backtest_strategy
 from src.features.duckdb_client import load_training_data
 from src.strategies.base import LiveSignal, Signal
 from src.strategies.registry import list_strategies, load_strategy

 logger = logging.getLogger(__name__)

-CACHE_DIR = Path("data/cache")
+CACHE_DIR = get_market("us").data_root / "cache"
```

- [ ] **Step 5: Fix `dashboard/pages/4_Live_Signals.py`**

Modify `dashboard/pages/4_Live_Signals.py:1-11`:

```diff
 # dashboard/pages/4_Live_Signals.py
 import json
 from pathlib import Path

 import pandas as pd
 import streamlit as st

+from config.markets import get_market
 from dashboard.config import CONFIDENCE_THRESHOLD, PARQUET_DIR, OHLCV_COLS, FEATURE_COLS
 from dashboard.data_loader import get_live_signals

-CACHE_DIR = Path("data/cache")
+CACHE_DIR = get_market("us").data_root / "cache"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: 3 passed

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `335 passed, 15 failed, 2 skipped` — the measured baseline (332 passed) plus this task's 3 new tests. Same 15 pre-existing failures named in Global Constraints.

- [ ] **Step 8: Commit**

```bash
git add dashboard/config.py dashboard/data_loader.py dashboard/pages/4_Live_Signals.py tests/unit/test_data_paths.py
git commit -m "fix: resolve dashboard data paths through market registry"
```

---

### Task 2: Fix the raw-ingestion / feature-building trio

**Files:**
- Modify: `scripts/scrape_top20.py:75`
- Modify: `scripts/build_features.py:13-14`
- Modify: `scripts/refresh_data.py:15-16`
- Modify: `tests/unit/test_data_paths.py` (append)

**Interfaces:**
- Consumes: same `config.markets.get_market` as Task 1.
- Produces: `scripts.scrape_top20._OUTPUT_DIR`, `scripts.build_features._RAW_DIR` / `_FEATURE_DIR`, `scripts.refresh_data._RAW_DIR` / `_FEATURE_DIR` now resolve under `markets/us/data/`. `refresh_data.py` imports `_STOCK_TICKERS` from `build_features.py` — unaffected by this task, only the path constants change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_paths.py`:

```python
def test_scrape_top20_output_dir_resolves_under_markets_us_data():
    from scripts.scrape_top20 import _OUTPUT_DIR
    assert _OUTPUT_DIR == _US_ROOT / "raw" / "ohlcv"
    assert _OUTPUT_DIR.exists()


def test_build_features_dirs_resolve_under_markets_us_data():
    from scripts.build_features import _RAW_DIR, _FEATURE_DIR
    assert _RAW_DIR == _US_ROOT / "raw" / "ohlcv"
    assert _FEATURE_DIR == _US_ROOT / "features"
    assert _RAW_DIR.exists()
    assert _FEATURE_DIR.exists()


def test_refresh_data_dirs_resolve_under_markets_us_data():
    from scripts.refresh_data import _RAW_DIR, _FEATURE_DIR
    assert _RAW_DIR == _US_ROOT / "raw" / "ohlcv"
    assert _FEATURE_DIR == _US_ROOT / "features"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: the 3 new tests FAIL (constants still equal the stale `Path("data/raw/ohlcv")` / `Path("data/features")` literals); the 3 tests from Task 1 still PASS.

- [ ] **Step 3: Fix `scripts/scrape_top20.py`**

Modify `scripts/scrape_top20.py:1-9` (imports) and line 75:

```diff
 from datetime import datetime, timezone
 import logging
 from pathlib import Path

 import polars as pl

+from config.markets import get_market
 from src.ingestion.historical_collector import fetch_ohlcv

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)
```

```diff
-_OUTPUT_DIR = Path("data/raw/ohlcv")
+_OUTPUT_DIR = get_market("us").data_root / "raw" / "ohlcv"
```

- [ ] **Step 4: Fix `scripts/build_features.py`**

Modify `scripts/build_features.py:1-15`:

```diff
 from pathlib import Path
 import logging

 import polars as pl

+from config.markets import get_market
 from src.features.technical_indicators import add_technical_indicators
 from src.features.cross_asset_features import add_cross_asset_features
 from src.features.label_generator import add_labels

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_RAW_DIR     = Path("data/raw/ohlcv")
-_FEATURE_DIR = Path("data/features")
+_RAW_DIR     = get_market("us").data_root / "raw" / "ohlcv"
+_FEATURE_DIR = get_market("us").data_root / "features"
```

- [ ] **Step 5: Fix `scripts/refresh_data.py`**

Modify `scripts/refresh_data.py:1-17`:

```diff
 """Incrementally refresh OHLCV data for all 151 tickers to today, then rebuild features."""
 from __future__ import annotations
 import logging
 from datetime import datetime, timedelta, timezone
 from pathlib import Path

 import polars as pl

+from config.markets import get_market
 from scripts.build_features import _STOCK_TICKERS, build_features_for_ticker
 from src.ingestion.historical_collector import fetch_ohlcv

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_RAW_DIR = Path("data/raw/ohlcv")
-_FEATURE_DIR = Path("data/features")
+_RAW_DIR = get_market("us").data_root / "raw" / "ohlcv"
+_FEATURE_DIR = get_market("us").data_root / "features"
 _AUX_TICKERS = ["SPY", "^VIX"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: 6 passed

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `338 passed, 15 failed, 2 skipped`. Same 15 pre-existing failures.

- [ ] **Step 8: Commit**

```bash
git add scripts/scrape_top20.py scripts/build_features.py scripts/refresh_data.py tests/unit/test_data_paths.py
git commit -m "fix: resolve raw-ingestion and feature-building paths through market registry"
```

---

### Task 3: Fix the 4 training scripts

**Files:**
- Modify: `scripts/incremental_train.py:13-25`
- Modify: `scripts/train_new_models.py:11-31`
- Modify: `scripts/train_lstm_only.py:1-16`
- Modify: `scripts/train_models.py:1-19`
- Modify: `tests/unit/test_data_paths.py` (append)

**Interfaces:**
- Consumes: same `config.markets.get_market` as Task 1. `REGISTRY_DIR` for all 4 scripts already comes from `dashboard.config` (fixed in Task 1) — only each script's own `_FEATURE_DIR` needs fixing here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_paths.py`:

```python
def test_incremental_train_feature_dir_resolves_under_markets_us_data():
    from scripts.incremental_train import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_new_models_feature_dir_resolves_under_markets_us_data():
    from scripts.train_new_models import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_lstm_only_feature_dir_resolves_under_markets_us_data():
    from scripts.train_lstm_only import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_models_feature_dir_resolves_under_markets_us_data():
    from scripts.train_models import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: the 4 new tests FAIL; the 6 tests from Tasks 1-2 still PASS.

- [ ] **Step 3: Fix `scripts/incremental_train.py`**

Modify `scripts/incremental_train.py:13-25`:

```diff
 from pathlib import Path
 from datetime import datetime
 import logging

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, REGISTRY_DIR
 from src.features.duckdb_client import load_training_data
 from src.models.evaluator import evaluate
 from src.models.registry import list_models, load_model, save_model

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_FEATURE_DIR = Path("data/features")
+_FEATURE_DIR = get_market("us").data_root / "features"
 _TRAIN_RATIO = 0.8
 _MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]
```

- [ ] **Step 4: Fix `scripts/train_new_models.py`**

Modify `scripts/train_new_models.py:11-32`:

```diff
 from pathlib import Path
 from datetime import datetime, timedelta, timezone
 import logging

 import polars as pl

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, REGISTRY_DIR
 from src.features.duckdb_client import load_training_data
 from src.models.base_classifier import BaseClassifier
 from src.models.evaluator import evaluate
 from src.models.registry import save_model
 from src.models.zoo.logistic_regression import LogisticRegressionClassifier
 from src.models.zoo.naive_bayes import NaiveBayesClassifier
 from src.models.zoo.mlp_model import MLPClassifier_
 from src.models.zoo.lstm_model import LSTMClassifier
 from src.models.zoo.svm_model import SVMClassifier

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_FEATURE_DIR = Path("data/features")
+_FEATURE_DIR = get_market("us").data_root / "features"
 _TRAIN_RATIO = 0.8
 _SVM_SAMPLE_SIZE = 30_000
 _LSTM_WINDOW_DAYS = 420
```

- [ ] **Step 5: Fix `scripts/train_lstm_only.py`**

Modify `scripts/train_lstm_only.py:1-16` (the whole top of the file):

```diff
 """Retry just the LSTM leg of scripts/train_new_models.py after the tz-comparison fix."""
 from pathlib import Path
 from datetime import datetime, timedelta, timezone
 import logging

 import polars as pl

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, REGISTRY_DIR
 from src.features.duckdb_client import load_training_data
 from src.models.zoo.lstm_model import LSTMClassifier
 from scripts.train_new_models import train_and_save, _LSTM_WINDOW_DAYS

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_FEATURE_DIR = Path("data/features")
+_FEATURE_DIR = get_market("us").data_root / "features"
```

- [ ] **Step 6: Fix `scripts/train_models.py`**

Modify `scripts/train_models.py:1-20`:

```diff
 from pathlib import Path
 import logging

 import numpy as np
 import polars as pl

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, REGISTRY_DIR
 from src.features.duckdb_client import load_training_data
 from src.models.base_classifier import BaseClassifier
 from src.models.evaluator import evaluate
 from src.models.registry import save_model
 from src.models.zoo.random_forest import RandomForestClassifier_
 from src.models.zoo.xgboost_model import XGBoostClassifier
 from src.models.zoo.lightgbm_model import LightGBMClassifier

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_FEATURE_DIR = Path("data/features")
+_FEATURE_DIR = get_market("us").data_root / "features"
 _TRAIN_RATIO = 0.8
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: 10 passed

- [ ] **Step 8: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `342 passed, 15 failed, 2 skipped`. Same 15 pre-existing failures. (Also re-run `pytest tests/unit/test_train_models.py -v` specifically — 3 passed — since Task 3 touches the file that test covers.)

- [ ] **Step 9: Commit**

```bash
git add scripts/incremental_train.py scripts/train_new_models.py scripts/train_lstm_only.py scripts/train_models.py tests/unit/test_data_paths.py
git commit -m "fix: resolve training-script feature paths through market registry"
```

---

### Task 4: Fix the 3 precompute scripts

**Files:**
- Modify: `scripts/precompute_new_strategies.py:10-26`
- Modify: `scripts/precompute_dashboard.py:9-28`
- Modify: `scripts/precompute_full.py:9-25`
- Modify: `tests/unit/test_data_paths.py` (append)

**Interfaces:**
- Consumes: same `config.markets.get_market` as Task 1. `PARQUET_DIR`/`FEATURE_COLS`/`OHLCV_COLS` for all 3 scripts already come from `dashboard.config` (fixed in Task 1) — only each script's own `CACHE_DIR` needs fixing here. `precompute_new_strategies.py` and `precompute_full.py` both import `step_leaderboard`/`step_signals` from `precompute_dashboard.py` at call time inside `main()` — unaffected by this task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_paths.py`:

```python
def test_precompute_new_strategies_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_new_strategies import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
    assert CACHE_DIR.exists()


def test_precompute_dashboard_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_dashboard import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"


def test_precompute_full_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_full import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: the 3 new tests FAIL; the 10 tests from Tasks 1-3 still PASS.

- [ ] **Step 3: Fix `scripts/precompute_new_strategies.py`**

Modify `scripts/precompute_new_strategies.py:10-26`:

```diff
 from __future__ import annotations
 import json
 import logging
 from datetime import datetime, timezone
 from pathlib import Path

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
 from src.backtesting.grader import grade_model
 from src.backtesting.metrics import BacktestMetrics
 from src.backtesting.strategy_runner import walk_forward_backtest_strategy
 from src.features.duckdb_client import load_training_data
 from src.strategies.registry import load_strategy

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-CACHE_DIR = Path("data/cache")
+CACHE_DIR = get_market("us").data_root / "cache"
```

- [ ] **Step 4: Fix `scripts/precompute_dashboard.py`**

Modify `scripts/precompute_dashboard.py:9-28`:

```diff
 from __future__ import annotations
 import json
 import logging
 from datetime import datetime, timezone
 from pathlib import Path

 import polars as pl

+from config.markets import get_market
 from dashboard.config import CONFIDENCE_THRESHOLD, FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
 from scripts.build_features import build_live_features
 from src.backtesting.grader import grade_model
 from src.backtesting.metrics import BacktestMetrics
 from src.backtesting.strategy_runner import _is_stateless, _select_cols, walk_forward_backtest_strategy
 from src.features.duckdb_client import load_training_data
 from src.strategies.registry import list_strategies, load_strategy

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-CACHE_DIR = Path("data/cache")
+CACHE_DIR = get_market("us").data_root / "cache"
```

- [ ] **Step 5: Fix `scripts/precompute_full.py`**

Modify `scripts/precompute_full.py:9-25`:

```diff
 from __future__ import annotations
 import json
 import logging
 from datetime import datetime, timezone
 from pathlib import Path

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
 from src.backtesting.grader import grade_model
 from src.backtesting.metrics import BacktestMetrics
 from src.backtesting.strategy_runner import walk_forward_backtest_strategy
 from src.features.duckdb_client import load_training_data
 from src.strategies.registry import list_strategies, load_strategy

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-CACHE_DIR = Path("data/cache")
+CACHE_DIR = get_market("us").data_root / "cache"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: 13 passed

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `345 passed, 15 failed, 2 skipped`. Same 15 pre-existing failures.

- [ ] **Step 8: Commit**

```bash
git add scripts/precompute_new_strategies.py scripts/precompute_dashboard.py scripts/precompute_full.py tests/unit/test_data_paths.py
git commit -m "fix: resolve precompute-script cache paths through market registry"
```

---

### Task 5: Fix the isolated live-signal generation pair

**Files:**
- Modify: `scripts/signal_one_strategy.py:9-29`
- Modify: `scripts/run_signals_isolated.py:27-41,149-161`
- Modify: `tests/unit/test_data_paths.py` (append)

**Interfaces:**
- Consumes: same `config.markets.get_market` as Task 1. `PARQUET_DIR` for both scripts already comes from `dashboard.config` (fixed in Task 1).
- Produces: `run_signals_isolated.py` spawns `signal_one_strategy.py` as a subprocess (`subprocess.Popen([sys.executable, "-u", "scripts/signal_one_strategy.py", name], ...)`) and both scripts must agree on `_LIVE_CACHE`/`_TRAIN_CACHE` paths — the parent writes those parquet caches, the child reads them. Keep both files' constants byte-for-byte identical expressions (both derive from `get_market("us").data_root / "cache"`), since a later plan may thread `--market` through this parent/child pair together — the important invariant right now is that they still agree.

`run_signals_isolated.py` also has an inline `Path("data/cache/signals.json")` literal inside `_merge_and_write()` (not a module-level constant like the others) — this task adds a module-level `_SIGNALS_PATH` constant and uses it there instead, consistent with every other file in this plan.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_paths.py`:

```python
def test_signal_one_strategy_paths_resolve_under_markets_us_data():
    from scripts.signal_one_strategy import _OUT_DIR, _LIVE_CACHE, _TRAIN_CACHE
    assert _OUT_DIR == _US_ROOT / "cache" / "signals_partial"
    assert _LIVE_CACHE == _US_ROOT / "cache" / "_tmp_live_features.parquet"
    assert _TRAIN_CACHE == _US_ROOT / "cache" / "_tmp_training_data.parquet"


def test_run_signals_isolated_paths_resolve_under_markets_us_data():
    from scripts.run_signals_isolated import _PARTIAL_DIR, _LIVE_CACHE, _TRAIN_CACHE, _SIGNALS_PATH
    assert _PARTIAL_DIR == _US_ROOT / "cache" / "signals_partial"
    assert _LIVE_CACHE == _US_ROOT / "cache" / "_tmp_live_features.parquet"
    assert _TRAIN_CACHE == _US_ROOT / "cache" / "_tmp_training_data.parquet"
    assert _SIGNALS_PATH == _US_ROOT / "cache" / "signals.json"


def test_signal_one_strategy_and_run_signals_isolated_agree_on_cache_paths():
    # The parent (run_signals_isolated) writes _LIVE_CACHE/_TRAIN_CACHE for
    # the child (signal_one_strategy, spawned as a subprocess) to read —
    # they must always point at the exact same files.
    from scripts.signal_one_strategy import _LIVE_CACHE as child_live, _TRAIN_CACHE as child_train
    from scripts.run_signals_isolated import _LIVE_CACHE as parent_live, _TRAIN_CACHE as parent_train
    assert child_live == parent_live
    assert child_train == parent_train
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: the 3 new tests FAIL — `_SIGNALS_PATH` doesn't exist yet in `run_signals_isolated.py` (`ImportError`), and the other constants still equal the stale `data/cache/...` literals; the 13 tests from Tasks 1-4 still PASS.

- [ ] **Step 3: Fix `scripts/signal_one_strategy.py`**

Modify `scripts/signal_one_strategy.py:9-29`:

```diff
 from __future__ import annotations
 import json
 import logging
 import sys
 from datetime import datetime, timezone
 from pathlib import Path

 import polars as pl

+from config.markets import get_market
 from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
 from scripts.build_features import build_live_features
 from src.backtesting.strategy_runner import _is_stateless, _select_cols
 from src.features.duckdb_client import load_training_data
 from src.strategies.registry import load_strategy

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_OUT_DIR = Path("data/cache/signals_partial")
-_LIVE_CACHE = Path("data/cache/_tmp_live_features.parquet")
-_TRAIN_CACHE = Path("data/cache/_tmp_training_data.parquet")
+_CACHE_DIR = get_market("us").data_root / "cache"
+_OUT_DIR = _CACHE_DIR / "signals_partial"
+_LIVE_CACHE = _CACHE_DIR / "_tmp_live_features.parquet"
+_TRAIN_CACHE = _CACHE_DIR / "_tmp_training_data.parquet"
```

- [ ] **Step 4: Fix `scripts/run_signals_isolated.py`**

Modify `scripts/run_signals_isolated.py:19-42` (imports and constants):

```diff
 from __future__ import annotations
 import json
 import logging
 import subprocess
 import sys
 import threading
 import time
 from datetime import datetime, timezone
 from pathlib import Path

 import psutil

+from config.markets import get_market
 from dashboard.config import PARQUET_DIR
 from scripts.build_features import build_live_features
 from src.features.duckdb_client import load_training_data
 from src.strategies.registry import list_strategies

 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 logger = logging.getLogger(__name__)

-_PARTIAL_DIR = Path("data/cache/signals_partial")
-_LIVE_CACHE = Path("data/cache/_tmp_live_features.parquet")
-_TRAIN_CACHE = Path("data/cache/_tmp_training_data.parquet")
+_CACHE_DIR = get_market("us").data_root / "cache"
+_PARTIAL_DIR = _CACHE_DIR / "signals_partial"
+_LIVE_CACHE = _CACHE_DIR / "_tmp_live_features.parquet"
+_TRAIN_CACHE = _CACHE_DIR / "_tmp_training_data.parquet"
+_SIGNALS_PATH = _CACHE_DIR / "signals.json"
 _MIN_FREE_GB = 3.0
 _POLL_SECONDS = 1.0
 _EXCLUDE = {"knn_classifier"}
```

Then modify the inline literal in `_merge_and_write()` at `scripts/run_signals_isolated.py:149-161`:

```diff
 def _merge_and_write() -> None:
     all_signals: list[dict] = []
     for f in sorted(_PARTIAL_DIR.glob("*.json")):
         data = json.loads(f.read_text())
         all_signals.extend(data["signals"])

     buy_count = sum(1 for s in all_signals if s["signal"] == "Buy")
     logger.info("  total signals: %d  buy: %d", len(all_signals), buy_count)
-    Path("data/cache/signals.json").write_text(json.dumps({
+    _SIGNALS_PATH.write_text(json.dumps({
         "generated_at": datetime.now(timezone.utc).isoformat(),
         "signals": all_signals,
     }, indent=2, default=str))
-    logger.info("  wrote data/cache/signals.json")
+    logger.info("  wrote %s", _SIGNALS_PATH)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_data_paths.py -v`
Expected: 16 passed

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: `348 passed, 15 failed, 2 skipped`. Same 15 pre-existing failures.

- [ ] **Step 7: Commit**

```bash
git add scripts/signal_one_strategy.py scripts/run_signals_isolated.py tests/unit/test_data_paths.py
git commit -m "fix: resolve isolated-signal-generation cache paths through market registry"
```

---

## What this plan does NOT do (deliberately deferred)

- No `--market` CLI flag anywhere — that's for whichever future plan actually adds China data (nothing to point the flag at yet).
- No dashboard market switcher (selectbox, per-request path switching) — the dashboard still only ever shows US data; this plan only makes the US path resolve correctly again.
- Does not touch `dags/feature_engineering_dag.py` or `dags/model_retrain_dag.py` — those hardcode a *different*, Airflow-container-specific path (`/opt/airflow/data/...`), entirely independent of the local `markets/us/data/` restructuring. They were never broken by the data move and don't need fixing here.
- Does not touch `scripts/build_features.py`'s hardcoded `_STOCK_TICKERS` list, the CSI 300 constituent ingestion, or any China-specific data work.
