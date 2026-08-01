# China Full-Universe Training & Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand China from the 15-ticker pilot to the real, live CSI 500 (500 tickers), generalize `train_models.py` and `precompute_dashboard.py` to run per-market, and actually run the full ingestion → features → training → backtesting pipeline for China.

**Architecture:** Same "add a `--market` flag, resolve paths via `config.markets.get_market(market)`" pattern already used for `build_features.py`/`refresh_data.py` in prior plans. `_STOCK_TICKERS_CHINA` grows from 15 to 500 real tickers. `train_models.py` and `precompute_dashboard.py` gain the same flag; `build_live_features()` (used by `precompute_dashboard.py`'s signal step) is generalized too, since it was previously hardcoded to US paths/tickers and would otherwise silently produce wrong output for China.

**Tech Stack:** Python, polars, scikit-learn/xgboost/lightgbm, pytest.

## Global Constraints

- Only 3 models train on China: `random_forest`, `xgboost`, `lightgbm` (via `train_models.py`). `train_new_models.py`'s full zoo (logistic_regression, naive_bayes, mlp, svm, lstm) stays US-only — not touched in this plan.
- `incremental_train.py` is not touched in this plan.
- No dashboard UI changes. No Airflow DAG changes (`dags/*.py`).
- CSI 500 ticker/name source of truth: `docs/reference/csi500-constituents.csv` (already committed — 500 rows, fetched live from `https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls`, index code `000905`).
- `step_backtests()`'s `step_days` default stays `21` (monthly) — do not change it to any other value.
- Feature-file coverage validation floor: **90%** of the 500 tickers must have a valid feature parquet (some yfinance misses at this scale are expected and tolerated).
- Benchmark ticker for China stays `510300.SS` (unchanged from the prior plan) regardless of which constituent universe is used.

---

### Task 1: Expand China ticker universe to the real CSI 500

**Files:**
- Modify: `scripts/build_features.py:137-146` (`_STOCK_TICKERS_CHINA` list)
- Modify: `config/markets.py:29-37` (`label`, `universe` fields on the `china` `MarketConfig`)
- Modify: `config/stocks.yaml:5-22` (rename `csi300` section to `csi500`, update tickers/description)
- Test: `tests/unit/test_build_features.py:84-91`
- Test: `tests/unit/test_markets.py:19-26`

**Interfaces:**
- Consumes: `docs/reference/csi500-constituents.csv` (columns `ticker,chinese_name,english_name`, 500 rows, already committed).
- Produces: `_STOCK_TICKERS_CHINA: list[str]` (500 entries) — consumed by `_TICKERS_BY_MARKET["china"]`, which `refresh_data.py`, `build_features.py`, and (after Task 3) `build_live_features()` all read.

- [ ] **Step 1: Update the two existing tests to expect 500 tickers and the renamed universe**

In `tests/unit/test_build_features.py`, replace the existing `test_tickers_by_market_has_expected_markets` function with:

```python
def test_tickers_by_market_has_expected_markets():
    from scripts.build_features import _STOCK_TICKERS, _STOCK_TICKERS_CHINA, _TICKERS_BY_MARKET
    assert _TICKERS_BY_MARKET["us"] == _STOCK_TICKERS
    assert _TICKERS_BY_MARKET["china"] == _STOCK_TICKERS_CHINA
    assert len(_STOCK_TICKERS_CHINA) == 500
    assert all(t.endswith(".SS") or t.endswith(".SZ") for t in _STOCK_TICKERS_CHINA)
    assert len(set(_STOCK_TICKERS_CHINA)) == 500  # no duplicates
```

In `tests/unit/test_markets.py`, change line 23 from:

```python
    assert china.universe == "csi300"
```

to:

```python
    assert china.universe == "csi500"
```

- [ ] **Step 2: Run the tests to confirm they fail against the current (15-ticker, csi300) code**

Run: `python -m pytest tests/unit/test_build_features.py::test_tickers_by_market_has_expected_markets tests/unit/test_markets.py::test_china_market_config -v`
Expected: both FAIL (length is 15 not 500; universe is "csi300" not "csi500").

- [ ] **Step 3: Regenerate the ticker list from the committed CSV and rewrite the two target files**

Create a temporary script at `scratch_expand_csi500.py` in the repo root with this exact content:

```python
import csv

with open("docs/reference/csi500-constituents.csv", encoding="utf-8-sig") as f:
    tickers = [row["ticker"] for row in csv.DictReader(f)]
assert len(tickers) == 500, f"expected 500 tickers, got {len(tickers)}"
assert len(set(tickers)) == 500, "duplicate tickers found"

py_lines = []
for i in range(0, len(tickers), 8):
    chunk = tickers[i:i + 8]
    py_lines.append("    " + ", ".join(f'"{t}"' for t in chunk) + ",")
py_block = "\n".join(py_lines)

yaml_block = "\n".join(f'      - "{t}"' for t in tickers)

bf_path = "scripts/build_features.py"
bf_text = open(bf_path, encoding="utf-8").read()
old_bf = '''_STOCK_TICKERS_CHINA: list[str] = [
    "600519.SS", "601318.SS", "600036.SS", "601398.SS", "000858.SZ",
    "000333.SZ", "002594.SZ", "300750.SZ", "600887.SS", "601012.SS",
    "002415.SZ", "300059.SZ", "601888.SS", "600030.SS", "000651.SZ",
]'''
assert old_bf in bf_text, "old China ticker block not found in build_features.py"
new_bf = f"_STOCK_TICKERS_CHINA: list[str] = [\n{py_block}\n]"
open(bf_path, "w", encoding="utf-8").write(bf_text.replace(old_bf, new_bf))

yaml_path = "config/stocks.yaml"
yaml_text = open(yaml_path, encoding="utf-8").read()
old_yaml = '''  csi300:
    description: "CSI 300 pilot subset (15 tickers, China A-share) — full 300-constituent list is a follow-up plan"
    tickers:
      - "600519.SS"
      - "601318.SS"
      - "600036.SS"
      - "601398.SS"
      - "000858.SZ"
      - "000333.SZ"
      - "002594.SZ"
      - "300750.SZ"
      - "600887.SS"
      - "601012.SS"
      - "002415.SZ"
      - "300059.SZ"
      - "601888.SS"
      - "600030.SS"
      - "000651.SZ"'''
assert old_yaml in yaml_text, "old csi300 section not found in stocks.yaml"
new_yaml = (
    '  csi500:\n'
    '    description: "CSI 500 constituents (500 tickers, China A-share, mid-cap) '
    '— sourced live from csindex.com.cn index code 000905 on 2026-07-31"\n'
    '    tickers:\n' + yaml_block
)
open(yaml_path, "w", encoding="utf-8").write(yaml_text.replace(old_yaml, new_yaml))

print("build_features.py and stocks.yaml updated:", len(tickers), "tickers")
```

Run it from the repo root: `python scratch_expand_csi500.py`
Expected output: `build_features.py and stocks.yaml updated: 500 tickers`

Then delete the temporary script: remove `scratch_expand_csi500.py` (it is not part of the codebase — it was only a one-time data-transformation step).

- [ ] **Step 4: Update `config/markets.py`'s China `MarketConfig`**

In `config/markets.py`, change:

```python
    "china": MarketConfig(
        name="china",
        label="China A-Share (CSI 300)",
        data_root=_REPO_ROOT / "markets" / "china" / "data",
        universe="csi300",
        benchmark_ticker="510300.SS",
        vol_index_ticker=None,
        currency="CNY",
    ),
```

to:

```python
    "china": MarketConfig(
        name="china",
        label="China A-Share (CSI 500)",
        data_root=_REPO_ROOT / "markets" / "china" / "data",
        universe="csi500",
        benchmark_ticker="510300.SS",
        vol_index_ticker=None,
        currency="CNY",
    ),
```

- [ ] **Step 5: Run the tests again to confirm they pass, and sanity-check the rewritten files parse**

Run: `python -m pytest tests/unit/test_build_features.py::test_tickers_by_market_has_expected_markets tests/unit/test_markets.py -v`
Expected: all PASS.

Run: `python -c "import ast; ast.parse(open('scripts/build_features.py', encoding='utf-8').read()); print('build_features.py OK')"`
Run: `python -c "import yaml; d = yaml.safe_load(open('config/stocks.yaml', encoding='utf-8')); assert len(d['universes']['csi500']['tickers']) == 500; print('stocks.yaml OK')"`
Expected: both print their `OK` message with no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_features.py config/markets.py config/stocks.yaml tests/unit/test_build_features.py tests/unit/test_markets.py
git commit -m "feat: expand China ticker universe from 15-pilot to real CSI 500 (500 tickers)"
```

---

### Task 2: Add `--market` flag to `train_models.py`

**Files:**
- Modify: `scripts/train_models.py`
- Test: `tests/unit/test_train_models.py`

**Interfaces:**
- Consumes: `config.markets.get_market(market) -> MarketConfig` (existing).
- Produces: `_market_paths(market: str) -> tuple[Path, Path]` returning `(feature_dir, registry_dir)` — a new small helper other code doesn't need to know about, but the task's own tests exercise directly. `main(market: str = "us")` — existing callers with no args are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_train_models.py`:

```python
def test_market_paths_for_us():
    from scripts.train_models import _market_paths
    from config.markets import get_market
    feature_dir, registry_dir = _market_paths("us")
    assert feature_dir == get_market("us").data_root / "features"
    assert registry_dir == get_market("us").data_root / "registry"


def test_market_paths_for_china():
    from scripts.train_models import _market_paths
    from config.markets import get_market
    feature_dir, registry_dir = _market_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert registry_dir == get_market("china").data_root / "registry"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_train_models.py::test_market_paths_for_us tests/unit/test_train_models.py::test_market_paths_for_china -v`
Expected: FAIL with `ImportError: cannot import name '_market_paths'`.

- [ ] **Step 3: Rewrite `scripts/train_models.py`**

Replace the full file content with:

```python
from pathlib import Path
import logging

import numpy as np
import polars as pl

from config.markets import MARKETS, get_market
from dashboard.ui_config import FEATURE_COLS
from src.features.duckdb_client import load_training_data
from src.models.base_classifier import BaseClassifier
from src.models.evaluator import evaluate
from src.models.registry import save_model
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.xgboost_model import XGBoostClassifier
from src.models.zoo.lightgbm_model import LightGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = get_market("us").data_root / "features"
_TRAIN_RATIO = 0.8


def _market_paths(market: str) -> tuple[Path, Path]:
    market_cfg = get_market(market)
    return market_cfg.data_root / "features", market_cfg.data_root / "registry"


def train_and_save(
    model: BaseClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
    registry_dir: Path,
) -> Path:
    clean = df.drop_nulls(subset=feature_cols + ["label"]).sort("time")
    if len(clean) == 0:
        raise ValueError("No training data after dropping nulls")

    split = int(len(clean) * _TRAIN_RATIO)
    train_df = clean[:split]
    test_df  = clean[split:]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test  = test_df.select(feature_cols).to_numpy()
    y_test  = test_df["label"].to_numpy()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluation = evaluate(y_test, y_pred)

    path = save_model(
        model=model,
        evaluation=evaluation,
        params=model.default_params,
        feature_cols=feature_cols,
        registry_dir=registry_dir,
    )
    logger.info(
        "Saved %s — acc=%.3f  prec_buy=%.3f  f1_macro=%.3f",
        model.name, evaluation.accuracy, evaluation.precision_buy, evaluation.f1_macro,
    )
    return path


def main(market: str = "us") -> None:
    feature_dir, registry_dir = _market_paths(market)

    if not feature_dir.exists() or not any(feature_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No feature parquets found in {feature_dir}/. "
            "Run scripts/build_features.py first."
        )

    logger.info("Loading feature data from %s ...", feature_dir)
    df = load_training_data(feature_dir)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    registry_dir.mkdir(parents=True, exist_ok=True)

    models: list[BaseClassifier] = [
        RandomForestClassifier_(),
        XGBoostClassifier(),
        LightGBMClassifier(),
    ]

    for model in models:
        logger.info("Training %s ...", model.name)
        try:
            train_and_save(model, df, FEATURE_COLS, registry_dir)
        except Exception as exc:
            logger.error("FAILED %s: %s", model.name, exc)

    print("\nTraining complete. Registry contents:")
    for p in sorted(registry_dir.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
```

Note: `_FEATURE_DIR` stays as a module-level constant (unchanged value, US default) because `tests/unit/test_data_paths.py::test_train_models_feature_dir_resolves_under_markets_us_data` imports it directly — it's just no longer read by `main()`, which now resolves its own paths per-market via `_market_paths()`.

- [ ] **Step 4: Run the tests to verify they pass, plus the full existing test_train_models.py and test_data_paths.py suites**

Run: `python -m pytest tests/unit/test_train_models.py tests/unit/test_data_paths.py -v`
Expected: all PASS (including the pre-existing `test_train_models_feature_dir_resolves_under_markets_us_data`).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_models.py tests/unit/test_train_models.py
git commit -m "feat: add --market flag to train_models.py"
```

---

### Task 3: Generalize `build_live_features()` for `--market`

**Files:**
- Modify: `scripts/build_features.py` (the `build_live_features` function only)
- Test: `tests/unit/test_build_features.py`

**Interfaces:**
- Consumes: `_TICKERS_BY_MARKET`, `config.markets.get_market`, `src.features.cross_asset_features.synthetic_vol_index` (all existing, already imported in this file).
- Produces: `build_live_features(market: str = "us") -> pl.DataFrame` — signature changes from `build_live_features(raw_dir: Path = _RAW_DIR)`. The three existing callers (`scripts/precompute_dashboard.py`, `scripts/signal_one_strategy.py`, `scripts/run_signals_isolated.py`) all call it with zero arguments today, so the new default (`market="us"`) preserves their exact current behavior — none of them need to change in this task.

This task exists because `build_live_features()` currently hardcodes `raw_dir / "SPY.parquet"`, `raw_dir / "^VIX.parquet"`, and loops over the module-level `_STOCK_TICKERS` (the US list) regardless of what `raw_dir` is passed — it was never actually market-parameterized, unlike every other function in this file. Left as-is, Task 4's `precompute_dashboard.py --market china` would silently compute wrong (or crash on missing US-named files) live signals for China.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_build_features.py`:

```python
def test_build_live_features_default_market_is_us():
    import inspect
    from scripts.build_features import build_live_features
    sig = inspect.signature(build_live_features)
    assert sig.parameters["market"].default == "us"


def test_build_live_features_china_reads_from_china_raw_dir_and_tickers():
    from scripts.build_features import build_live_features, _STOCK_TICKERS_CHINA
    df = build_live_features(market="china")
    assert len(df) > 0
    assert set(df["ticker"].unique().to_list()).issubset(set(_STOCK_TICKERS_CHINA))
```

(The second test relies on the China pilot's real raw OHLCV data already committed under `markets/china/data/raw/ohlcv/` from the prior plan — at least the 15 pilot tickers' `.parquet` files exist there today, which is enough for `build_live_features` to return a non-empty frame.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_build_features.py::test_build_live_features_default_market_is_us tests/unit/test_build_features.py::test_build_live_features_china_reads_from_china_raw_dir_and_tickers -v`
Expected: the first FAILs with `KeyError` (no `market` parameter exists yet); the second FAILs (function signature doesn't accept `market=`).

- [ ] **Step 3: Replace `build_live_features` in `scripts/build_features.py`**

Replace this existing function:

```python
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
```

with:

```python
def build_live_features(market: str = "us") -> pl.DataFrame:
    """Full per-ticker feature history through the latest raw trading day,
    with no label-driven trim on the tail. Used only for live signals —
    training/backtesting must keep using the labeled markets/<market>/data/features/*.parquet
    (via load_training_data) so their results stay unaffected."""
    market_cfg = get_market(market)
    raw_dir = market_cfg.data_root / "raw" / "ohlcv"
    tickers = _TICKERS_BY_MARKET[market]

    benchmark_df = pl.read_parquet(raw_dir / f"{market_cfg.benchmark_ticker}.parquet")
    if market_cfg.vol_index_ticker:
        vix_df = pl.read_parquet(raw_dir / f"{market_cfg.vol_index_ticker}.parquet")
    else:
        vix_df = synthetic_vol_index(benchmark_df)

    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            continue
        try:
            frames.append(build_features_for_ticker(
                ticker, raw_dir, benchmark_df, vix_df, drop_label_nulls=False,
            ))
        except Exception as exc:
            logger.warning("  live features FAILED %s: %s", ticker, exc)
    return pl.concat(frames, how="vertical_relaxed")
```

- [ ] **Step 4: Run the tests to verify they pass, plus the full test_build_features.py suite**

Run: `python -m pytest tests/unit/test_build_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_features.py tests/unit/test_build_features.py
git commit -m "fix: generalize build_live_features() for --market (was silently US-only)"
```

---

### Task 4: Add `--market` flag to `precompute_dashboard.py`

**Files:**
- Modify: `scripts/precompute_dashboard.py`
- Test: new tests in `tests/unit/test_data_paths.py`

**Interfaces:**
- Consumes: `_market_paths` pattern (same shape as Task 2's, but returns `(feature_dir, cache_dir)` here); `build_live_features(market=...)` from Task 3.
- Produces: `main(market: str = "us")`. `step_data_summary`, `step_backtests`, `step_leaderboard`, `step_signals` all gain optional `feature_dir`/`cache_dir` (and `step_signals` also gains `market`) parameters, **defaulting to the existing US module constants** — this is required so that `scripts/backtest_new_strategies.py` and `scripts/backtest_stacking_only.py` (which call `step_backtests(...)`, `step_leaderboard()`, `step_signals(...)` directly with no path arguments) keep working completely unchanged. Do not modify those two scripts.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_data_paths.py`:

```python
def test_precompute_dashboard_market_paths_for_us():
    from scripts.precompute_dashboard import _market_paths
    feature_dir, cache_dir = _market_paths("us")
    assert feature_dir == _US_ROOT / "features"
    assert cache_dir == _US_ROOT / "cache"


def test_precompute_dashboard_market_paths_for_china():
    from scripts.precompute_dashboard import _market_paths
    from config.markets import get_market
    feature_dir, cache_dir = _market_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert cache_dir == get_market("china").data_root / "cache"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_data_paths.py::test_precompute_dashboard_market_paths_for_us tests/unit/test_data_paths.py::test_precompute_dashboard_market_paths_for_china -v`
Expected: FAIL with `ImportError: cannot import name '_market_paths'`.

- [ ] **Step 3: Modify `scripts/precompute_dashboard.py`**

Change the imports (add `MARKETS`):

```python
from config.markets import get_market
```

to:

```python
from config.markets import MARKETS, get_market
```

Change the module docstring's second line from:

```
Run locally to pre-compute all dashboard data and write JSON to markets/us/data/cache/.
```

to:

```
Run locally to pre-compute all dashboard data and write JSON to the given
market's markets/<market>/data/cache/ (default: us).
```

Add this helper right after the `CACHE_DIR = get_market("us").data_root / "cache"` line:

```python
def _market_paths(market: str) -> tuple[Path, Path]:
    market_cfg = get_market(market)
    return market_cfg.data_root / "features", market_cfg.data_root / "cache"
```

Change `step_data_summary`'s signature and body from:

```python
def step_data_summary() -> None:
    logger.info("[1/4] data summary")
    # Read directly from parquets — never read from cache here, that defeats the purpose.
    df = load_training_data(PARQUET_DIR)
    tickers = sorted(df["ticker"].unique().to_list()) if "ticker" in df.columns else []
    summary = {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": tickers,
        "date_range_start": str(df["time"].min()) if "time" in df.columns else "N/A",
        "date_range_end":   str(df["time"].max()) if "time" in df.columns else "N/A",
        "generated_at": _now(),
    }
    _write(CACHE_DIR / "data_summary.json", summary)
```

to:

```python
def step_data_summary(feature_dir: Path = PARQUET_DIR, cache_dir: Path = CACHE_DIR) -> None:
    logger.info("[1/4] data summary")
    # Read directly from parquets — never read from cache here, that defeats the purpose.
    df = load_training_data(feature_dir)
    tickers = sorted(df["ticker"].unique().to_list()) if "ticker" in df.columns else []
    summary = {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": tickers,
        "date_range_start": str(df["time"].min()) if "time" in df.columns else "N/A",
        "date_range_end":   str(df["time"].max()) if "time" in df.columns else "N/A",
        "generated_at": _now(),
    }
    _write(cache_dir / "data_summary.json", summary)
```

Change `step_leaderboard`'s signature and body from:

```python
def step_leaderboard() -> None:
    logger.info("[3/4] leaderboard — aggregating from per-strategy backtest caches")
    # Build the leaderboard from the individual backtest files written in step_backtests().
    # Never call get_leaderboard() here — it reads the old leaderboard.json and writes it back.
    grades: list[dict] = []
    for name in list_strategies():
        cache_path = CACHE_DIR / f"backtest_{_safe(name)}.json"
        if cache_path.exists():
            d = json.loads(cache_path.read_text())
            grades.append(d["grade"])
        else:
            logger.warning("  no backtest cache for %s — skipping from leaderboard", name)

    grades.sort(key=lambda g: g["composite_score"], reverse=True)
    _write(CACHE_DIR / "leaderboard.json", {
        "generated_at": _now(),
        "grades": grades,
    })
```

to:

```python
def step_leaderboard(cache_dir: Path = CACHE_DIR) -> None:
    logger.info("[3/4] leaderboard — aggregating from per-strategy backtest caches")
    # Build the leaderboard from the individual backtest files written in step_backtests().
    # Never call get_leaderboard() here — it reads the old leaderboard.json and writes it back.
    grades: list[dict] = []
    for name in list_strategies():
        cache_path = cache_dir / f"backtest_{_safe(name)}.json"
        if cache_path.exists():
            d = json.loads(cache_path.read_text())
            grades.append(d["grade"])
        else:
            logger.warning("  no backtest cache for %s — skipping from leaderboard", name)

    grades.sort(key=lambda g: g["composite_score"], reverse=True)
    _write(cache_dir / "leaderboard.json", {
        "generated_at": _now(),
        "grades": grades,
    })
```

Change `step_backtests`'s signature and body from:

```python
def step_backtests(names: list[str] | None = None, step_days: int = 21) -> None:
    logger.info("[2/4] per-strategy backtests — running walk-forward fresh (no cache read)")
    # Load data once; reuse across all strategies.
    df = load_training_data(PARQUET_DIR)
    for name in (names if names is not None else list_strategies()):
```

to:

```python
def step_backtests(
    names: list[str] | None = None,
    step_days: int = 21,
    feature_dir: Path = PARQUET_DIR,
    cache_dir: Path = CACHE_DIR,
) -> None:
    logger.info("[2/4] per-strategy backtests — running walk-forward fresh (no cache read)")
    # Load data once; reuse across all strategies.
    df = load_training_data(feature_dir)
    for name in (names if names is not None else list_strategies()):
```

(leave the rest of `step_backtests`'s body unchanged except the two `CACHE_DIR / f"backtest_{_safe(name)}.json"` occurrences later in the same function — change both to `cache_dir / f"backtest_{_safe(name)}.json"`.)

Change `step_signals`'s signature and body from:

```python
def step_signals(exclude: list[str] | None = None) -> None:
    logger.info("[4/4] live signals — computing fresh from all strategies (no cache read)")
    # Fit on the labeled/trimmed training data — unchanged, since fit() needs
    # a real label and this keeps training/backtesting behavior untouched.
    df = load_training_data(PARQUET_DIR)

    # Predict on a separately-built, un-trimmed feature snapshot so "today's"
    # signal actually reflects the latest raw trading day instead of trailing
    # by forward_days (the labeled parquet can't have a label for those rows,
    # so it drops them — but predict() never needs a label at all).
    live_df = build_live_features()
```

to:

```python
def step_signals(
    exclude: list[str] | None = None,
    market: str = "us",
    feature_dir: Path = PARQUET_DIR,
    cache_dir: Path = CACHE_DIR,
) -> None:
    logger.info("[4/4] live signals — computing fresh from all strategies (no cache read)")
    # Fit on the labeled/trimmed training data — unchanged, since fit() needs
    # a real label and this keeps training/backtesting behavior untouched.
    df = load_training_data(feature_dir)

    # Predict on a separately-built, un-trimmed feature snapshot so "today's"
    # signal actually reflects the latest raw trading day instead of trailing
    # by forward_days (the labeled parquet can't have a label for those rows,
    # so it drops them — but predict() never needs a label at all).
    live_df = build_live_features(market=market)
```

Then later in the same function, change:

```python
    _write(CACHE_DIR / "signals.json", {
        "generated_at": _now(),
        "signals": all_signals,
    })
```

to:

```python
    _write(cache_dir / "signals.json", {
        "generated_at": _now(),
        "signals": all_signals,
    })
```

Finally, change `main()` and the CLI block from:

```python
def main() -> None:
    logger.info("=== Precomputing dashboard cache → %s ===", CACHE_DIR)
    step_data_summary()   # [1/4] parquet → data_summary.json
    step_backtests()      # [2/4] per-strategy walk-forward → backtest_*.json
    step_leaderboard()    # [3/4] aggregate backtest_*.json → leaderboard.json
    step_signals()        # [4/4] live signals → signals.json
    logger.info("=== Done. Run: git add markets/us/data/cache/ && git commit && git push ===")


if __name__ == "__main__":
    main()
```

to:

```python
def main(market: str = "us") -> None:
    feature_dir, cache_dir = _market_paths(market)
    logger.info("=== Precomputing dashboard cache (%s) → %s ===", market, cache_dir)
    step_data_summary(feature_dir, cache_dir)                                   # [1/4]
    step_backtests(feature_dir=feature_dir, cache_dir=cache_dir)                # [2/4]
    step_leaderboard(cache_dir)                                                 # [3/4]
    step_signals(market=market, feature_dir=feature_dir, cache_dir=cache_dir)   # [4/4]
    logger.info("=== Done. Run: git add %s/ && git commit && git push ===", cache_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    args = parser.parse_args()
    main(args.market)
```

- [ ] **Step 4: Run the tests to verify they pass, plus confirm `backtest_new_strategies.py`/`backtest_stacking_only.py` still import cleanly**

Run: `python -m pytest tests/unit/test_data_paths.py -v`
Expected: all PASS, including the two new tests.

Run: `python -c "import scripts.backtest_new_strategies, scripts.backtest_stacking_only; print('both import OK')"`
Expected: prints `both import OK` with no errors (confirms their unchanged zero-arg calls to `step_backtests`/`step_leaderboard`/`step_signals` still type-check against the new optional-parameter signatures).

- [ ] **Step 5: Commit**

```bash
git add scripts/precompute_dashboard.py tests/unit/test_data_paths.py
git commit -m "feat: add --market flag to precompute_dashboard.py"
```

---

### Task 5: Real ingestion + feature build for the full China universe

**Files:**
- No source changes — this task runs the already-generalized pipeline for real and adds validation tests.
- Delete: `tests/unit/test_china_pilot_features.py`
- Create: `tests/unit/test_china_features.py`

**Interfaces:**
- Consumes: `scripts/refresh_data.py --market china`, `scripts/build_features.py --market china` (both already support `--market` from the prior plan).
- Produces: `markets/china/data/raw/ohlcv/*.parquet` and `markets/china/data/features/*.parquet` for up to 500 tickers (some misses expected and tolerated, per the 90% coverage floor in Global Constraints).

- [ ] **Step 1: Run the real ingestion and feature-build commands**

Run (this fetches ~500 tickers from yfinance and will take a while — run it and wait for it to finish; do not interrupt it):

```bash
python scripts/refresh_data.py --market china
python scripts/build_features.py --market china
```

Expected: both commands finish (exit code 0) and print per-ticker OK/FAIL summaries. Some `FAIL` lines are expected and fine — check the final "Done. N succeeded, M failed." line and confirm `N / (N+M) >= 0.90`.

- [ ] **Step 2: Delete the old pilot-scoped test file**

Delete `tests/unit/test_china_pilot_features.py` — it iterated unconditionally over all 15 pilot tickers, assuming every one has a feature file. That assumption doesn't hold at 500-ticker scale (some misses are expected), so it's being replaced by `test_china_features.py` below, not just extended.

- [ ] **Step 3: Write the new validation test file**

Create `tests/unit/test_china_features.py`:

```python
from pathlib import Path

import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS
from scripts.build_features import _STOCK_TICKERS_CHINA

_CHINA_FEATURES_DIR = get_market("china").data_root / "features"
_COVERAGE_FLOOR = 0.90
_NULL_RATE_CEILING = 0.65


def _existing_feature_paths() -> list[tuple[str, Path]]:
    return [
        (ticker, _CHINA_FEATURES_DIR / f"{ticker}.parquet")
        for ticker in _STOCK_TICKERS_CHINA
        if (_CHINA_FEATURES_DIR / f"{ticker}.parquet").exists()
    ]


def test_china_feature_coverage_meets_floor():
    existing = _existing_feature_paths()
    coverage = len(existing) / len(_STOCK_TICKERS_CHINA)
    assert coverage >= _COVERAGE_FLOOR, (
        f"only {len(existing)}/{len(_STOCK_TICKERS_CHINA)} "
        f"({coverage:.1%}) tickers have feature files, floor is {_COVERAGE_FLOOR:.0%}"
    )


def test_china_features_have_full_schema_and_no_null_labels():
    existing = _existing_feature_paths()
    assert len(existing) > 0
    for ticker, path in existing:
        df = pl.read_parquet(path)
        for col in FEATURE_COLS:
            assert col in df.columns, f"{ticker} missing feature col {col}"
        assert "label" in df.columns
        assert "forward_return_5d" in df.columns
        assert df["label"].null_count() == 0
        assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
        assert len(df) > 0


def test_china_features_have_no_weekend_timestamps():
    # Regression guard for the timezone bug (fixed in the prior plan) that
    # shifted every China date back one calendar day, landing bars on
    # Sunday. Chinese exchanges trade Monday-Friday only.
    existing = _existing_feature_paths()
    for ticker, path in existing:
        df = pl.read_parquet(path)
        weekdays = set(df.select(pl.col("time").dt.weekday()).to_series().unique().to_list())
        assert weekdays.issubset({1, 2, 3, 4, 5}), f"{ticker} has weekend timestamps: {weekdays}"


def test_china_benchmark_derived_columns_null_rate_is_bounded():
    # vix_level and rel_strength_spy both derive from the benchmark join;
    # tickers listed before the benchmark's own history start will have
    # leading nulls in both — expected and bounded, not a defect.
    existing = _existing_feature_paths()
    for ticker, path in existing:
        df = pl.read_parquet(path)
        for col in ("vix_level", "rel_strength_spy"):
            null_rate = df[col].null_count() / len(df)
            assert null_rate <= _NULL_RATE_CEILING, f"{ticker}.{col} null rate {null_rate:.1%} exceeds ceiling"
```

- [ ] **Step 4: Run the new test file to verify it passes against the real data just ingested**

Run: `python -m pytest tests/unit/test_china_features.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit the code changes and the real data together**

```bash
git add tests/unit/test_china_features.py
git rm tests/unit/test_china_pilot_features.py
git add markets/china/data/raw markets/china/data/features
git commit -m "data: expand China ingestion to full CSI 500 universe (500 tickers), validated"
```

---

### Task 6: Real training + backtesting run for China

**Files:**
- No source changes — this task runs the already-generalized pipeline for real and adds validation tests.
- Create: `tests/unit/test_china_training_output.py`

**Interfaces:**
- Consumes: `scripts/train_models.py --market china` (Task 2), `scripts/precompute_dashboard.py --market china` (Task 4, which internally calls the Task 3-generalized `build_live_features(market="china")`).
- Produces: `markets/china/data/registry/{random_forest,xgboost,lightgbm}/*.json` and `markets/china/data/cache/{data_summary,leaderboard,signals,backtest_*}.json`.

- [ ] **Step 1: Run the real training and precompute commands**

Run (trains 3 models on the full China feature set, then runs walk-forward backtests for every registered strategy at `step_days=21` plus leaderboard/signal generation — this will take a while; do not interrupt it):

```bash
python scripts/train_models.py --market china
python scripts/precompute_dashboard.py --market china
```

Expected: both commands finish (exit code 0). `train_models.py`'s final "Training complete. Registry contents:" listing shows 3 model directories under `markets/china/data/registry/`. `precompute_dashboard.py`'s log shows all 4 steps (`[1/4]` through `[4/4]`) completing.

- [ ] **Step 2: Write the validation test file**

Create `tests/unit/test_china_training_output.py`:

```python
import json

from config.markets import get_market
from src.strategies.registry import list_strategies

_CHINA_REGISTRY_DIR = get_market("china").data_root / "registry"
_CHINA_CACHE_DIR = get_market("china").data_root / "cache"
_EXPECTED_MODELS = ["random_forest", "xgboost", "lightgbm"]


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def test_china_registry_has_all_three_core_models():
    for name in _EXPECTED_MODELS:
        model_dir = _CHINA_REGISTRY_DIR / name
        assert model_dir.exists(), f"no registry dir for {name}"
        json_files = list(model_dir.glob("*.json"))
        assert len(json_files) > 0, f"no saved model json for {name}"


def test_china_cache_has_leaderboard_and_signals():
    leaderboard_path = _CHINA_CACHE_DIR / "leaderboard.json"
    signals_path = _CHINA_CACHE_DIR / "signals.json"
    assert leaderboard_path.exists()
    assert signals_path.exists()

    leaderboard = json.loads(leaderboard_path.read_text())
    assert len(leaderboard["grades"]) > 0

    signals = json.loads(signals_path.read_text())
    assert len(signals["signals"]) > 0


def test_china_cache_has_backtest_file_per_strategy():
    for name in list_strategies():
        backtest_path = _CHINA_CACHE_DIR / f"backtest_{_safe(name)}.json"
        assert backtest_path.exists(), f"missing backtest cache for {name}"
```

- [ ] **Step 3: Run the new test file to verify it passes against the real training/backtest output**

Run: `python -m pytest tests/unit/test_china_training_output.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 4: Run the full test suite to confirm no regressions anywhere**

Run: `python -m pytest tests/unit -q`
Expected: pass count at or above the pre-plan baseline (357 passed, 15 pre-existing failures unrelated to this plan, 2 skipped), plus the new tests added across all 6 tasks.

- [ ] **Step 5: Commit the code changes and the real data together**

```bash
git add tests/unit/test_china_training_output.py
git add markets/china/data/registry markets/china/data/cache
git commit -m "data: train 3 core models and run full backtest/leaderboard/signals for China CSI 500"
```
