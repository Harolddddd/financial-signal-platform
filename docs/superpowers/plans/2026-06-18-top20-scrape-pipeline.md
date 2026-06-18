# Top-20 Scrape & End-to-End Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape OHLCV data for the top 20 S&P 500 stocks + SPY + ^VIX (2000–2020), build features, train three models, and surface everything in the existing Streamlit dashboard — all from local Parquet files, no TimescaleDB required.

**Architecture:** Three standalone scripts run in sequence: `scrape_top20.py` fetches raw OHLCV via yfinance, `build_features.py` applies technical indicators + cross-asset features and writes `data/features/`, `train_models.py` loads features and saves three trained classifiers to `data/registry/`. The existing dashboard reads from those two directories without modification.

**Tech Stack:** Python 3.11, yfinance 0.2.x, polars, scikit-learn, xgboost, lightgbm, joblib, pytest 8.x

## Global Constraints

- Python >= 3.11; all function signatures require type hints
- Polars for all DataFrame operations; Pandas only where yfinance forces it
- No API keys required — yfinance daily OHLCV is free
- Scripts are re-runnable and idempotent (overwrite existing output files)
- Paths are relative to project root (`data/raw/ohlcv/`, `data/features/`, `data/registry/`)
- Tests use pytest only; no `unittest.TestCase`
- Each script guards `main()` behind `if __name__ == "__main__":` so it is importable in tests

---

### Task 1: `scripts/scrape_top20.py` — Fetch Raw OHLCV

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/scrape_top20.py`
- Create: `tests/unit/test_scrape_top20.py`

**Interfaces:**
- Consumes: `src.ingestion.historical_collector.fetch_ohlcv(ticker, start, end) -> pl.DataFrame`
- Produces:
  - `TICKERS: list[str]` — 20 stock tickers
  - `BENCHMARK_TICKERS: list[str]` — `["SPY", "^VIX"]`
  - `START: datetime`, `END: datetime` — UTC date range
  - `scrape_ticker(ticker: str, start: datetime, end: datetime, output_dir: Path) -> tuple[int, str | None]`
    — returns `(row_count, None)` on success, `(0, error_message)` on failure
  - Output files: `data/raw/ohlcv/<TICKER>.parquet`

- [ ] **Step 1: Create `scripts/__init__.py`**

```python
```
(Empty file — makes `scripts` a package importable in tests.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_scrape_top20.py
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest


_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
_END   = datetime(2020, 12, 31, tzinfo=timezone.utc)

_SAMPLE_DF = pl.DataFrame({
    "time":         ["2020-01-02 00:00:00+0000", "2020-01-03 00:00:00+0000"],
    "ticker":       ["AAPL", "AAPL"],
    "open":         [295.0, 300.0],
    "high":         [300.0, 305.0],
    "low":          [290.0, 295.0],
    "close":        [298.0, 302.0],
    "volume":       [1_000_000, 1_100_000],
    "adj_close":    [298.0, 302.0],
    "dividends":    [0.0, 0.0],
    "stock_splits": [0.0, 0.0],
})


def test_tickers_list_has_20_stocks():
    from scripts.scrape_top20 import TICKERS
    assert len(TICKERS) == 20
    assert "AAPL" in TICKERS
    assert "MSFT" in TICKERS
    assert "NVDA" in TICKERS


def test_benchmark_tickers():
    from scripts.scrape_top20 import BENCHMARK_TICKERS
    assert "SPY" in BENCHMARK_TICKERS
    assert "^VIX" in BENCHMARK_TICKERS


def test_scrape_ticker_writes_parquet_and_returns_row_count(tmp_path):
    with patch("scripts.scrape_top20.fetch_ohlcv", return_value=_SAMPLE_DF):
        from scripts.scrape_top20 import scrape_ticker
        rows, err = scrape_ticker("AAPL", _START, _END, tmp_path)
    assert err is None
    assert rows == 2
    assert (tmp_path / "AAPL.parquet").exists()
    loaded = pl.read_parquet(tmp_path / "AAPL.parquet")
    assert len(loaded) == 2


def test_scrape_ticker_returns_error_on_fetch_failure(tmp_path):
    with patch("scripts.scrape_top20.fetch_ohlcv", side_effect=ValueError("no data")):
        from scripts.scrape_top20 import scrape_ticker
        rows, err = scrape_ticker("FAKE", _START, _END, tmp_path)
    assert rows == 0
    assert "no data" in err
    assert not (tmp_path / "FAKE.parquet").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/test_scrape_top20.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.scrape_top20'`

- [ ] **Step 4: Implement `scripts/scrape_top20.py`**

```python
from datetime import datetime, timezone
import logging
from pathlib import Path

import polars as pl

from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "AVGO", "TSM",
    "JPM", "LLY", "V", "WMT", "UNH",
    "XOM", "MA", "ORCL", "JNJ", "HD",
]
BENCHMARK_TICKERS: list[str] = ["SPY", "^VIX"]

START = datetime(2000, 1, 1, tzinfo=timezone.utc)
END   = datetime(2020, 12, 31, tzinfo=timezone.utc)

_OUTPUT_DIR = Path("data/raw/ohlcv")


def scrape_ticker(
    ticker: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> tuple[int, str | None]:
    try:
        df = fetch_ohlcv(ticker, start, end)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_dir / f"{ticker}.parquet")
        return len(df), None
    except Exception as exc:
        return 0, str(exc)


def main() -> None:
    all_tickers = TICKERS + BENCHMARK_TICKERS
    successes: list[tuple[str, int]] = []
    failures:  list[tuple[str, str]] = []

    for ticker in all_tickers:
        logger.info("Fetching %s ...", ticker)
        rows, err = scrape_ticker(ticker, START, END, _OUTPUT_DIR)
        if err:
            logger.warning("FAILED %s: %s", ticker, err)
            failures.append((ticker, err))
        else:
            logger.info("OK     %s — %d rows", ticker, rows)
            successes.append((ticker, rows))

    print(f"\n{'='*50}")
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    for ticker, rows in successes:
        print(f"  OK    {ticker:10s}  {rows:>6,} rows")
    for ticker, err in failures:
        print(f"  FAIL  {ticker:10s}  {err}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_scrape_top20.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/scrape_top20.py tests/unit/test_scrape_top20.py
git commit -m "feat: scrape_top20 script — fetch OHLCV for 20 stocks + SPY + VIX via yfinance"
```

---

### Task 2: `scripts/build_features.py` — Feature Engineering from Raw Parquet

**Files:**
- Create: `scripts/build_features.py`
- Create: `tests/unit/test_build_features.py`

**Interfaces:**
- Consumes:
  - `data/raw/ohlcv/<TICKER>.parquet` — output from Task 1
  - `src.features.technical_indicators.add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame`
  - `src.features.cross_asset_features.add_cross_asset_features(stock_df, spy_df, vix_df) -> pl.DataFrame`
  - `src.features.label_generator.add_labels(df: pl.DataFrame) -> pl.DataFrame`
- Produces:
  - `add_neutral_sentiment(df: pl.DataFrame) -> pl.DataFrame`
    — adds `sent_pos_avg_3d=0.5`, `sent_pos_avg_5d=0.5`, `sent_pos_avg_10d=0.5`, `sent_pos_mom_3d=0.0`, `news_vol_spike=0`
  - `build_features_for_ticker(ticker, raw_dir, spy_df, vix_df) -> pl.DataFrame`
    — returns DataFrame with all FEATURE_COLS + `forward_return_5d` + `label`
  - Output files: `data/features/<TICKER>.parquet` (one per stock, not for SPY/VIX)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_build_features.py
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest


def _make_ohlcv(ticker: str, n: int = 300) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base + np.cumsum(rng.normal(0, 1, n))
    times = [
        datetime(2010, 1, 1, tzinfo=timezone.utc).replace(
            year=2010 + i // 252, month=1 + (i % 252) // 21, day=1 + i % 21
        )
        for i in range(n)
    ]
    return pl.DataFrame({
        "time":         times,
        "ticker":       [ticker] * n,
        "open":         closes * 0.99,
        "high":         closes * 1.01,
        "low":          closes * 0.98,
        "close":        closes,
        "volume":       [1_000_000] * n,
        "adj_close":    closes,
        "dividends":    [0.0] * n,
        "stock_splits": [0.0] * n,
    })


def test_add_neutral_sentiment_adds_all_columns():
    from scripts.build_features import add_neutral_sentiment
    df = pl.DataFrame({"time": [datetime(2010, 1, 4, tzinfo=timezone.utc)], "ticker": ["AAPL"]})
    out = add_neutral_sentiment(df)
    assert "sent_pos_avg_3d"  in out.columns
    assert "sent_pos_avg_5d"  in out.columns
    assert "sent_pos_avg_10d" in out.columns
    assert "sent_pos_mom_3d"  in out.columns
    assert "news_vol_spike"   in out.columns
    assert out["sent_pos_avg_5d"][0] == pytest.approx(0.5)
    assert out["sent_pos_mom_3d"][0] == pytest.approx(0.0)
    assert out["news_vol_spike"][0] == 0


def test_build_features_for_ticker_returns_required_cols(tmp_path):
    from scripts.build_features import build_features_for_ticker
    from dashboard.config import FEATURE_COLS

    aapl = _make_ohlcv("AAPL")
    spy  = _make_ohlcv("SPY")
    vix  = _make_ohlcv("^VIX")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    aapl.write_parquet(raw_dir / "AAPL.parquet")

    df = build_features_for_ticker("AAPL", raw_dir, spy, vix)

    assert "label" in df.columns
    assert "forward_return_5d" in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature col: {col}"
    assert df["label"].null_count() == 0


def test_build_features_for_ticker_has_no_null_labels(tmp_path):
    from scripts.build_features import build_features_for_ticker

    aapl = _make_ohlcv("AAPL")
    spy  = _make_ohlcv("SPY")
    vix  = _make_ohlcv("^VIX")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    aapl.write_parquet(raw_dir / "AAPL.parquet")

    df = build_features_for_ticker("AAPL", raw_dir, spy, vix)
    assert df["label"].null_count() == 0
    assert set(df["label"].unique().to_list()).issubset({"Buy", "Hold", "Sell"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_build_features.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.build_features'`

- [ ] **Step 3: Implement `scripts/build_features.py`**

```python
from pathlib import Path
import logging

import polars as pl

from src.features.technical_indicators import add_technical_indicators
from src.features.cross_asset_features import add_cross_asset_features
from src.features.label_generator import add_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR     = Path("data/raw/ohlcv")
_FEATURE_DIR = Path("data/features")

_STOCK_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "AVGO", "TSM",
    "JPM", "LLY", "V", "WMT", "UNH",
    "XOM", "MA", "ORCL", "JNJ", "HD",
]


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
) -> pl.DataFrame:
    df = pl.read_parquet(raw_dir / f"{ticker}.parquet")
    df = add_technical_indicators(df)
    df = add_cross_asset_features(df, spy_df, vix_df)
    df = add_neutral_sentiment(df)
    df = add_labels(df)
    return df.drop_nulls(subset=["label"])


def main() -> None:
    spy_path = _RAW_DIR / "SPY.parquet"
    vix_path = _RAW_DIR / "^VIX.parquet"

    if not spy_path.exists() or not vix_path.exists():
        raise FileNotFoundError(
            "SPY.parquet or ^VIX.parquet missing from data/raw/ohlcv/. "
            "Run scripts/scrape_top20.py first."
        )

    spy_df = pl.read_parquet(spy_path)
    vix_df = pl.read_parquet(vix_path)

    _FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    successes: list[tuple[str, int]] = []
    failures:  list[tuple[str, str]] = []

    for ticker in _STOCK_TICKERS:
        raw_path = _RAW_DIR / f"{ticker}.parquet"
        if not raw_path.exists():
            logger.warning("Skipping %s — raw parquet not found", ticker)
            failures.append((ticker, "raw parquet not found"))
            continue
        try:
            df = build_features_for_ticker(ticker, _RAW_DIR, spy_df, vix_df)
            out_path = _FEATURE_DIR / f"{ticker}.parquet"
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
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_build_features.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_features.py tests/unit/test_build_features.py
git commit -m "feat: build_features script — OHLCV parquet → feature parquet with technical + cross-asset indicators"
```

---

### Task 3: `scripts/train_models.py` — Train & Register Three Classifiers

**Files:**
- Create: `scripts/train_models.py`
- Create: `tests/unit/test_train_models.py`

**Interfaces:**
- Consumes:
  - `src.features.duckdb_client.load_training_data(parquet_dir) -> pl.DataFrame`
  - `dashboard.config.FEATURE_COLS`, `dashboard.config.REGISTRY_DIR`
  - `src.models.zoo.random_forest.RandomForestClassifier_`
  - `src.models.zoo.xgboost_model.XGBoostClassifier`
  - `src.models.zoo.lightgbm_model.LightGBMClassifier`
  - `src.models.evaluator.evaluate(y_true, y_pred) -> EvaluationResult`
  - `src.models.registry.save_model(model, evaluation, params, feature_cols, registry_dir) -> Path`
- Produces:
  - `train_and_save(model, df, feature_cols, registry_dir) -> Path`
    — fits on 80% train split, evaluates on 20% test split, saves to registry, returns joblib path
  - Output files: `data/registry/<model_name>/<version>.joblib` + `<version>.json`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_train_models.py
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _make_feature_df(n: int = 500) -> pl.DataFrame:
    from dashboard.config import FEATURE_COLS
    rng = np.random.default_rng(0)
    data: dict = {col: rng.random(n).tolist() for col in FEATURE_COLS}
    data["time"]   = [datetime(2005, 1, 1, tzinfo=timezone.utc)] * n
    data["ticker"] = ["AAPL"] * n
    data["label"]  = rng.choice(["Buy", "Hold", "Sell"], n).tolist()
    return pl.DataFrame(data)


def test_train_and_save_creates_registry_files(tmp_path):
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = _make_feature_df()
    model = RandomForestClassifier_()
    path = train_and_save(model, df, FEATURE_COLS, tmp_path)

    assert path.exists()
    json_path = path.with_suffix(".json")
    assert json_path.exists()


def test_train_and_save_writes_valid_json(tmp_path):
    import json
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = _make_feature_df()
    model = RandomForestClassifier_()
    path = train_and_save(model, df, FEATURE_COLS, tmp_path)

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["model_name"] == "random_forest"
    assert "accuracy" in meta["evaluation"]
    assert meta["feature_cols"] == FEATURE_COLS


def test_train_and_save_raises_on_empty_df(tmp_path):
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = pl.DataFrame({col: [] for col in FEATURE_COLS + ["time", "ticker", "label"]})
    with pytest.raises(ValueError, match="No training data"):
        train_and_save(RandomForestClassifier_(), df, FEATURE_COLS, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_train_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.train_models'`

- [ ] **Step 3: Implement `scripts/train_models.py`**

```python
from pathlib import Path
import logging

import numpy as np
import polars as pl

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

_FEATURE_DIR = Path("data/features")
_TRAIN_RATIO = 0.8


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


def main() -> None:
    if not _FEATURE_DIR.exists() or not any(_FEATURE_DIR.glob("*.parquet")):
        raise FileNotFoundError(
            "No feature parquets found in data/features/. "
            "Run scripts/build_features.py first."
        )

    logger.info("Loading feature data from %s ...", _FEATURE_DIR)
    df = load_training_data(_FEATURE_DIR)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    models: list[BaseClassifier] = [
        RandomForestClassifier_(),
        XGBoostClassifier(),
        LightGBMClassifier(),
    ]

    for model in models:
        logger.info("Training %s ...", model.name)
        try:
            train_and_save(model, df, FEATURE_COLS, REGISTRY_DIR)
        except Exception as exc:
            logger.error("FAILED %s: %s", model.name, exc)

    print("\nTraining complete. Registry contents:")
    for p in sorted(REGISTRY_DIR.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_train_models.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/unit/ -v --tb=short
```
Expected: All previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_models.py tests/unit/test_train_models.py
git commit -m "feat: train_models script — fit RandomForest/XGBoost/LightGBM and save to registry"
```

---

### Task 4: End-to-End Run & Dashboard Smoke Check

**Files:** None created. Validation only.

**Goal:** Confirm the three scripts produce data the dashboard can read.

- [ ] **Step 1: Run the scrape script**

```bash
python scripts/scrape_top20.py
```
Expected: Summary table shows 22 files OK (some tickers may show fewer than 5,200 rows if IPO was after 2000 — this is correct). All 22 `.parquet` files present in `data/raw/ohlcv/`.

- [ ] **Step 2: Verify raw parquet files exist**

```bash
python -c "
from pathlib import Path
files = list(Path('data/raw/ohlcv').glob('*.parquet'))
print(f'{len(files)} parquet files found:')
for f in sorted(files): print(' ', f.name)
"
```
Expected: 22 files listed.

- [ ] **Step 3: Run the feature engineering script**

```bash
python scripts/build_features.py
```
Expected: Summary shows 20 tickers OK. `data/features/` contains 20 `.parquet` files.

- [ ] **Step 4: Spot-check a feature file**

```bash
python -c "
import polars as pl
df = pl.read_parquet('data/features/AAPL.parquet')
print(df.shape)
print(df.columns)
print(df.head(3))
"
```
Expected: Shape roughly `(~4900, 27+)`. Columns include `sma_20`, `rsi_14`, `rel_strength_spy`, `vix_level`, `sent_pos_avg_5d`, `label`. No nulls in `label`.

- [ ] **Step 5: Run the training script**

```bash
python scripts/train_models.py
```
Expected: Three models trained and saved. Output lists `.json` metadata paths under `data/registry/`.

- [ ] **Step 6: Verify registry contents**

```bash
python -c "
from pathlib import Path, src.models.registry import list_models
records = list_models(Path('data/registry'))
for r in records:
    print(r.model_name, r.version[:15], f'acc={r.evaluation.accuracy:.3f}')
"
```

Or equivalently:
```bash
python -c "
from pathlib import Path
from src.models.registry import list_models
records = list_models(Path('data/registry'))
for r in records:
    print(r.model_name, r.version[:15], f'acc={r.evaluation.accuracy:.3f}')
"
```
Expected: Three records printed (random_forest, xgboost, lightgbm).

- [ ] **Step 7: Launch the dashboard**

```bash
streamlit run dashboard/app.py
```
Expected (in browser):
- **Page 1 — Data Overview:** Shows 20 tickers, date range starting ~2000, AAPL price chart with SMA 20.
- **Page 2 — Model Leaderboard:** Shows three models with A–D grades.
- **Page 3 — Backtest Results:** Walk-forward results renderable.
- **Page 4 — Live Signals:** Signals generated from top-graded model.

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "chore: end-to-end smoke validation complete — 20 stocks scraped, features built, 3 models trained"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| Scrape top 20 by market cap + SPY + ^VIX | Task 1 |
| yfinance, no API key, 2000–2020 | Task 1 |
| `data/raw/ohlcv/<TICKER>.parquet` output | Task 1 |
| Partial history for post-2000 IPOs handled | Task 1 (`scrape_ticker` catches exceptions, returns rows from IPO date) |
| Technical indicators | Task 2 (`add_technical_indicators`) |
| Cross-asset features (SPY, VIX) | Task 2 (`add_cross_asset_features`) |
| Neutral sentiment defaults | Task 2 (`add_neutral_sentiment`) |
| Labels (`Buy`/`Hold`/`Sell`) | Task 2 (`add_labels`) |
| `data/features/<TICKER>.parquet` schema matches `FEATURE_COLS` | Task 2 (test asserts every col) |
| Train RandomForest, XGBoost, LightGBM | Task 3 |
| 80/20 time-ordered split for evaluation | Task 3 |
| Save to `data/registry/` via `save_model()` | Task 3 |
| Dashboard shows all 4 pages with real data | Task 4 |

### Placeholder scan
No TBDs, no "similar to" references. All code blocks are complete.

### Type consistency
- `scrape_ticker` returns `tuple[int, str | None]` — used consistently in tests and `main()`.
- `build_features_for_ticker` accepts `pl.DataFrame` for spy/vix — consistent with `add_cross_asset_features(stock_df, spy_df, vix_df)`.
- `train_and_save` accepts `BaseClassifier` — `RandomForestClassifier_`, `XGBoostClassifier`, `LightGBMClassifier` all extend `BaseClassifier`.
- `save_model` signature: `(model, evaluation, params, feature_cols, registry_dir)` — matches `registry.py:26`.
