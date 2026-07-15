# Expand Universe + New ML Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the stock universe from 92 to ~147 tickers (adding 55 tickers including ETFs), extend historical data back to 1980, add 4 new ML strategies (RandomForest, XGBoost, LightGBM, ExtraTrees), retrain all 30 strategies via walk-forward backtesting, and update the dashboard cache.

**Architecture:** All data is fetched via yfinance, stored as per-ticker parquet files in `data/raw/ohlcv/`, then feature-engineered into `data/features/`. New strategies follow the existing `Strategy` base class pattern (fit + predict), registered in `strategies.yaml`, and picked up automatically by `precompute_dashboard.py`. Tasks 4–7 (strategy implementations) can run in parallel with Task 2 (data fetch) since they are independent.

**Tech Stack:** Python 3.14, Polars, yfinance, scikit-learn 1.9, XGBoost 3.3, LightGBM 4.6, DuckDB

## Global Constraints

- PYTHONPATH must be set: `PYTHONPATH="c:/Users/h1810/.vscode/EXP"` for all script runs
- All ML strategy files must follow the pattern in `src/strategies/statistical/gradient_boosting.py`
- `data_source = "features"` for all 4 new ML strategies (uses the 22 precomputed FEATURE_COLS)
- `handles_nan = True` for XGBoost and LightGBM (native NaN support); omit for RF and ExtraTrees (runner calls dropna before fit; strategies use fillna(0.0) in predict)
- Buy signal confidence threshold: `>= 0.6` in all predict() implementations
- The 22 feature columns are: `sma_10, sma_20, sma_50, sma_200, ema_12, ema_26, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_lower, bb_width, atr_14, hist_vol_21, sent_pos_avg_3d, sent_pos_avg_5d, sent_pos_avg_10d, sent_pos_mom_3d, news_vol_spike, rel_strength_spy, vix_level`
- `_META = {"time", "ticker", "label", "forward_return_5d"}` — exclude from feature columns in fit/predict

---

### Task 1: Update data collection scripts

**Files:**
- Modify: `scripts/scrape_top20.py`
- Modify: `scripts/build_features.py`

**Interfaces:**
- Produces: 55 new tickers added to TICKERS in scrape_top20.py; same tickers added to _STOCK_TICKERS in build_features.py; START date changed from 1990 → 1980

- [ ] **Step 1: Extend start date and add 55 new tickers to scrape_top20.py**

In `scripts/scrape_top20.py`, change the `START` constant:
```python
START = datetime(1980, 1, 1, tzinfo=timezone.utc)
```

Append these entries to the `TICKERS` list (after the existing 92 entries):
```python
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
```

- [ ] **Step 2: Add same 55 tickers to build_features.py**

In `scripts/build_features.py`, append the following to `_STOCK_TICKERS` after the existing entries:

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add scripts/scrape_top20.py scripts/build_features.py
git commit -m "feat: expand universe to 147 tickers, extend history to 1980"
```

---

### Task 2: Fetch data for all 147 tickers

**Files:**
- Produces: `data/raw/ohlcv/<TICKER>.parquet` for all 147 tickers + SPY + ^VIX (149 files total, refreshed from 1980)

- [ ] **Step 1: Run data fetch (background — ~20 minutes)**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python scripts/scrape_top20.py > data/raw/scrape.log 2>&1
```

Monitor progress:
```bash
tail -f data/raw/scrape.log
```

Expected final output: `Done. 149 succeeded, 0 failed.`

- [ ] **Step 2: Verify new parquet files exist**

```bash
python -c "
import os
files = os.listdir('data/raw/ohlcv')
print(f'Total files: {len(files)}')
for t in ['QQQ', 'GLD', 'TLT', 'ABBV', 'CSCO', 'BA']:
    exists = f'{t}.parquet' in files
    print(f'  {t}: {\"OK\" if exists else \"MISSING\"}')
"
# Expected: Total files: 149; all spot-checks OK
```

---

### Task 3: Rebuild features for all tickers

**Files:**
- Produces: `data/features/<TICKER>.parquet` for all 147 tickers (147 files with 22 feature columns + labels)

**Interfaces:**
- Consumes: `data/raw/ohlcv/<TICKER>.parquet` (from Task 2), `data/raw/ohlcv/SPY.parquet`, `data/raw/ohlcv/^VIX.parquet`

- [ ] **Step 1: Run feature build (~3 minutes)**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python scripts/build_features.py 2>&1 | tee data/features/build.log
```

Expected: `Done. 147 succeeded, 0 failed.`

- [ ] **Step 2: Verify features updated**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import duckdb
conn = duckdb.connect()
r = conn.execute(\"SELECT COUNT(DISTINCT ticker) as n_tickers, COUNT(*) as n_rows, CAST(MIN(time) AS VARCHAR) as earliest, CAST(MAX(time) AS VARCHAR) as latest FROM read_parquet('data/features/*.parquet')\").fetchone()
print(f'Tickers: {r[0]}')
print(f'Rows: {r[1]:,}')
print(f'Date range: {r[2][:10]} to {r[3][:10]}')
"
# Expected: Tickers: 147, Rows: ~1,200,000+, earliest ~1980-01-xx
```

---

### Task 4: Implement RandomForest strategy

**Files:**
- Create: `src/strategies/statistical/random_forest.py`
- Create: `tests/unit/test_random_forest_strategy.py`

**Interfaces:**
- Produces: `RandomForestStrategy` — `data_source="features"`, no `handles_nan` (defaults False), `fit(df: pd.DataFrame)`, `predict(df: pd.DataFrame) -> PredictionResult`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_random_forest_strategy.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.random_forest import RandomForestStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = RandomForestStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = RandomForestStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = RandomForestStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = RandomForestStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert RandomForestStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(RandomForestStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_random_forest_strategy.py -v 2>&1 | tail -5
# Expected: ERROR — ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement the strategy**

Create `src/strategies/statistical/random_forest.py`:

```python
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class RandomForestStrategy(Strategy):
    data_source = "features"

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_random_forest_strategy.py -v 2>&1 | tail -10
# Expected: 6 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/random_forest.py tests/unit/test_random_forest_strategy.py
git commit -m "feat: add RandomForest strategy"
```

---

### Task 5: Implement XGBoost strategy

**Files:**
- Create: `src/strategies/statistical/xgboost_strategy.py`
- Create: `tests/unit/test_xgboost_strategy.py`

**Interfaces:**
- Produces: `XGBoostStrategy` — `data_source="features"`, `handles_nan=True`, uses `LabelEncoder` internally so XGBoost receives integer labels; `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_xgboost_strategy.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.xgboost_strategy import XGBoostStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = XGBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = XGBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = XGBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = XGBoostStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert XGBoostStrategy.data_source == "features"


def test_handles_nan_is_true():
    assert XGBoostStrategy.handles_nan is True
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_xgboost_strategy.py -v 2>&1 | tail -5
# Expected: ERROR — ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement the strategy**

Create `src/strategies/statistical/xgboost_strategy.py`:

```python
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class XGBoostStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42) -> None:
        self._model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, tree_method="hist",
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        self._le = LabelEncoder()
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = self._le.fit_transform(df["label"].to_numpy())
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._le.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_xgboost_strategy.py -v 2>&1 | tail -10
# Expected: 6 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/xgboost_strategy.py tests/unit/test_xgboost_strategy.py
git commit -m "feat: add XGBoost strategy"
```

---

### Task 6: Implement LightGBM strategy

**Files:**
- Create: `src/strategies/statistical/lightgbm_strategy.py`
- Create: `tests/unit/test_lightgbm_strategy.py`

**Interfaces:**
- Produces: `LightGBMStrategy` — `data_source="features"`, `handles_nan=True`, passes string labels directly to LGBM; `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_lightgbm_strategy.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.lightgbm_strategy import LightGBMStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = LightGBMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = LightGBMStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert LightGBMStrategy.data_source == "features"


def test_handles_nan_is_true():
    assert LightGBMStrategy.handles_nan is True
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_lightgbm_strategy.py -v 2>&1 | tail -5
# Expected: ERROR — ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement the strategy**

Create `src/strategies/statistical/lightgbm_strategy.py`:

```python
from __future__ import annotations
import pandas as pd
from lightgbm import LGBMClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class LightGBMStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42) -> None:
        self._model = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_lightgbm_strategy.py -v 2>&1 | tail -10
# Expected: 6 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/lightgbm_strategy.py tests/unit/test_lightgbm_strategy.py
git commit -m "feat: add LightGBM strategy"
```

---

### Task 7: Implement ExtraTrees strategy

**Files:**
- Create: `src/strategies/statistical/extra_trees.py`
- Create: `tests/unit/test_extra_trees_strategy.py`

**Interfaces:**
- Produces: `ExtraTreesStrategy` — `data_source="features"`, no `handles_nan` (defaults False), uses fillna(0.0) in fit/predict; `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_extra_trees_strategy.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.extra_trees import ExtraTreesStrategy

_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["time"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["ticker"] = "TEST"
    data["label"] = rng.choice(["Buy", "Hold", "Sell"], n)
    data["forward_return_5d"] = rng.standard_normal(n) * 0.01
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = ExtraTreesStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = ExtraTreesStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = ExtraTreesStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = ExtraTreesStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert ExtraTreesStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(ExtraTreesStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_extra_trees_strategy.py -v 2>&1 | tail -5
# Expected: ERROR — ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement the strategy**

Create `src/strategies/statistical/extra_trees.py`:

```python
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class ExtraTreesStrategy(Strategy):
    data_source = "features"

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42) -> None:
        self._model = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -m pytest tests/unit/test_extra_trees_strategy.py -v 2>&1 | tail -10
# Expected: 6 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/extra_trees.py tests/unit/test_extra_trees_strategy.py
git commit -m "feat: add ExtraTrees strategy"
```

---

### Task 8: Register all 4 new strategies in strategies.yaml

**Files:**
- Modify: `src/strategies/strategies.yaml`

**Interfaces:**
- Consumes: `RandomForestStrategy`, `XGBoostStrategy`, `LightGBMStrategy`, `ExtraTreesStrategy` (Tasks 4–7)
- Produces: `strategies.yaml` with 30 total entries; `list_strategies()` returns 30 names

- [ ] **Step 1: Append 4 entries to strategies.yaml**

Add the following to the end of `src/strategies/strategies.yaml`:

```yaml
  - name: random_forest
    class: src.strategies.statistical.random_forest.RandomForestStrategy
    params:
      n_estimators: 100
      max_depth: 10

  - name: xgboost_strategy
    class: src.strategies.statistical.xgboost_strategy.XGBoostStrategy
    params:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1

  - name: lightgbm_strategy
    class: src.strategies.statistical.lightgbm_strategy.LightGBMStrategy
    params:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1

  - name: extra_trees
    class: src.strategies.statistical.extra_trees.ExtraTreesStrategy
    params:
      n_estimators: 100
      max_depth: 10
```

- [ ] **Step 2: Verify all 30 strategies load correctly**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
from src.strategies.registry import list_strategies, load_strategy
names = list_strategies()
print(f'Total: {len(names)}')
for name in ['random_forest', 'xgboost_strategy', 'lightgbm_strategy', 'extra_trees']:
    s = load_strategy(name)
    nan = getattr(s, 'handles_nan', False)
    print(f'  {name}: data_source={s.data_source} handles_nan={nan}')
"
# Expected:
# Total: 30
#   random_forest: data_source=features handles_nan=False
#   xgboost_strategy: data_source=features handles_nan=True
#   lightgbm_strategy: data_source=features handles_nan=True
#   extra_trees: data_source=features handles_nan=False
```

- [ ] **Step 3: Commit**

```bash
git add src/strategies/strategies.yaml
git commit -m "feat: register 4 new ML strategies — 30 total in registry"
```

---

### Task 9: Run precompute_dashboard and update cache

**Files:**
- Produces: `data/cache/data_summary.json`, `data/cache/backtest_<name>.json` (30 files), `data/cache/leaderboard.json`, `data/cache/signals.json`

**Interfaces:**
- Consumes: all 147 feature parquets (Task 3), all 30 strategies (Task 8)

- [ ] **Step 1: Confirm prerequisites**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import duckdb
conn = duckdb.connect()
r = conn.execute(\"SELECT COUNT(DISTINCT ticker) FROM read_parquet('data/features/*.parquet')\").fetchone()
print('Feature tickers:', r[0])
from src.strategies.registry import list_strategies
print('Strategies:', len(list_strategies()))
"
# Expected: Feature tickers: 147, Strategies: 30
```

- [ ] **Step 2: Launch precompute (background — ~60–90 minutes)**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python scripts/precompute_dashboard.py > data/cache/precompute.log 2>&1
```

Monitor progress (each strategy logs its grade when done):
```bash
tail -f data/cache/precompute.log
```

Watch for lines like:
```
INFO   strategy: random_forest
INFO     trades=XXXXX  sharpe=0.XXX  prec_buy=0.XXX  grade=X
```

- [ ] **Step 3: Verify all 30 strategies in leaderboard**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import json
d = json.load(open('data/cache/leaderboard.json'))
print('Generated:', d['generated_at'])
print('Total strategies:', len(d['grades']))
new = [g for g in d['grades'] if g['model_name'] in {'random_forest','xgboost_strategy','lightgbm_strategy','extra_trees'}]
for g in new:
    print(f\"  {g['model_name']:25s} {g['grade']}  {g['composite_score']:.3f}\")
"
# Expected: 30 total, all 4 new strategies listed with grades
```

- [ ] **Step 4: Verify signals include new strategies**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import json
d = json.load(open('data/cache/signals.json'))
strats = set(s['strategy'] for s in d['signals'])
new = {'random_forest', 'xgboost_strategy', 'lightgbm_strategy', 'extra_trees'}
print('New strategies present:', new & strats)
buy = [s for s in d['signals'] if s['signal'] == 'Buy']
print('Total Buy signals:', len(buy))
"
# Expected: all 4 new strategies present
```

- [ ] **Step 5: Verify data_summary reflects new ticker count**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import json
d = json.load(open('data/cache/data_summary.json'))
print('Tickers:', d['n_tickers'])
print('Rows:', f\"{d['n_rows']:,}\")
print('Date range:', d['date_range_start'][:10], '-', d['date_range_end'][:10])
"
# Expected: Tickers: 147, Rows ~1,200,000+, date range starts ~1980-01
```

---

### Task 10: Commit and push

- [ ] **Step 1: Stage cache files and verify no secrets**

```bash
git status
# Should see: data/cache/*.json files, no .env or credential files
git add data/cache/
git status
```

- [ ] **Step 2: Commit cache**

```bash
git commit -m "data: precompute cache — 30 strategies, 147 tickers, history from 1980"
```

- [ ] **Step 3: Push to remote**

```bash
git push
```

- [ ] **Step 4: Final summary check**

```bash
PYTHONPATH="c:/Users/h1810/.vscode/EXP" python -c "
import json
lb = json.load(open('data/cache/leaderboard.json'))
ds = json.load(open('data/cache/data_summary.json'))
sig = json.load(open('data/cache/signals.json'))
print(f'Tickers: {ds[\"n_tickers\"]}')
print(f'Rows: {ds[\"n_rows\"]:,}')
print(f'Date range: {ds[\"date_range_start\"][:10]} to {ds[\"date_range_end\"][:10]}')
print(f'Strategies: {len(lb[\"grades\"])}')
print(f'Top 5: {[(g[\"model_name\"], g[\"grade\"]) for g in lb[\"grades\"][:5]]}')
buy = sum(1 for s in sig[\"signals\"] if s[\"signal\"]==\"Buy\")
print(f'Buy signals: {buy}')
"
```
