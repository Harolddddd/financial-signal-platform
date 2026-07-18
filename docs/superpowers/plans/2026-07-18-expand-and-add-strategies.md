# Expand Data + Add 5 Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh feature dataset to 2026-07-18, add 5 new trading strategies (CatBoost, HistGradientBoosting, SuperTrend, ADXTrend, MoneyFlowIndex), retrain all 43 strategies, and update the dashboard cache.

**Architecture:** Three sequential stages — (1) incremental OHLCV fetch + feature rebuild, (2) implement and register 5 new strategies, (3) run walk-forward backtest for new strategies only then regenerate leaderboard + signals. The skip-if-exists guard in `scripts/precompute_new_strategies.py` preserves the 38 existing backtest JSONs.

**Tech Stack:** Python 3.14, yfinance, polars, pandas, catboost, scikit-learn 1.9+, streamlit

## Global Constraints

- PYTHONPATH must be `c:/Users/h1810/.vscode/EXP` for all script runs
- All strategies inherit `src.strategies.base.Strategy`, implement `predict()` returning `PredictionResult(confidence=pd.Series, signal=pd.Series)`
- ML strategies: `data_source = "features"`, `handles_nan = True`; no `fillna()` needed (native NaN support)
- Rule-based strategies: `data_source = "ohlcv"`, stateless (no `fit()` override), `confidence` in `[0.0, 1.0]`
- `_META = {"time", "ticker", "label", "forward_return_5d"}` — excluded from ML feature columns
- Buy threshold: `>= 0.6` on `predict_proba` for ML; natural `[0,1]` for rule-based
- 6 unit tests per strategy, matching the pattern in `tests/unit/test_vwap_cross.py` (rule-based) or `tests/unit/test_xgboost_strategy.py` (ML)
- Walk-forward: 400-day train, 21-day test, 21-day step
- `confidence` must be 0.0 on Hold rows (rule-based); Buy signals must have `confidence > 0`
- Skip-if-exists guard already in `scripts/precompute_new_strategies.py` line 71–73 — do NOT remove it

---

## File Map

| File | Action |
|---|---|
| `scripts/refresh_data.py` | Create — incremental OHLCV fetch + feature rebuild |
| `src/strategies/statistical/catboost_strategy.py` | Create |
| `tests/unit/test_catboost_strategy.py` | Create |
| `src/strategies/statistical/hist_gradient_boosting.py` | Create |
| `tests/unit/test_hist_gradient_boosting.py` | Create |
| `src/strategies/rule_based/supertrend.py` | Create |
| `tests/unit/test_supertrend.py` | Create |
| `src/strategies/rule_based/adx_trend.py` | Create |
| `tests/unit/test_adx_trend.py` | Create |
| `src/strategies/rule_based/money_flow_index.py` | Create |
| `tests/unit/test_money_flow_index.py` | Create |
| `src/strategies/strategies.yaml` | Modify — append 5 entries |
| `scripts/precompute_new_strategies.py` | Modify — add 5 names to `_NEW_STRATEGIES` |

---

### Task 1: Incremental data refresh script

**Files:**
- Create: `scripts/refresh_data.py`

**Interfaces:**
- Consumes: `src.ingestion.historical_collector.fetch_ohlcv(ticker, start, end)` → `pl.DataFrame`
- Consumes: `scripts.build_features._STOCK_TICKERS` (list of 151 tickers) and `build_features_for_ticker(ticker, raw_dir, spy_df, vix_df)` → `pl.DataFrame`
- Produces: updated `data/raw/ohlcv/*.parquet` and `data/features/*.parquet` with rows up to today

- [ ] **Step 1: Write the script**

```python
# scripts/refresh_data.py
"""Incrementally refresh OHLCV data for all 151 tickers to today, then rebuild features."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from scripts.build_features import _STOCK_TICKERS, build_features_for_ticker
from src.ingestion.historical_collector import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_DIR = Path("data/raw/ohlcv")
_FEATURE_DIR = Path("data/features")
_AUX_TICKERS = ["SPY", "^VIX"]


def _refresh_raw(ticker: str, today: datetime) -> None:
    raw_path = _RAW_DIR / f"{ticker}.parquet"
    if not raw_path.exists():
        logger.warning("  %s — no raw file, skipping", ticker)
        return
    existing = pl.read_parquet(raw_path)
    max_ts = existing["time"].max()
    fetch_start = max_ts.replace(tzinfo=timezone.utc) + timedelta(days=1)
    if fetch_start.date() >= today.date():
        logger.info("  %s already current (%s)", ticker, max_ts.date())
        return
    try:
        new_rows = fetch_ohlcv(ticker, fetch_start, today)
        if len(new_rows) == 0:
            logger.info("  %s — no new rows (market closed?)", ticker)
            return
        combined = (
            pl.concat([existing, new_rows])
            .unique(subset=["time"], keep="last")
            .sort("time")
        )
        combined.write_parquet(raw_path)
        logger.info("  %s +%d rows → %d total", ticker, len(new_rows), len(combined))
    except Exception as exc:
        logger.warning("  %s fetch failed: %s", ticker, exc)


def main() -> None:
    today = datetime.now(timezone.utc)
    logger.info("=== Data refresh → %s ===", today.date())

    logger.info("[1/2] Refreshing raw OHLCV (%d stock tickers + aux)", len(_STOCK_TICKERS))
    for ticker in list(_STOCK_TICKERS) + _AUX_TICKERS:
        _refresh_raw(ticker, today)

    spy_df = pl.read_parquet(_RAW_DIR / "SPY.parquet")
    vix_df = pl.read_parquet(_RAW_DIR / "^VIX.parquet")

    logger.info("[2/2] Rebuilding features for %d tickers", len(_STOCK_TICKERS))
    _FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker in _STOCK_TICKERS:
        if not (_RAW_DIR / f"{ticker}.parquet").exists():
            continue
        try:
            df = build_features_for_ticker(ticker, _RAW_DIR, spy_df, vix_df)
            df.write_parquet(_FEATURE_DIR / f"{ticker}.parquet")
            ok += 1
        except Exception as exc:
            logger.warning("  features FAIL %s: %s", ticker, exc)
            fail += 1

    logger.info("=== Done. features OK=%d fail=%d ===", ok, fail)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python scripts/refresh_data.py 2>&1 | Tee-Object -FilePath refresh_data.log
```

Expected: lines like `AAPL +8 rows → 11490 total`, followed by `features OK=151 fail=0`. Verify with:

```
python -c "import polars as pl; df = pl.read_parquet('data/features/AAPL.parquet'); print('latest:', df['time'].max())"
```

Expected output: `latest: 2026-07-17 00:00:00 UTC` (or 2026-07-18 if market data available).

- [ ] **Step 3: Commit**

```bash
git add scripts/refresh_data.py data/raw/ohlcv/ data/features/
git commit -m "data: refresh OHLCV + features to 2026-07-18, 151 tickers"
```

---

### Task 2: CatBoostStrategy + test

**Files:**
- Create: `src/strategies/statistical/catboost_strategy.py`
- Test: `tests/unit/test_catboost_strategy.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `CatBoostStrategy` class with `data_source = "features"`, `handles_nan = True`

- [ ] **Step 1: Install catboost if needed**

```
python -c "import catboost; print(catboost.__version__)"
```

If ImportError: `pip install catboost`

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_catboost_strategy.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.catboost_strategy import CatBoostStrategy

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
    s = CatBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = CatBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = CatBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = CatBoostStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert CatBoostStrategy.data_source == "features"


def test_handles_nan_is_true():
    assert CatBoostStrategy.handles_nan is True
```

- [ ] **Step 3: Run tests — expect ImportError or AttributeError (red)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_catboost_strategy.py -v
```

Expected: ERRORS (module not found).

- [ ] **Step 4: Write the implementation**

```python
# src/strategies/statistical/catboost_strategy.py
from __future__ import annotations
import pandas as pd
from catboost import CatBoostClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class CatBoostStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, iterations: int = 200, depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        )
        self._classes: list[str] = []
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)
        self._classes = [str(c) for c in self._model.classes_]

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        if "Buy" not in self._classes:
            n = len(df)
            return PredictionResult(
                confidence=pd.Series([0.0] * n),
                signal=pd.Series(["Hold"] * n),
            )
        buy_idx = self._classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 5: Run tests — expect all 6 pass (green)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_catboost_strategy.py -v
```

Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/strategies/statistical/catboost_strategy.py tests/unit/test_catboost_strategy.py
git commit -m "feat: add CatBoostStrategy"
```

---

### Task 3: HistGradientBoostingStrategy + test

**Files:**
- Create: `src/strategies/statistical/hist_gradient_boosting.py`
- Test: `tests/unit/test_hist_gradient_boosting.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`; `LabelEncoder` from sklearn
- Produces: `HistGradientBoostingStrategy` class with `data_source = "features"`, `handles_nan = True`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_hist_gradient_boosting.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.hist_gradient_boosting import HistGradientBoostingStrategy

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
    s = HistGradientBoostingStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = HistGradientBoostingStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = HistGradientBoostingStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = HistGradientBoostingStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert HistGradientBoostingStrategy.data_source == "features"


def test_handles_nan_is_true():
    assert HistGradientBoostingStrategy.handles_nan is True
```

- [ ] **Step 2: Run tests — expect ERRORS (red)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_hist_gradient_boosting.py -v
```

Expected: ERRORS (module not found).

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/statistical/hist_gradient_boosting.py
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class HistGradientBoostingStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, max_iter: int = 200, max_depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
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
            return PredictionResult(
                confidence=pd.Series([0.0] * n),
                signal=pd.Series(["Hold"] * n),
            )
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run tests — expect 6 passed (green)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_hist_gradient_boosting.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/hist_gradient_boosting.py tests/unit/test_hist_gradient_boosting.py
git commit -m "feat: add HistGradientBoostingStrategy"
```

---

### Task 4: SuperTrend + test

**Files:**
- Create: `src/strategies/rule_based/supertrend.py`
- Test: `tests/unit/test_supertrend.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `SuperTrend` class with `data_source = "ohlcv"`, stateless

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_supertrend.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.supertrend import SuperTrend


@pytest.fixture
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    spread = np.abs(rng.normal(0, 0.5, n)) + 0.5
    high = close + spread
    low = close - spread
    open_ = low + rng.random(n) * (high - low)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def test_predict_correct_length(ohlcv_df):
    s = SuperTrend()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = SuperTrend()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = SuperTrend()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert SuperTrend.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = SuperTrend()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = SuperTrend()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run tests — expect ERRORS (red)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_supertrend.py -v
```

Expected: ERRORS (module not found).

- [ ] **Step 3: Write the implementation**

SuperTrend requires an iterative loop because each bar's value depends on the previous bar's value.

```python
# src/strategies/rule_based/supertrend.py
from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class SuperTrend(Strategy):
    """Buy when close is above the ATR-based SuperTrend line (uptrend confirmation)."""
    data_source = "ohlcv"

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0) -> None:
        self.atr_period = atr_period
        self.multiplier = multiplier

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        hl2 = (high + low) / 2
        upper_band = (hl2 + self.multiplier * atr).to_numpy()
        lower_band = (hl2 - self.multiplier * atr).to_numpy()
        close_arr = close.to_numpy()

        # Iterative: each bar depends on the previous supertrend value
        n = len(df)
        st_arr = lower_band.copy()
        for i in range(1, n):
            prev = st_arr[i - 1]
            if np.isnan(prev) or np.isnan(lower_band[i]) or np.isnan(upper_band[i]):
                st_arr[i] = lower_band[i]
                continue
            if close_arr[i - 1] > prev:
                st_arr[i] = max(lower_band[i], prev)
            else:
                st_arr[i] = min(upper_band[i], prev)

        st_series = pd.Series(st_arr, index=df.index)
        buy = close > st_series
        gap = ((close - st_series) / close.replace(0.0, float("nan"))).clip(0.0, 0.05) / 0.05
        confidence = gap.where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
```

- [ ] **Step 4: Run tests — expect 6 passed (green)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_supertrend.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/supertrend.py tests/unit/test_supertrend.py
git commit -m "feat: add SuperTrend strategy"
```

---

### Task 5: ADX Trend Filter + test

**Files:**
- Create: `src/strategies/rule_based/adx_trend.py`
- Test: `tests/unit/test_adx_trend.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `ADXTrend` class with `data_source = "ohlcv"`, stateless

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_adx_trend.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.adx_trend import ADXTrend


@pytest.fixture
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    spread = np.abs(rng.normal(0, 0.5, n)) + 0.5
    high = close + spread
    low = close - spread
    open_ = low + rng.random(n) * (high - low)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def test_predict_correct_length(ohlcv_df):
    s = ADXTrend()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = ADXTrend()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = ADXTrend()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert ADXTrend.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = ADXTrend()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = ADXTrend()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run tests — expect ERRORS (red)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_adx_trend.py -v
```

Expected: ERRORS (module not found).

- [ ] **Step 3: Write the implementation**

Wilder's EWM uses `alpha = 1/period`, equivalent to `ewm(span=2*period-1, adjust=False)`.

```python
# src/strategies/rule_based/adx_trend.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ADXTrend(Strategy):
    """Buy when ADX > threshold (strong trend) and +DI > -DI (bullish direction)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 14, adx_threshold: float = 25.0) -> None:
        self.period = period
        self.adx_threshold = adx_threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        span = 2 * self.period - 1
        tr_smooth = tr.ewm(span=span, adjust=False).mean()
        plus_dm_s = plus_dm.ewm(span=span, adjust=False).mean()
        minus_dm_s = minus_dm.ewm(span=span, adjust=False).mean()

        safe_tr = tr_smooth.replace(0.0, float("nan"))
        plus_di = 100.0 * plus_dm_s / safe_tr
        minus_di = 100.0 * minus_dm_s / safe_tr

        di_sum = (plus_di + minus_di).replace(0.0, float("nan"))
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(span=span, adjust=False).mean()

        buy = (adx > self.adx_threshold) & (plus_di > minus_di)
        conf_raw = (adx / 100.0).clip(0.0, 1.0) * ((plus_di - minus_di) / 100.0).clip(0.0, 1.0)
        confidence = conf_raw.where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
```

- [ ] **Step 4: Run tests — expect 6 passed (green)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_adx_trend.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/adx_trend.py tests/unit/test_adx_trend.py
git commit -m "feat: add ADXTrend strategy"
```

---

### Task 6: Money Flow Index + test

**Files:**
- Create: `src/strategies/rule_based/money_flow_index.py`
- Test: `tests/unit/test_money_flow_index.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `MoneyFlowIndex` class with `data_source = "ohlcv"`, stateless

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_money_flow_index.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.money_flow_index import MoneyFlowIndex


@pytest.fixture
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    spread = np.abs(rng.normal(0, 0.5, n)) + 0.5
    high = close + spread
    low = close - spread
    open_ = low + rng.random(n) * (high - low)
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def test_predict_correct_length(ohlcv_df):
    s = MoneyFlowIndex()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = MoneyFlowIndex()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = MoneyFlowIndex()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert MoneyFlowIndex.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = MoneyFlowIndex()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = MoneyFlowIndex()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run tests — expect ERRORS (red)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_money_flow_index.py -v
```

Expected: ERRORS (module not found).

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/rule_based/money_flow_index.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MoneyFlowIndex(Strategy):
    """Buy when MFI is below the oversold threshold (volume-weighted RSI recovery signal)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 14, oversold: float = 30.0) -> None:
        self.period = period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        prev_typical = typical.shift(1)
        raw_flow = typical * df["volume"]

        positive_flow = raw_flow.where(typical > prev_typical, 0.0)
        negative_flow = raw_flow.where(typical < prev_typical, 0.0)

        pos_sum = positive_flow.rolling(self.period).sum()
        neg_sum = negative_flow.rolling(self.period).sum().replace(0.0, float("nan"))

        mfr = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + mfr))

        buy = mfi < self.oversold
        confidence = ((self.oversold - mfi) / self.oversold).clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
```

- [ ] **Step 4: Run tests — expect 6 passed (green)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_money_flow_index.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/money_flow_index.py tests/unit/test_money_flow_index.py
git commit -m "feat: add MoneyFlowIndex strategy"
```

---

### Task 7: Register 5 new strategies in YAML

**Files:**
- Modify: `src/strategies/strategies.yaml`

**Interfaces:**
- Consumes: class paths from Tasks 2–6
- Produces: 43 strategies total registered and loadable via `load_strategy(name)`

- [ ] **Step 1: Append entries to `src/strategies/strategies.yaml`**

Add these 5 entries at the end of the `strategies:` list (after the `vwap_cross` entry on line 239):

```yaml
  - name: catboost_strategy
    class: src.strategies.statistical.catboost_strategy.CatBoostStrategy
    params:
      iterations: 200
      depth: 6
      learning_rate: 0.1

  - name: hist_gradient_boosting
    class: src.strategies.statistical.hist_gradient_boosting.HistGradientBoostingStrategy
    params:
      max_iter: 200
      max_depth: 6
      learning_rate: 0.1

  - name: supertrend
    class: src.strategies.rule_based.supertrend.SuperTrend
    params:
      atr_period: 10
      multiplier: 3.0

  - name: adx_trend
    class: src.strategies.rule_based.adx_trend.ADXTrend
    params:
      period: 14
      adx_threshold: 25.0

  - name: money_flow_index
    class: src.strategies.rule_based.money_flow_index.MoneyFlowIndex
    params:
      period: 14
      oversold: 30.0
```

- [ ] **Step 2: Verify 43 strategies load without error**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python -c "
from src.strategies.registry import list_strategies, load_strategy
names = list_strategies()
print(f'Total: {len(names)}')
for n in names:
    load_strategy(n)
    print(f'  OK: {n}')
"
```

Expected: `Total: 43` and all 43 `OK:` lines with no exceptions.

- [ ] **Step 3: Commit**

```bash
git add src/strategies/strategies.yaml
git commit -m "feat: register 5 new strategies — 43 total in registry"
```

---

### Task 8: Update precompute script + run backtests + commit cache

**Files:**
- Modify: `scripts/precompute_new_strategies.py` — add 5 names to `_NEW_STRATEGIES`

**Interfaces:**
- Consumes: all 5 strategy classes registered in Task 7, refreshed feature parquets from Task 1
- Produces: 5 new `data/cache/backtest_*.json` files, updated `leaderboard.json`, updated `signals.json`

- [ ] **Step 1: Add 5 names to `_NEW_STRATEGIES` in `scripts/precompute_new_strategies.py`**

The current list (line 28–37) is:
```python
_NEW_STRATEGIES = [
    "svm_strategy",
    "mlp_classifier",
    "adaboost",
    "knn_classifier",
    "ichimoku_cloud",
    "chaikin_money_flow",
    "aroon_oscillator",
    "vwap_cross",
]
```

Replace with:
```python
_NEW_STRATEGIES = [
    "svm_strategy",
    "mlp_classifier",
    "adaboost",
    "knn_classifier",
    "ichimoku_cloud",
    "chaikin_money_flow",
    "aroon_oscillator",
    "vwap_cross",
    "catboost_strategy",
    "hist_gradient_boosting",
    "supertrend",
    "adx_trend",
    "money_flow_index",
]
```

The skip-if-exists guard at lines 71–73 ensures the 8 already-computed strategies are skipped. Only the 5 new ones run.

- [ ] **Step 2: Run the precompute (background — takes 30–120 min)**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python scripts/precompute_new_strategies.py 2>&1 | Tee-Object -FilePath precompute_new.log
```

Monitor: watch `precompute_new.log` for lines like `wrote data\cache\backtest_catboost_strategy.json`. After all 5 complete, `step_leaderboard()` and `step_signals()` run automatically and the script prints `=== Done. ===`.

- [ ] **Step 3: Verify all 43 backtest JSONs exist**

```
python -c "
from pathlib import Path
jsons = list(Path('data/cache').glob('backtest_*.json'))
print(f'Backtest JSONs: {len(jsons)}')
for j in sorted(jsons): print(' ', j.name)
"
```

Expected: `Backtest JSONs: 43` (38 previous + 5 new).

- [ ] **Step 4: Commit cache**

```bash
git add data/cache/ scripts/precompute_new_strategies.py
git commit -m "data: precompute cache — 43 strategies, 151 tickers, refreshed to 2026-07-18"
```

---

## Run all new strategy tests together (sanity check after Task 6)

After Tasks 2–6, run the full new-strategy suite to confirm nothing regressed:

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_catboost_strategy.py tests/unit/test_hist_gradient_boosting.py tests/unit/test_supertrend.py tests/unit/test_adx_trend.py tests/unit/test_money_flow_index.py -v
```

Expected: `30 passed`.
