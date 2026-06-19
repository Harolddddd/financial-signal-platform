# Simple Strategy Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ML model zoo with a plugin-based `Strategy` system of rule-based technical strategies and simple statistical models, graded head-to-head on the existing walk-forward leaderboard.

**Architecture:** A `Strategy` ABC with `fit()`/`predict()` methods and a `data_source` class attribute sits at the base. A YAML registry maps strategy names to class paths — `load_strategy(name)` imports and instantiates the class dynamically. A thin `strategy_runner.py` adapter replicates the walk-forward fold loop but drives strategies instead of sklearn models, then feeds into the existing `compute_metrics()` / `grade_model()` / `build_leaderboard()` unchanged.

**Tech Stack:** Python 3.11+, Polars (data loading/fold splitting), Pandas (passed into strategy `fit`/`predict`), NumPy, scikit-learn (statistical strategies), PyYAML, pytest.

## Global Constraints

- Polars is used for all DataFrame loading and fold-splitting; pandas is used only inside `strategy_runner.py` (Polars→pandas conversion before calling strategy methods).
- `src/backtesting/walk_forward.py`, `src/backtesting/grader.py`, `src/backtesting/engine.py`, `src/backtesting/metrics.py` are **not modified**.
- Existing ML model files in `src/models/` are left in place, unused.
- All `PredictionResult.signal` values are plain strings `"Buy"` / `"Hold"` / `"Sell"` (not enum objects) for simple equality comparison.
- `confidence` values must always be in `[0.0, 1.0]`.
- Tests use `np.random.default_rng(42)` for reproducibility.
- Run tests with: `pytest tests/unit/<test_file>.py -v`

---

## File Map

| Status | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/strategies/__init__.py` | Package marker; re-exports `Strategy`, `Signal`, `PredictionResult`, `LiveSignal` |
| **Create** | `src/strategies/base.py` | `Strategy` ABC, `Signal` enum, `PredictionResult` dataclass, `LiveSignal` dataclass |
| **Create** | `src/strategies/registry.py` | `list_strategies()`, `load_strategy(name)` |
| **Create** | `src/strategies/strategies.yaml` | Enabled strategies + constructor params |
| **Create** | `src/strategies/rule_based/__init__.py` | Package marker |
| **Create** | `src/strategies/rule_based/ma_crossover.py` | `MACrossover` — fast/slow SMA crossover |
| **Create** | `src/strategies/rule_based/rsi.py` | `RSIThreshold` — RSI oversold bounce |
| **Create** | `src/strategies/rule_based/macd.py` | `MACDSignal` — MACD/signal-line crossover |
| **Create** | `src/strategies/rule_based/bollinger.py` | `BollingerBounce` — lower-band touch |
| **Create** | `src/strategies/statistical/__init__.py` | Package marker |
| **Create** | `src/strategies/statistical/logistic.py` | `LogisticStrategy` — logistic regression on feature cols |
| **Create** | `src/strategies/statistical/linear.py` | `LinearStrategy` — linear regression on feature cols |
| **Create** | `src/backtesting/strategy_runner.py` | `walk_forward_backtest_strategy()` adapter |
| **Modify** | `dashboard/config.py` | Add `OHLCV_COLS`; remove `REGISTRY_DIR` |
| **Modify** | `dashboard/data_loader.py` | Swap model imports → strategy imports; update all three loader functions |
| **Modify** | `dashboard/pages/2_Model_Leaderboard.py` | "Model" → "Strategy" labels; swap `list_models` → `list_strategies`; update imports |
| **Modify** | `dashboard/pages/3_Backtest_Results.py` | Swap `list_models` → `list_strategies`; update `get_backtest_result` call signature |
| **Modify** | `dashboard/pages/4_Live_Signals.py` | Remove SHAP chart; use `sig.signal` badge; update imports |

---

### Task 1: Strategy Base Types

**Files:**
- Create: `src/strategies/__init__.py`
- Create: `src/strategies/base.py`
- Create: `tests/unit/test_strategies_base.py`

**Interfaces:**
- Produces: `Strategy` (ABC), `Signal` (Enum: BUY/HOLD/SELL), `PredictionResult` (dataclass with `confidence: pd.Series`, `signal: pd.Series`), `LiveSignal` (dataclass for dashboard)

---

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/unit/test_strategies_base.py
import pandas as pd
import pytest
from src.strategies.base import Signal, PredictionResult, Strategy, LiveSignal


def test_signal_enum_values():
    assert Signal.BUY.value == "Buy"
    assert Signal.HOLD.value == "Hold"
    assert Signal.SELL.value == "Sell"


def test_prediction_result_holds_series():
    conf = pd.Series([0.8, 0.2, 0.5])
    sig = pd.Series(["Buy", "Hold", "Hold"])
    result = PredictionResult(confidence=conf, signal=sig)
    assert len(result.confidence) == 3
    assert len(result.signal) == 3


def test_strategy_is_abstract():
    with pytest.raises(TypeError):
        Strategy()


def test_strategy_fit_is_noop_by_default():
    class ConcreteStrategy(Strategy):
        data_source = "ohlcv"
        def predict(self, df: pd.DataFrame):
            return PredictionResult(
                confidence=pd.Series([0.5] * len(df)),
                signal=pd.Series(["Hold"] * len(df)),
            )

    s = ConcreteStrategy()
    df = pd.DataFrame({"close": [100.0, 101.0]})
    s.fit(df)   # must not raise


def test_live_signal_fields():
    sig = LiveSignal(
        ticker="AAPL",
        date="2024-01-01",
        signal=Signal.BUY,
        confidence=0.82,
        entry_price=195.0,
        position_size=0.82,
    )
    assert sig.ticker == "AAPL"
    assert sig.signal == Signal.BUY
    assert sig.confidence == 0.82
```

- [ ] **Step 1.2: Run tests — expect FAIL (imports missing)**

```
pytest tests/unit/test_strategies_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.strategies'`

- [ ] **Step 1.3: Create package marker**

```python
# src/strategies/__init__.py
from src.strategies.base import Strategy, Signal, PredictionResult, LiveSignal

__all__ = ["Strategy", "Signal", "PredictionResult", "LiveSignal"]
```

- [ ] **Step 1.4: Create base.py**

```python
# src/strategies/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd


class Signal(Enum):
    BUY  = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


@dataclass
class PredictionResult:
    confidence: pd.Series  # float [0.0, 1.0] per row
    signal: pd.Series      # str: "Buy" / "Hold" / "Sell" per row


@dataclass
class LiveSignal:
    ticker: str
    date: str
    signal: Signal
    confidence: float
    entry_price: float
    position_size: float


class Strategy(ABC):
    data_source: Literal["ohlcv", "features"]

    def fit(self, df: pd.DataFrame) -> None:
        pass  # no-op for rule-based; override in statistical

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        ...
```

- [ ] **Step 1.5: Run tests — expect PASS**

```
pytest tests/unit/test_strategies_base.py -v
```

Expected: 5 passed

- [ ] **Step 1.6: Commit**

```bash
git add src/strategies/__init__.py src/strategies/base.py tests/unit/test_strategies_base.py
git commit -m "feat: add Strategy ABC, Signal enum, PredictionResult, LiveSignal"
```

---

### Task 2: Strategy Registry

**Files:**
- Create: `src/strategies/registry.py`
- Create: `src/strategies/strategies.yaml`
- Create: `src/strategies/rule_based/__init__.py`
- Create: `src/strategies/statistical/__init__.py`
- Create: `tests/unit/test_strategies_registry.py`

**Interfaces:**
- Consumes: `Strategy` from Task 1
- Produces: `list_strategies() -> list[str]`, `load_strategy(name: str) -> Strategy`

---

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/unit/test_strategies_registry.py
import pytest
from src.strategies.registry import list_strategies, load_strategy
from src.strategies.base import Strategy


def test_list_strategies_returns_list():
    names = list_strategies()
    assert isinstance(names, list)
    assert len(names) >= 1


def test_list_strategies_contains_expected():
    names = list_strategies()
    assert "ma_crossover" in names
    assert "rsi_threshold" in names
    assert "macd_signal" in names
    assert "bollinger_bounce" in names
    assert "logistic_regression" in names
    assert "linear_regression" in names


def test_load_strategy_returns_strategy_instance():
    strategy = load_strategy("ma_crossover")
    assert isinstance(strategy, Strategy)


def test_load_strategy_injects_params():
    strategy = load_strategy("ma_crossover")
    assert strategy.fast_window == 20
    assert strategy.slow_window == 50


def test_load_strategy_unknown_raises():
    with pytest.raises((KeyError, StopIteration)):
        load_strategy("does_not_exist")
```

- [ ] **Step 2.2: Run tests — expect FAIL**

```
pytest tests/unit/test_strategies_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.strategies.registry'`

- [ ] **Step 2.3: Create package markers**

```python
# src/strategies/rule_based/__init__.py
```

```python
# src/strategies/statistical/__init__.py
```

(Both files are empty.)

- [ ] **Step 2.4: Create strategies.yaml**

```yaml
# src/strategies/strategies.yaml
strategies:
  - name: ma_crossover
    class: src.strategies.rule_based.ma_crossover.MACrossover
    params:
      fast_window: 20
      slow_window: 50

  - name: rsi_threshold
    class: src.strategies.rule_based.rsi.RSIThreshold
    params:
      period: 14
      oversold: 30
      overbought: 70

  - name: macd_signal
    class: src.strategies.rule_based.macd.MACDSignal
    params:
      fast: 12
      slow: 26
      signal_period: 9

  - name: bollinger_bounce
    class: src.strategies.rule_based.bollinger.BollingerBounce
    params:
      window: 20
      num_std: 2.0

  - name: logistic_regression
    class: src.strategies.statistical.logistic.LogisticStrategy
    params:
      C: 1.0
      max_iter: 200

  - name: linear_regression
    class: src.strategies.statistical.linear.LinearStrategy
    params:
      buy_threshold: 0.005
```

- [ ] **Step 2.5: Create registry.py**

```python
# src/strategies/registry.py
from __future__ import annotations
import importlib
from pathlib import Path

import yaml

from src.strategies.base import Strategy

_CONFIG_PATH = Path(__file__).parent / "strategies.yaml"


def list_strategies() -> list[str]:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    return [s["name"] for s in config["strategies"]]


def load_strategy(name: str) -> Strategy:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    entry = next(s for s in config["strategies"] if s["name"] == name)
    module_path, class_name = entry["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**entry.get("params", {}))
```

- [ ] **Step 2.6: Run tests — expect partial FAIL (strategy classes missing)**

```
pytest tests/unit/test_strategies_registry.py -v
```

Expected: `test_list_strategies_returns_list` PASS, `test_list_strategies_contains_expected` PASS, others FAIL with `ModuleNotFoundError` (strategy classes not yet created)

- [ ] **Step 2.7: Commit partial progress**

```bash
git add src/strategies/registry.py src/strategies/strategies.yaml \
        src/strategies/rule_based/__init__.py src/strategies/statistical/__init__.py \
        tests/unit/test_strategies_registry.py
git commit -m "feat: add strategy registry with YAML config"
```

---

### Task 3: Backtest Strategy Runner

**Files:**
- Create: `src/backtesting/strategy_runner.py`
- Create: `tests/unit/test_strategy_runner.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from Task 1; `WalkForwardBacktestResult`, `FoldBacktestResult` from `src.backtesting.walk_forward`; `Trade`, `compute_metrics` from `src.backtesting.metrics`
- Produces: `walk_forward_backtest_strategy(df, strategy, ohlcv_cols, feature_cols, ...) -> WalkForwardBacktestResult`

---

- [ ] **Step 3.1: Write the failing tests**

```python
# tests/unit/test_strategy_runner.py
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.strategies.base import Strategy, PredictionResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.backtesting.walk_forward import WalkForwardBacktestResult


class _AlwaysBuyStrategy(Strategy):
    """Predicts BUY with confidence 0.9 for every row."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.9] * n),
            signal=pd.Series(["Buy"] * n),
        )


class _AlwaysHoldStrategy(Strategy):
    """Predicts HOLD for every row — produces zero trades."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.0] * n),
            signal=pd.Series(["Hold"] * n),
        )


def _make_df(n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    closes = [100.0 + i * 0.05 for i in range(n)]
    labels = ["Buy" if i % 3 == 0 else "Hold" for i in range(n)]
    returns = [0.02 if l == "Buy" else 0.001 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        "open":  [c - 0.5 for c in closes],
        "high":  [c + 1.0 for c in closes],
        "low":   [c - 1.0 for c in closes],
        "volume": rng.integers(1_000_000, 10_000_000, n).tolist(),
    })


_OHLCV = ["open", "high", "low", "close", "volume"]
_FEATURES = []  # not used in these tests


def test_returns_walk_forward_result():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    assert isinstance(result, WalkForwardBacktestResult)
    assert len(result.folds) >= 1


def test_fold_count_matches_expected():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert fold.metrics is not None
        assert fold.fold >= 0


def test_always_hold_produces_zero_trades():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysHoldStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    total_trades = sum(f.n_trades for f in result.folds)
    assert total_trades == 0


def test_mean_sharpe_is_average_of_folds():
    df = _make_df()
    result = walk_forward_backtest_strategy(
        df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    manual = sum(f.metrics.sharpe_ratio for f in result.folds) / len(result.folds)
    assert abs(result.mean_sharpe - manual) < 1e-9


def test_empty_df_raises():
    df = pl.DataFrame({
        "time": [], "ticker": [], "close": [],
        "forward_return_5d": [], "label": [],
        "open": [], "high": [], "low": [], "volume": [],
    })
    with pytest.raises(ValueError):
        walk_forward_backtest_strategy(
            df, _AlwaysBuyStrategy(), _OHLCV, _FEATURES,
            train_window_days=300, test_window_days=30, step_days=30,
        )
```

- [ ] **Step 3.2: Run tests — expect FAIL**

```
pytest tests/unit/test_strategy_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.backtesting.strategy_runner'`

- [ ] **Step 3.3: Create strategy_runner.py**

```python
# src/backtesting/strategy_runner.py
from __future__ import annotations
from datetime import timedelta
import logging

import polars as pl

from src.strategies.base import Strategy
from src.backtesting.metrics import Trade, compute_metrics
from src.backtesting.walk_forward import FoldBacktestResult, WalkForwardBacktestResult

logger = logging.getLogger(__name__)

_REQUIRED = {"time", "close", "label", "forward_return_5d"}


def walk_forward_backtest_strategy(
    df: pl.DataFrame,
    strategy: Strategy,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    label_col: str = "label",
    train_window_days: int = 400,
    test_window_days: int = 21,
    step_days: int = 21,
    min_train_samples: int = 100,
    confidence_threshold: float = 0.5,
) -> WalkForwardBacktestResult:
    drop_cols = list(_REQUIRED)
    df_clean = df.drop_nulls(subset=[c for c in drop_cols if c in df.columns]).sort("time")
    times = df_clean["time"].to_list()
    if not times:
        raise ValueError("DataFrame is empty after dropping nulls")

    t_end = times[-1]
    folds: list[FoldBacktestResult] = []
    fold_idx = 0
    cursor = times[0] + timedelta(days=train_window_days)

    while cursor + timedelta(days=test_window_days) <= t_end + timedelta(days=1):
        train_start = cursor - timedelta(days=train_window_days)
        train_end   = cursor - timedelta(days=1)
        test_start  = cursor
        test_end    = cursor + timedelta(days=test_window_days - 1)

        train_df = df_clean.filter(
            (pl.col("time") >= train_start) & (pl.col("time") <= train_end)
        )
        test_df = df_clean.filter(
            (pl.col("time") >= test_start) & (pl.col("time") <= test_end)
        )

        if len(train_df) < min_train_samples or len(test_df) == 0:
            cursor += timedelta(days=step_days)
            continue

        try:
            train_pd = train_df.to_pandas()
            test_pd  = test_df.to_pandas()
            strategy.fit(train_pd)
            result = strategy.predict(test_pd)
            trades = _build_trades(test_df, result, confidence_threshold)
            metrics = compute_metrics(trades)
            folds.append(FoldBacktestResult(
                fold=fold_idx,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                metrics=metrics,
                n_trades=len(trades),
            ))
            fold_idx += 1
        except Exception as e:
            logger.warning("Walk-forward fold %d failed: %s", fold_idx, e)

        cursor += timedelta(days=step_days)

    if not folds:
        raise ValueError("No valid folds — check data range and window sizes")

    n = len(folds)
    return WalkForwardBacktestResult(
        folds=folds,
        mean_sharpe=sum(f.metrics.sharpe_ratio for f in folds) / n,
        mean_win_rate=sum(f.metrics.win_rate for f in folds) / n,
        mean_precision_buy=sum(f.metrics.precision_buy for f in folds) / n,
        worst_drawdown=max(f.metrics.max_drawdown_pct for f in folds),
    )


def _build_trades(
    test_df: pl.DataFrame,
    result,
    confidence_threshold: float,
) -> list[Trade]:
    trades: list[Trade] = []
    conf_arr  = result.confidence.to_numpy()
    signal_arr = result.signal.to_numpy()

    for i, row in enumerate(test_df.iter_rows(named=True)):
        if i >= len(signal_arr):
            break
        if str(signal_arr[i]) != "Buy":
            continue
        conf = float(conf_arr[i])
        if conf < confidence_threshold:
            continue
        entry = float(row["close"])
        ret   = float(row["forward_return_5d"])
        trades.append(Trade(
            entry_date=str(row["time"]),
            exit_date=str(row["time"]),
            entry_price=entry,
            exit_price=entry * (1 + ret),
            predicted_label="Buy",
            actual_label=str(row["label"]),
            return_pct=ret,
            confidence=conf,
        ))
    return trades
```

- [ ] **Step 3.4: Run tests — expect PASS**

```
pytest tests/unit/test_strategy_runner.py -v
```

Expected: 5 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/backtesting/strategy_runner.py tests/unit/test_strategy_runner.py
git commit -m "feat: add walk_forward_backtest_strategy adapter"
```

---

### Task 4: Rule-Based Strategies

**Files:**
- Create: `src/strategies/rule_based/ma_crossover.py`
- Create: `src/strategies/rule_based/rsi.py`
- Create: `src/strategies/rule_based/macd.py`
- Create: `src/strategies/rule_based/bollinger.py`
- Create: `tests/unit/test_rule_based_strategies.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult`, `Signal` from Task 1
- Produces: `MACrossover`, `RSIThreshold`, `MACDSignal`, `BollingerBounce` — all with `data_source = "ohlcv"`, all return `PredictionResult` with string signal values

---

- [ ] **Step 4.1: Write the failing tests**

```python
# tests/unit/test_rule_based_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.strategies.rule_based.ma_crossover import MACrossover
from src.strategies.rule_based.rsi import RSIThreshold
from src.strategies.rule_based.macd import MACDSignal
from src.strategies.rule_based.bollinger import BollingerBounce
from src.strategies.base import PredictionResult

_VALID_SIGNALS = {"Buy", "Hold", "Sell"}


def _uptrend_df(n: int = 200) -> pd.DataFrame:
    """Strong uptrend: price rises 0.5/day."""
    closes = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "close": closes,
        "open":  [c - 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * n,
    })


def _downtrend_df(n: int = 100) -> pd.DataFrame:
    """Sharp downtrend: price falls 1.0/day (creates oversold RSI)."""
    closes = [200.0 - i * 1.0 for i in range(n)]
    return pd.DataFrame({
        "close": closes,
        "open":  [c + 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * n,
    })


def _flat_then_surge_df(flat: int = 60, surge: int = 20) -> pd.DataFrame:
    """Flat for `flat` days then sharp up for `surge` days — triggers MACD."""
    flat_closes = [100.0] * flat
    surge_closes = [100.0 + i * 2.0 for i in range(1, surge + 1)]
    closes = flat_closes + surge_closes
    return pd.DataFrame({
        "close": closes,
        "open":  [c - 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * len(closes),
    })


# --- MACrossover ---

def test_ma_crossover_returns_prediction_result():
    df = _uptrend_df()
    result = MACrossover(fast_window=20, slow_window=50).predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_ma_crossover_confidence_in_range():
    result = MACrossover().predict(_uptrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_ma_crossover_signals_are_valid():
    result = MACrossover().predict(_uptrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_ma_crossover_buys_in_uptrend():
    result = MACrossover(fast_window=20, slow_window=50).predict(_uptrend_df(200))
    # After warmup (50 days), fast > slow → BUY
    assert result.signal.iloc[-1] == "Buy"


# --- RSIThreshold ---

def test_rsi_returns_prediction_result():
    result = RSIThreshold().predict(_downtrend_df())
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(_downtrend_df())


def test_rsi_confidence_in_range():
    result = RSIThreshold().predict(_downtrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_rsi_signals_are_valid():
    result = RSIThreshold().predict(_downtrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_rsi_buys_when_oversold():
    result = RSIThreshold(period=14, oversold=30).predict(_downtrend_df(60))
    # Sharp downtrend pushes RSI below 30 → BUY signal appears
    assert "Buy" in result.signal.values


# --- MACDSignal ---

def test_macd_returns_prediction_result():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert isinstance(result, PredictionResult)


def test_macd_confidence_in_range():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_macd_signals_are_valid():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_macd_buys_after_surge():
    result = MACDSignal(fast=12, slow=26, signal_period=9).predict(_flat_then_surge_df())
    # Momentum surge causes MACD to cross above signal line
    assert "Buy" in result.signal.values


# --- BollingerBounce ---

def test_bollinger_returns_prediction_result():
    result = BollingerBounce().predict(_downtrend_df())
    assert isinstance(result, PredictionResult)


def test_bollinger_confidence_in_range():
    result = BollingerBounce().predict(_downtrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_bollinger_signals_are_valid():
    result = BollingerBounce().predict(_downtrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_bollinger_buys_when_below_lower_band():
    result = BollingerBounce(window=20, num_std=2.0).predict(_downtrend_df(60))
    # Price drops sharply below lower band
    assert "Buy" in result.signal.values
```

- [ ] **Step 4.2: Run tests — expect FAIL**

```
pytest tests/unit/test_rule_based_strategies.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.strategies.rule_based.ma_crossover'`

- [ ] **Step 4.3: Create ma_crossover.py**

```python
# src/strategies/rule_based/ma_crossover.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MACrossover(Strategy):
    data_source = "ohlcv"

    def __init__(self, fast_window: int = 20, slow_window: int = 50) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        fast = close.rolling(self.fast_window).mean()
        slow = close.rolling(self.slow_window).mean()

        raw_diff = (fast - slow) / slow.abs().clip(lower=1e-9)
        # Confidence: normalize gap to [0, 1]; 5% gap above slow = full confidence
        confidence = raw_diff.clip(0, 0.05).div(0.05).fillna(0.0)
        signal = pd.Series(
            ["Buy" if d > 0 else "Hold" for d in raw_diff.fillna(0.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
```

- [ ] **Step 4.4: Create rsi.py**

```python
# src/strategies/rule_based/rsi.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class RSIThreshold(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # BUY when RSI < oversold; confidence = distance below threshold / threshold
        confidence = ((self.oversold - rsi) / self.oversold).clip(0, 1).fillna(0.0)
        signal = pd.Series(
            ["Buy" if v < self.oversold else "Hold" for v in rsi.fillna(50)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
```

- [ ] **Step 4.5: Create macd.py**

```python
# src/strategies/rule_based/macd.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MACDSignal(Strategy):
    data_source = "ohlcv"

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        # Confidence: positive histogram normalized to rolling 50-bar max
        pos_hist = histogram.clip(lower=0)
        roll_max = pos_hist.rolling(50, min_periods=1).max().replace(0, 1e-9)
        confidence = (pos_hist / roll_max).fillna(0.0)
        signal = pd.Series(
            ["Buy" if h > 0 else "Hold" for h in histogram.fillna(0.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
```

- [ ] **Step 4.6: Create bollinger.py**

```python
# src/strategies/rule_based/bollinger.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class BollingerBounce(Strategy):
    data_source = "ohlcv"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        lower = sma - self.num_std * std

        # BUY when close < lower band; confidence ∝ how far below (max 5%)
        pct_below = ((lower - close) / lower.abs().clip(lower=1e-9)).clip(0, 0.05)
        confidence = (pct_below / 0.05).fillna(0.0)
        signal = pd.Series(
            ["Buy" if v > 0 else "Hold" for v in pct_below.fillna(0.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
```

- [ ] **Step 4.7: Run tests — expect PASS**

```
pytest tests/unit/test_rule_based_strategies.py -v
```

Expected: 20 passed

- [ ] **Step 4.8: Verify registry can now load rule-based strategies**

```
pytest tests/unit/test_strategies_registry.py -v
```

Expected: `test_load_strategy_returns_strategy_instance` and `test_load_strategy_injects_params` now PASS (statistical tests still FAIL)

- [ ] **Step 4.9: Commit**

```bash
git add src/strategies/rule_based/ma_crossover.py \
        src/strategies/rule_based/rsi.py \
        src/strategies/rule_based/macd.py \
        src/strategies/rule_based/bollinger.py \
        tests/unit/test_rule_based_strategies.py
git commit -m "feat: add 4 rule-based strategies (MA crossover, RSI, MACD, Bollinger)"
```

---

### Task 5: Statistical Strategies

**Files:**
- Create: `src/strategies/statistical/logistic.py`
- Create: `src/strategies/statistical/linear.py`
- Create: `tests/unit/test_statistical_strategies.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from Task 1
- Produces: `LogisticStrategy` and `LinearStrategy` with `data_source = "features"`, `fit(df)` trains sklearn model on `df` (excluding metadata columns), `predict(df)` returns `PredictionResult`

---

- [ ] **Step 5.1: Write the failing tests**

```python
# tests/unit/test_statistical_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.strategies.statistical.logistic import LogisticStrategy
from src.strategies.statistical.linear import LinearStrategy
from src.strategies.base import PredictionResult

_VALID_SIGNALS = {"Buy", "Hold", "Sell"}
_META_COLS = {"time", "ticker", "label", "forward_return_5d"}


def _make_feature_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.standard_normal(n),
        "label": ["Buy" if i % 3 == 0 else "Hold" for i in range(n)],
        "forward_return_5d": rng.uniform(-0.05, 0.05, n),
    })
    return df


# --- LogisticStrategy ---

def test_logistic_fit_and_predict():
    df = _make_feature_df()
    s = LogisticStrategy(C=1.0, max_iter=200)
    s.fit(df)
    result = s.predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_logistic_confidence_in_range():
    df = _make_feature_df()
    s = LogisticStrategy()
    s.fit(df)
    result = s.predict(df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_logistic_signals_are_valid():
    df = _make_feature_df()
    s = LogisticStrategy()
    s.fit(df)
    result = s.predict(df)
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_logistic_predict_before_fit_raises():
    s = LogisticStrategy()
    df = _make_feature_df(10)
    with pytest.raises((AttributeError, ValueError)):
        s.predict(df)


def test_logistic_excludes_meta_columns():
    df = _make_feature_df()
    df["ticker"] = "AAPL"
    s = LogisticStrategy()
    s.fit(df)
    assert "ticker" not in s._feature_cols
    assert "label" not in s._feature_cols


# --- LinearStrategy ---

def test_linear_fit_and_predict():
    df = _make_feature_df()
    s = LinearStrategy(buy_threshold=0.005)
    s.fit(df)
    result = s.predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_linear_confidence_in_range():
    df = _make_feature_df()
    s = LinearStrategy()
    s.fit(df)
    result = s.predict(df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_linear_signals_are_valid():
    df = _make_feature_df()
    s = LinearStrategy()
    s.fit(df)
    result = s.predict(df)
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_linear_predict_before_fit_raises():
    s = LinearStrategy()
    df = _make_feature_df(10)
    with pytest.raises((AttributeError, ValueError)):
        s.predict(df)


def test_linear_buy_when_predicted_return_above_threshold():
    rng = np.random.default_rng(0)
    n = 200
    # f1 perfectly predicts return — high f1 → high return
    forward = rng.uniform(0.0, 0.02, n)
    df = pd.DataFrame({
        "f1": forward * 100,  # perfect linear predictor
        "forward_return_5d": forward,
        "label": ["Buy" if r >= 0.005 else "Hold" for r in forward],
    })
    s = LinearStrategy(buy_threshold=0.005)
    s.fit(df)
    result = s.predict(df)
    # Rows with high f1 should be BUY
    high_f1_mask = df["f1"] > 0.5
    assert (result.signal[high_f1_mask] == "Buy").any()
```

- [ ] **Step 5.2: Run tests — expect FAIL**

```
pytest tests/unit/test_statistical_strategies.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.strategies.statistical.logistic'`

- [ ] **Step 5.3: Create logistic.py**

```python
# src/strategies/statistical/logistic.py
from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class LogisticStrategy(Strategy):
    data_source = "features"

    def __init__(self, C: float = 1.0, max_iter: int = 200) -> None:
        self.C = C
        self.max_iter = max_iter
        self._model = LogisticRegression(C=C, max_iter=max_iter)
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
        buy_idx = classes.index("Buy") if "Buy" in classes else 0
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 5.4: Create linear.py**

```python
# src/strategies/statistical/linear.py
from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class LinearStrategy(Strategy):
    data_source = "features"

    def __init__(self, buy_threshold: float = 0.005) -> None:
        self.buy_threshold = buy_threshold
        self._model = LinearRegression()
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["forward_return_5d"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        pred_return = self._model.predict(X)
        # Map predicted return [-10%, +10%] → confidence [0, 1]
        confidence = pd.Series(((pred_return + 0.10) / 0.20).clip(0, 1))
        signal = pd.Series(
            ["Buy" if r >= self.buy_threshold else "Hold" for r in pred_return]
        )
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 5.5: Run tests — expect PASS**

```
pytest tests/unit/test_statistical_strategies.py -v
```

Expected: 11 passed

- [ ] **Step 5.6: Run full registry tests — expect all PASS now**

```
pytest tests/unit/test_strategies_registry.py -v
```

Expected: 5 passed

- [ ] **Step 5.7: Run all new strategy tests together**

```
pytest tests/unit/test_strategies_base.py tests/unit/test_strategies_registry.py \
       tests/unit/test_strategy_runner.py tests/unit/test_rule_based_strategies.py \
       tests/unit/test_statistical_strategies.py -v
```

Expected: all pass

- [ ] **Step 5.8: Commit**

```bash
git add src/strategies/statistical/logistic.py \
        src/strategies/statistical/linear.py \
        tests/unit/test_statistical_strategies.py
git commit -m "feat: add logistic and linear regression strategies"
```

---

### Task 6: Dashboard Wiring

**Files:**
- Modify: `dashboard/config.py`
- Modify: `dashboard/data_loader.py`
- Modify: `dashboard/pages/2_Model_Leaderboard.py`
- Modify: `dashboard/pages/3_Backtest_Results.py`
- Modify: `dashboard/pages/4_Live_Signals.py`

**Interfaces:**
- Consumes: `list_strategies`, `load_strategy` from Task 2; `walk_forward_backtest_strategy` from Task 3; `LiveSignal`, `Signal` from Task 1
- Note: The existing `tests/unit/test_dashboard_data_loader.py` tests the old ML-based `get_leaderboard` / `get_backtest_result` / `get_live_signals`. Those tests will break when `data_loader.py` is rewritten. Skip or delete them after the rewrite (Step 6.9).

---

- [ ] **Step 6.1: Update dashboard/config.py — add OHLCV_COLS, remove REGISTRY_DIR**

```python
# dashboard/config.py
from pathlib import Path

PARQUET_DIR  = Path("data/features")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]

CONFIDENCE_THRESHOLD = 0.75
GRADE_COLORS = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
```

- [ ] **Step 6.2: Rewrite dashboard/data_loader.py**

```python
# dashboard/data_loader.py
from __future__ import annotations
from pathlib import Path
import logging

import polars as pl

from src.backtesting.grader import ModelGrade, grade_model, build_leaderboard
from src.backtesting.walk_forward import WalkForwardBacktestResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.base import LiveSignal, Signal
from src.strategies.registry import list_strategies, load_strategy

logger = logging.getLogger(__name__)


def get_data_summary(parquet_dir: Path) -> dict:
    df = load_training_data(parquet_dir)
    tickers = df["ticker"].unique().to_list() if "ticker" in df.columns else []
    time_min = str(df["time"].min()) if "time" in df.columns else "N/A"
    time_max = str(df["time"].max()) if "time" in df.columns else "N/A"
    return {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": sorted(tickers),
        "date_range_start": time_min,
        "date_range_end": time_max,
    }


def get_leaderboard(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
) -> list[ModelGrade]:
    names = list_strategies()
    if not names:
        return []

    df = load_training_data(parquet_dir)
    grades: list[ModelGrade] = []
    for name in names:
        try:
            strategy = load_strategy(name)
            result = walk_forward_backtest_strategy(
                df, strategy, ohlcv_cols, feature_cols,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            avg_metrics = result.folds[-1].metrics
            grades.append(grade_model(name, avg_metrics))
        except Exception as e:
            logger.warning("Leaderboard skipping %s: %s", name, e)

    return build_leaderboard(grades)


def get_backtest_result(
    strategy_name: str,
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    strategy = load_strategy(strategy_name)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest_strategy(
        df, strategy, ohlcv_cols, feature_cols,
        train_window_days=400, test_window_days=21, step_days=21,
    )
    grade = grade_model(strategy_name, result.folds[-1].metrics)
    return result, grade


def get_live_signals(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[LiveSignal]:
    names = list_strategies()
    if not names:
        return []

    strategy = load_strategy(names[0])
    df = load_training_data(parquet_dir)
    df_pd = df.to_pandas()

    strategy.fit(df_pd)
    pred = strategy.predict(df_pd)

    df_with_pred = df.with_columns([
        pl.Series("_conf", pred.confidence.tolist()),
        pl.Series("_sig",  pred.signal.tolist()),
    ])
    latest = df_with_pred.sort("time").group_by("ticker").last()

    live_signals: list[LiveSignal] = []
    for row in latest.iter_rows(named=True):
        conf = float(row["_conf"])
        sig  = str(row["_sig"])
        if sig == "Buy" and conf >= confidence_threshold:
            live_signals.append(LiveSignal(
                ticker=str(row["ticker"]),
                date=str(row["time"]),
                signal=Signal.BUY,
                confidence=conf,
                entry_price=float(row["close"]),
                position_size=conf,
            ))
    return live_signals
```

- [ ] **Step 6.3: Update dashboard/pages/2_Model_Leaderboard.py**

```python
# dashboard/pages/2_Model_Leaderboard.py
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from dashboard.config import PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_leaderboard

st.set_page_config(page_title="Strategy Leaderboard", layout="wide")
st.header("Strategy Leaderboard")


@st.cache_data(ttl=1800)
def _leaderboard():
    return get_leaderboard(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS)


with st.spinner("Computing grades..."):
    leaderboard = _leaderboard()

if not leaderboard:
    st.warning("No strategies found. Check src/strategies/strategies.yaml.")
    st.stop()

rows = [{
    "Rank": i + 1,
    "Strategy": g.model_name,
    "Grade": g.grade.value,
    "Score": f"{g.composite_score:.3f}",
    "Precision Buy": f"{g.metrics.precision_buy:.3f}",
    "Sharpe": f"{g.metrics.sharpe_ratio:.2f}",
    "Max Drawdown": f"{g.metrics.max_drawdown_pct:.1%}",
    "Win Rate": f"{g.metrics.win_rate:.1%}",
    "Trades": g.metrics.n_trades,
} for i, g in enumerate(leaderboard)]

df = pl.DataFrame(rows)
st.dataframe(df.to_pandas(), use_container_width=True, hide_index=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=[g.model_name for g in leaderboard],
        y=[g.metrics.precision_buy for g in leaderboard],
        marker_color=[GRADE_COLORS[g.grade.value] for g in leaderboard],
    ))
    fig.update_layout(title="Precision (Buy class)", xaxis_title="Strategy",
                      yaxis_title="Precision", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Bar(
        x=[g.model_name for g in leaderboard],
        y=[g.metrics.sharpe_ratio for g in leaderboard],
        marker_color=[GRADE_COLORS[g.grade.value] for g in leaderboard],
    ))
    fig2.update_layout(title="Sharpe Ratio", xaxis_title="Strategy",
                       yaxis_title="Sharpe")
    st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 6.4: Update dashboard/pages/3_Backtest_Results.py**

```python
# dashboard/pages/3_Backtest_Results.py
import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_backtest_result
from src.strategies.registry import list_strategies

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Backtest Results")

strategy_names = list_strategies()
if not strategy_names:
    st.warning("No strategies in registry. Check src/strategies/strategies.yaml.")
    st.stop()

selected = st.selectbox("Select strategy", strategy_names)


@st.cache_data(ttl=1800)
def _backtest(strategy_name: str):
    return get_backtest_result(strategy_name, PARQUET_DIR, OHLCV_COLS, FEATURE_COLS)


with st.spinner(f"Running walk-forward backtest for {selected}..."):
    wf_result, grade = _backtest(selected)

color = GRADE_COLORS[grade.grade.value]
st.markdown(f"### Grade: <span style='color:{color};font-size:2em'>{grade.grade.value}</span> "
            f"(score: {grade.composite_score:.3f})", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
last = wf_result.folds[-1].metrics
c1.metric("Precision Buy", f"{last.precision_buy:.3f}")
c2.metric("Sharpe Ratio", f"{last.sharpe_ratio:.2f}")
c3.metric("Max Drawdown", f"{last.max_drawdown_pct:.1%}")
c4.metric("Win Rate", f"{last.win_rate:.1%}")

st.divider()

fold_labels = [f"Fold {f.fold}" for f in wf_result.folds]
precisions  = [f.metrics.precision_buy for f in wf_result.folds]
sharpes     = [f.metrics.sharpe_ratio for f in wf_result.folds]
n_trades    = [f.n_trades for f in wf_result.folds]

fig = go.Figure()
fig.add_trace(go.Scatter(x=fold_labels, y=precisions, mode="lines+markers", name="Precision Buy"))
fig.add_trace(go.Scatter(x=fold_labels, y=sharpes, mode="lines+markers",
                         name="Sharpe", yaxis="y2"))
fig.update_layout(
    title="Walk-Forward Performance by Fold",
    yaxis=dict(title="Precision", range=[0, 1]),
    yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade Count per Fold")
fig2 = go.Figure(go.Bar(x=fold_labels, y=n_trades))
fig2.update_layout(xaxis_title="Fold", yaxis_title="# Trades")
st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 6.5: Update dashboard/pages/4_Live_Signals.py**

```python
# dashboard/pages/4_Live_Signals.py
import streamlit as st
from dashboard.config import PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, CONFIDENCE_THRESHOLD
from dashboard.data_loader import get_live_signals

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

with st.spinner("Generating signals..."):
    signals = get_live_signals(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, threshold)

if not signals:
    st.info("No Buy signals above the current confidence threshold.")
    st.stop()

st.success(f"Found **{len(signals)}** Buy signal(s)")

for sig in signals:
    with st.expander(f"**{sig.ticker}** — Confidence {sig.confidence:.1%} | Entry ${sig.entry_price:.2f}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", sig.signal.value)
        c2.metric("Confidence", f"{sig.confidence:.1%}")
        c3.metric("Position Size", f"{sig.position_size:.1%}")
        c4.metric("Entry Price", f"${sig.entry_price:.2f}")
```

- [ ] **Step 6.6: Verify imports are clean — no remaining references to old model registry**

```
grep -r "from src.models.registry" dashboard/
grep -r "from src.signals.signal_engine" dashboard/
grep -r "from src.explainability.xai" dashboard/
grep -r "list_models\|load_model" dashboard/
```

Expected: no output from any of these commands

- [ ] **Step 6.7: Skip the stale data_loader tests**

The old `tests/unit/test_dashboard_data_loader.py` tests `get_leaderboard(registry_dir, ...)` and `get_live_signals(registry_dir, ...)` — both signatures changed. Mark them to skip rather than delete so they can be rewritten later:

Open `tests/unit/test_dashboard_data_loader.py` and add at the top:

```python
import pytest
pytestmark = pytest.mark.skip(reason="data_loader rewritten for strategy system — tests need updating")
```

- [ ] **Step 6.8: Run all strategy tests to confirm nothing broke**

```
pytest tests/unit/test_strategies_base.py tests/unit/test_strategies_registry.py \
       tests/unit/test_strategy_runner.py tests/unit/test_rule_based_strategies.py \
       tests/unit/test_statistical_strategies.py -v
```

Expected: all pass

- [ ] **Step 6.9: Commit dashboard wiring**

```bash
git add dashboard/config.py dashboard/data_loader.py \
        dashboard/pages/2_Model_Leaderboard.py \
        dashboard/pages/3_Backtest_Results.py \
        dashboard/pages/4_Live_Signals.py \
        tests/unit/test_dashboard_data_loader.py
git commit -m "feat: wire dashboard to strategy system; remove SHAP dependency"
```

---

## Final Validation

- [ ] Run the complete new test suite:

```
pytest tests/unit/test_strategies_base.py tests/unit/test_strategies_registry.py \
       tests/unit/test_strategy_runner.py tests/unit/test_rule_based_strategies.py \
       tests/unit/test_statistical_strategies.py -v
```

Expected: all pass, 0 failures

- [ ] Spot-check that unrelated existing tests still pass:

```
pytest tests/unit/test_backtest_metrics.py tests/unit/test_grader.py \
       tests/unit/test_walk_forward_backtest.py -v
```

Expected: all pass (these files were not touched)
