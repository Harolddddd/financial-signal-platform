# Simple Strategy Replacement Design

**Date:** 2026-06-18
**Status:** Approved

## Summary

Replace the existing complex ML model zoo (RandomForest, XGBoost, LightGBM, MLP, LSTM, SVM) with a plugin-based `Strategy` system comprising rule-based technical analysis strategies and simple statistical models. The existing walk-forward backtesting, grading, and dashboard infrastructure remain unchanged. New strategies are added by creating one file and adding one entry to a YAML config — no core code changes required.

---

## Architecture Overview

Three new layers replace the model registry and training pipeline:

```
src/strategies/
├── base.py                  # Strategy ABC, Signal enum, PredictionResult
├── registry.py              # list_strategies(), load_strategy()
├── strategies.yaml          # enabled strategies + constructor params
├── rule_based/
│   ├── ma_crossover.py      # Moving average crossover
│   ├── rsi.py               # RSI threshold
│   ├── macd.py              # MACD signal line crossover
│   └── bollinger.py         # Bollinger Band bounce
└── statistical/
    ├── logistic.py          # Logistic regression on feature store
    └── linear.py            # Linear regression on feature store
```

**What stays untouched:**
- `src/backtesting/walk_forward.py` — fold-splitting logic
- `src/backtesting/grader.py` — `grade_model()`, `build_leaderboard()`
- `dashboard/pages/2_Model_Leaderboard.py` — label change only ("Model" → "Strategy")
- `dashboard/pages/3_Backtest_Results.py` — no changes
- `dashboard/pages/1_Data_Overview.py` — no changes

**What changes:**
- `src/models/registry.py` — replaced by `src/strategies/registry.py`
- `dashboard/data_loader.py` — calls `list_strategies`/`load_strategy`; passes `ohlcv_cols`
- `dashboard/pages/4_Live_Signals.py` — uses `result.signal` for Buy/Hold/Sell display

---

## Strategy Interface

```python
# src/strategies/base.py

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
    confidence: pd.Series  # float [0.0, 1.0] — buy probability / signal strength
    signal: pd.Series      # Signal enum values


class Strategy(ABC):
    data_source: Literal["ohlcv", "features"]  # declared on each subclass

    def fit(self, df: pd.DataFrame) -> None:
        pass  # no-op for rule-based; overridden by statistical strategies

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        ...
```

### Output contract

| Strategy type | `confidence` | `signal` |
|---|---|---|
| Rule-based | Signal strength mapped to `[0, 1]` (e.g., RSI distance from threshold) | Primary output — derived from logic |
| Statistical | Predicted buy probability from model | Thresholded from confidence (e.g., `> 0.6 → BUY`) |

Both types return `PredictionResult` with both fields populated. The backtest loop uses `confidence` for grading metrics; the dashboard uses `signal` for the signal display and `confidence` for the meter.

---

## Registry & Configuration

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
      signal: 9

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

```python
# src/strategies/registry.py

import importlib
import yaml
from pathlib import Path
from .base import Strategy

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

**Adding a new strategy:** create one file in `rule_based/` or `statistical/`, add one entry to `strategies.yaml`. `registry.py` never changes.

---

## Backtest Runner Adapter

The existing `walk_forward_backtest()` passes pre-split `X/y` numpy arrays to sklearn-style models. Strategies receive a full DataFrame per fold instead. A thin adapter handles this:

```python
# src/backtesting/strategy_runner.py

def walk_forward_backtest_strategy(
    df: pd.DataFrame,
    strategy: Strategy,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    train_window_days: int = 400,
    test_window_days: int = 21,
    step_days: int = 21,
) -> WalkForwardResult:
    # Same fold-splitting logic as walk_forward_backtest()
    # Selects df[ohlcv_cols] or df[feature_cols] based on strategy.data_source
    # Calls strategy.fit(train_df), then strategy.predict(test_df)
    # Extracts result.confidence for grading metrics (same as predict_proba path)
```

The original `walk_forward_backtest()` is not deleted until strategies are validated on the leaderboard.

---

## Dashboard Wiring

### `dashboard/data_loader.py`
- `get_leaderboard()` — replace `list_models`/`load_model` with `list_strategies`/`load_strategy`; call `walk_forward_backtest_strategy()` instead of `walk_forward_backtest()`; pass both `ohlcv_cols` and `feature_cols`
- `get_backtest_result()` — same swap
- `get_live_signals()` — use `result.signal` for signal display; remove SHAP dependency entirely (no `attach_explanations()` call)

### `dashboard/pages/4_Live_Signals.py`
- Replace SHAP bar chart with a simple table of top features from `result.signal` per ticker
- `result.signal.value` drives the Buy/Hold/Sell badge; `result.confidence` drives the confidence meter

### `dashboard/config.py`
- Add `OHLCV_COLS: list[str]` constant
- Keep `FEATURE_COLS` for statistical strategies

### `dashboard/pages/2_Model_Leaderboard.py`
- Label: "Model" → "Strategy" in table header and chart axes only

---

## Adding a New Strategy (Checklist)

1. Create `src/strategies/rule_based/<name>.py` or `src/strategies/statistical/<name>.py`
2. Subclass `Strategy`, set `data_source`, implement `predict()` (and `fit()` if statistical)
3. Add entry to `src/strategies/strategies.yaml`
4. Run `python -m src.backtesting.strategy_runner --strategy <name>` to validate locally
5. Dashboard leaderboard picks it up automatically on next load

---

## Out of Scope

- SHAP explainability (removed entirely — no complex models to explain)
- Hyperparameter tuning / optimization
- Ensemble / combination strategies (future extension)
- The existing ML model files in `src/models/` (left in place, unused)
