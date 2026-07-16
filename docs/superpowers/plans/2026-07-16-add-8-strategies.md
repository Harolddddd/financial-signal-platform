# Add 8 New Trading Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 new trading strategies (4 ML + 4 rule-based), register them, run walk-forward backtests for the new strategies only, then refresh the dashboard cache (leaderboard + signals) covering all 38 strategies.

**Architecture:** Each strategy follows the established project pattern — ML strategies subclass `Strategy` with `data_source = "features"` and implement `fit()` / `predict()`; rule-based strategies use `data_source = "ohlcv"` and are stateless. An incremental precompute script runs backtests for only the 8 new strategies (preserving the existing 30 `backtest_*.json` files), then regenerates the leaderboard and live signals from all 38.

**Tech Stack:** Python 3.11, scikit-learn (SVM/MLP/AdaBoost/KNN), pandas, polars, PyYAML, pytest, existing walk-forward runner in `src/backtesting/strategy_runner.py`

## Global Constraints

- `data_source = "features"` for all ML strategies; `data_source = "ohlcv"` for all rule-based
- `_META = {"time", "ticker", "label", "forward_return_5d"}` — columns excluded from feature matrix in ML `fit()`
- `fillna(0.0)` in both `fit()` and `predict()` for all 4 ML strategies (none handles NaN natively)
- Buy signal threshold `>= 0.6` on `predict_proba` confidence — all strategies
- `handles_nan` must be absent or `False` on all 4 new ML strategies
- Walk-forward: `train_window_days=400, test_window_days=21, step_days=21` (unchanged)
- Grading: `0.40 × precision_buy + 0.30 × tanh(sharpe/2) + 0.30 × (1 − min(drawdown,50%)/50%)`
- PYTHONPATH must be `c:/Users/h1810/.vscode/EXP` for all script runs
- No new data fetch — use existing 151-ticker feature parquets in `data/features/`
- Existing `data/cache/backtest_*.json` for the original 30 strategies must NOT be modified

---

## File Map

**Create (strategy files):**
- `src/strategies/statistical/svm_strategy.py`
- `src/strategies/statistical/mlp_classifier.py`
- `src/strategies/statistical/adaboost.py`
- `src/strategies/statistical/knn_classifier.py`
- `src/strategies/rule_based/ichimoku_cloud.py`
- `src/strategies/rule_based/chaikin_money_flow.py`
- `src/strategies/rule_based/aroon_oscillator.py`
- `src/strategies/rule_based/vwap_cross.py`

**Create (test files):**
- `tests/unit/test_svm_strategy.py`
- `tests/unit/test_mlp_classifier.py`
- `tests/unit/test_adaboost.py`
- `tests/unit/test_knn_classifier.py`
- `tests/unit/test_ichimoku_cloud.py`
- `tests/unit/test_chaikin_money_flow.py`
- `tests/unit/test_aroon_oscillator.py`
- `tests/unit/test_vwap_cross.py`

**Create (script):**
- `scripts/precompute_new_strategies.py`

**Modify:**
- `src/strategies/strategies.yaml` — append 8 new entries

---

## Task 1: SVMStrategy

**Files:**
- Create: `src/strategies/statistical/svm_strategy.py`
- Test: `tests/unit/test_svm_strategy.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `SVMStrategy` class with `data_source = "features"`, `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_svm_strategy.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.svm_strategy import SVMStrategy

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
    s = SVMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = SVMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = SVMStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = SVMStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert SVMStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(SVMStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_svm_strategy.py -v
```
Expected: ImportError or ModuleNotFoundError — `svm_strategy` does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/statistical/svm_strategy.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}
_MAX_TRAIN_ROWS = 10_000  # SVM scales O(n²); cap to keep fold time manageable


class SVMStrategy(Strategy):
    data_source = "features"

    def __init__(self, kernel: str = "rbf", C: float = 1.0) -> None:
        self._model = SVC(
            kernel=kernel, C=C,
            probability=True, class_weight="balanced",
            max_iter=1000, random_state=42,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
        if len(X) > _MAX_TRAIN_ROWS:
            idx = np.random.default_rng(42).choice(len(X), _MAX_TRAIN_ROWS, replace=False)
            X, y = X[idx], y[idx]
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_svm_strategy.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/svm_strategy.py tests/unit/test_svm_strategy.py
git commit -m "feat: add SVMStrategy"
```

---

## Task 2: MLPStrategy

**Files:**
- Create: `src/strategies/statistical/mlp_classifier.py`
- Test: `tests/unit/test_mlp_classifier.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `MLPStrategy` class with `data_source = "features"`, `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mlp_classifier.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.mlp_classifier import MLPStrategy

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
    s = MLPStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = MLPStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = MLPStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = MLPStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert MLPStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(MLPStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_mlp_classifier.py -v
```
Expected: ImportError — `mlp_classifier` does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/statistical/mlp_classifier.py
from __future__ import annotations
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class MLPStrategy(Strategy):
    data_source = "features"

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        max_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self._model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_mlp_classifier.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/mlp_classifier.py tests/unit/test_mlp_classifier.py
git commit -m "feat: add MLPStrategy (neural network)"
```

---

## Task 3: AdaBoostStrategy

**Files:**
- Create: `src/strategies/statistical/adaboost.py`
- Test: `tests/unit/test_adaboost.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `AdaBoostStrategy` class with `data_source = "features"`, `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_adaboost.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.adaboost import AdaBoostStrategy

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
    s = AdaBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = AdaBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = AdaBoostStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = AdaBoostStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert AdaBoostStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(AdaBoostStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_adaboost.py -v
```
Expected: ImportError — `adaboost` module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/statistical/adaboost.py
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class AdaBoostStrategy(Strategy):
    data_source = "features"

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        # Note: do not pass algorithm= — it was removed in sklearn 1.6 when SAMME became the only option
        self._model = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_adaboost.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/adaboost.py tests/unit/test_adaboost.py
git commit -m "feat: add AdaBoostStrategy"
```

---

## Task 4: KNNStrategy

**Files:**
- Create: `src/strategies/statistical/knn_classifier.py`
- Test: `tests/unit/test_knn_classifier.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `KNNStrategy` class with `data_source = "features"`, `fit(df)`, `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_knn_classifier.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.knn_classifier import KNNStrategy

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
    s = KNNStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_or_hold(sample_df):
    s = KNNStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(sample_df):
    s = KNNStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = KNNStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert KNNStrategy.data_source == "features"


def test_handles_nan_is_false():
    assert not getattr(KNNStrategy(), "handles_nan", False)
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_knn_classifier.py -v
```
Expected: ImportError — `knn_classifier` module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/statistical/knn_classifier.py
from __future__ import annotations
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class KNNStrategy(Strategy):
    data_source = "features"

    def __init__(self, n_neighbors: int = 10) -> None:
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            n_jobs=-1,
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_knn_classifier.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/knn_classifier.py tests/unit/test_knn_classifier.py
git commit -m "feat: add KNNStrategy"
```

---

## Task 5: IchimokuCloud

**Files:**
- Create: `src/strategies/rule_based/ichimoku_cloud.py`
- Test: `tests/unit/test_ichimoku_cloud.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `IchimokuCloud` class with `data_source = "ohlcv"`, stateless `predict(df) -> PredictionResult`
- Input df columns: `open`, `high`, `low`, `close`, `volume`, `time`

- [ ] **Step 1: Write the failing test**

The Ichimoku calculation requires at least 52 (senkou_b) + 26 (cloud shift) = 78 warm-up bars. Use n=200.

```python
# tests/unit/test_ichimoku_cloud.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.ichimoku_cloud import IchimokuCloud


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
    s = IchimokuCloud()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = IchimokuCloud()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = IchimokuCloud()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert IchimokuCloud.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = IchimokuCloud()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = IchimokuCloud()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_ichimoku_cloud.py -v
```
Expected: ImportError — module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/rule_based/ichimoku_cloud.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class IchimokuCloud(Strategy):
    """Buy when price is above the cloud (Senkou A & B) AND Tenkan > Kijun."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low  = df["low"]
        close = df["close"]

        tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
        # Cloud lines: shift(26) → today's cloud was computed 26 bars ago
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        cloud_top = senkou_a.where(senkou_a >= senkou_b, senkou_b)

        above_cloud = close > cloud_top
        tenkan_above_kijun = tenkan > kijun
        buy = above_cloud & tenkan_above_kijun

        # Confidence: normalised gap between close and cloud top (capped at 5%)
        gap = ((close - cloud_top) / cloud_top.replace(0, 1e-9)).clip(0, 0.05) / 0.05
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_ichimoku_cloud.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/ichimoku_cloud.py tests/unit/test_ichimoku_cloud.py
git commit -m "feat: add IchimokuCloud strategy"
```

---

## Task 6: ChaikinMoneyFlow

**Files:**
- Create: `src/strategies/rule_based/chaikin_money_flow.py`
- Test: `tests/unit/test_chaikin_money_flow.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `ChaikinMoneyFlow` class with `data_source = "ohlcv"`, stateless `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_chaikin_money_flow.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.chaikin_money_flow import ChaikinMoneyFlow


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
    s = ChaikinMoneyFlow()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = ChaikinMoneyFlow()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = ChaikinMoneyFlow()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert ChaikinMoneyFlow.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = ChaikinMoneyFlow()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = ChaikinMoneyFlow()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_chaikin_money_flow.py -v
```
Expected: ImportError — module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/rule_based/chaikin_money_flow.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ChaikinMoneyFlow(Strategy):
    """Buy when the 20-period Chaikin Money Flow is positive (> 0)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        vol   = df["volume"]

        hl_range = (high - low).replace(0.0, float("nan"))
        mfm = ((close - low) - (high - close)) / hl_range  # money flow multiplier [-1, 1]
        mfv = mfm * vol                                      # money flow volume

        cmf = (
            mfv.rolling(self.period).sum()
            / vol.rolling(self.period).sum().replace(0.0, float("nan"))
        )

        buy = cmf > 0.0
        confidence = cmf.clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_chaikin_money_flow.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/chaikin_money_flow.py tests/unit/test_chaikin_money_flow.py
git commit -m "feat: add ChaikinMoneyFlow strategy"
```

---

## Task 7: AroonOscillator

**Files:**
- Create: `src/strategies/rule_based/aroon_oscillator.py`
- Test: `tests/unit/test_aroon_oscillator.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `AroonOscillator` class with `data_source = "ohlcv"`, stateless `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_aroon_oscillator.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.aroon_oscillator import AroonOscillator


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
    s = AroonOscillator()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = AroonOscillator()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = AroonOscillator()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert AroonOscillator.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = AroonOscillator()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = AroonOscillator()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_aroon_oscillator.py -v
```
Expected: ImportError — module does not exist yet.

- [ ] **Step 3: Write the implementation**

Note on the formula: `rolling(period+1).apply(lambda x: x.argmax(), raw=True)` returns 0-indexed position from the LEFT (oldest=0, newest=period). So position=period → high is most recent → aroon_up=100. Dividing by period scales to [0, 100].

```python
# src/strategies/rule_based/aroon_oscillator.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class AroonOscillator(Strategy):
    """Buy when the Aroon oscillator is above +50 (strong upward momentum)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 25) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low  = df["low"]
        p = self.period

        aroon_up   = (high.rolling(p + 1).apply(lambda x: x.argmax(), raw=True) / p) * 100
        aroon_down = (low.rolling(p + 1).apply(lambda x: x.argmin(), raw=True)  / p) * 100
        aroon_osc  = aroon_up - aroon_down  # range [-100, 100]

        buy = aroon_osc > 50.0
        # Confidence: scale oscillator from 50–100 → 0–1
        confidence = ((aroon_osc - 50.0) / 50.0).clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_aroon_oscillator.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/aroon_oscillator.py tests/unit/test_aroon_oscillator.py
git commit -m "feat: add AroonOscillator strategy"
```

---

## Task 8: VWAPCross

**Files:**
- Create: `src/strategies/rule_based/vwap_cross.py`
- Test: `tests/unit/test_vwap_cross.py`

**Interfaces:**
- Consumes: `Strategy`, `PredictionResult` from `src.strategies.base`
- Produces: `VWAPCross` class with `data_source = "ohlcv"`, stateless `predict(df) -> PredictionResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vwap_cross.py
import pandas as pd
import numpy as np
import pytest
from src.strategies.rule_based.vwap_cross import VWAPCross


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
    s = VWAPCross()
    result = s.predict(ohlcv_df)
    assert len(result.signal) == len(ohlcv_df)
    assert len(result.confidence) == len(ohlcv_df)


def test_signals_are_buy_or_hold(ohlcv_df):
    s = VWAPCross()
    result = s.predict(ohlcv_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})


def test_confidence_in_unit_range(ohlcv_df):
    s = VWAPCross()
    result = s.predict(ohlcv_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_data_source_is_ohlcv():
    assert VWAPCross.data_source == "ohlcv"


def test_stateless_no_fit_required(ohlcv_df):
    s = VWAPCross()
    result = s.predict(ohlcv_df)
    assert result is not None


def test_confidence_zero_when_hold(ohlcv_df):
    s = VWAPCross()
    result = s.predict(ohlcv_df)
    hold_mask = result.signal == "Hold"
    assert (result.confidence[hold_mask] == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_vwap_cross.py -v
```
Expected: ImportError — module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/strategies/rule_based/vwap_cross.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class VWAPCross(Strategy):
    """Buy when close is above the rolling VWAP (price-above-VWAP trend confirmation)."""
    data_source = "ohlcv"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        vol   = df["volume"]
        typical = (df["high"] + df["low"] + close) / 3

        vol_sum = vol.rolling(self.window).sum().replace(0.0, float("nan"))
        vwap = (typical * vol).rolling(self.window).sum() / vol_sum

        buy = close > vwap
        # Confidence: percentage gap above VWAP, scaled so 10% above → confidence=1
        gap = ((close - vwap) / vwap.replace(0.0, float("nan"))).clip(0.0, 0.10) / 0.10
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

- [ ] **Step 4: Run tests to verify they pass**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/test_vwap_cross.py -v
```
Expected: 6/6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/rule_based/vwap_cross.py tests/unit/test_vwap_cross.py
git commit -m "feat: add VWAPCross strategy"
```

---

## Task 9: Register All 8 Strategies in strategies.yaml

**Files:**
- Modify: `src/strategies/strategies.yaml`

**Interfaces:**
- Consumes: All 8 strategy classes from Tasks 1–8
- Produces: `list_strategies()` returns 38 names; `load_strategy(name)` can load all 8 new ones

- [ ] **Step 1: Append 8 entries to strategies.yaml**

Open `src/strategies/strategies.yaml` and append the following after the last `extra_trees` entry:

```yaml
  - name: svm_strategy
    class: src.strategies.statistical.svm_strategy.SVMStrategy
    params:
      kernel: rbf
      C: 1.0

  - name: mlp_classifier
    class: src.strategies.statistical.mlp_classifier.MLPStrategy
    params:
      hidden_layer_sizes: [64, 32]
      max_iter: 200

  - name: adaboost
    class: src.strategies.statistical.adaboost.AdaBoostStrategy
    params:
      n_estimators: 100

  - name: knn_classifier
    class: src.strategies.statistical.knn_classifier.KNNStrategy
    params:
      n_neighbors: 10

  - name: ichimoku_cloud
    class: src.strategies.rule_based.ichimoku_cloud.IchimokuCloud

  - name: chaikin_money_flow
    class: src.strategies.rule_based.chaikin_money_flow.ChaikinMoneyFlow
    params:
      period: 20

  - name: aroon_oscillator
    class: src.strategies.rule_based.aroon_oscillator.AroonOscillator
    params:
      period: 25

  - name: vwap_cross
    class: src.strategies.rule_based.vwap_cross.VWAPCross
    params:
      window: 20
```

- [ ] **Step 2: Verify the registry loads all 38 strategies**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python -c "from src.strategies.registry import list_strategies, load_strategy; names = list_strategies(); print(f'{len(names)} strategies'); [load_strategy(n) for n in names]; print('all loaded OK')"
```
Expected output:
```
38 strategies
all loaded OK
```

- [ ] **Step 3: Run all unit tests to confirm nothing is broken**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; pytest tests/unit/ -v --tb=short
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/strategies/strategies.yaml
git commit -m "feat: register 8 new strategies — 38 total in registry"
```

---

## Task 10: Incremental Precompute Script

**Files:**
- Create: `scripts/precompute_new_strategies.py`

**Interfaces:**
- Consumes: `walk_forward_backtest_strategy`, `load_strategy`, `list_strategies`, `load_training_data`, `grade_model`, `BacktestMetrics` from existing modules
- Consumes: `step_leaderboard`, `step_signals` from `scripts.precompute_dashboard` (imported directly to avoid duplication)
- Produces: 8 new `data/cache/backtest_*.json` files; updated `data/cache/leaderboard.json`; updated `data/cache/signals.json`

- [ ] **Step 1: Write the script**

```python
# scripts/precompute_new_strategies.py
"""
Backtest the 8 new strategies only, then regenerate leaderboard and signals.
Existing backtest_*.json for the original 30 strategies are untouched.

Usage:
    $env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"
    python scripts/precompute_new_strategies.py
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.registry import load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _metrics_dict(m) -> dict:
    return {
        "n_trades": m.n_trades,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "total_return_pct": m.total_return_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "max_drawdown_pct": m.max_drawdown_pct,
        "precision_buy": m.precision_buy,
        "recall_buy": m.recall_buy,
        "f1_buy": m.f1_buy,
        "accuracy": m.accuracy,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("  wrote %s", path)


def step_new_backtests(df) -> None:
    logger.info("[1/3] backtests — %d new strategies", len(_NEW_STRATEGIES))
    for name in _NEW_STRATEGIES:
        logger.info("  strategy: %s", name)
        try:
            strategy = load_strategy(name)
            wf = walk_forward_backtest_strategy(
                df, strategy, OHLCV_COLS, FEATURE_COLS,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            total_trades = sum(f.n_trades for f in wf.folds)
            avg_metrics = BacktestMetrics(
                n_trades=total_trades,
                win_rate=wf.mean_win_rate,
                profit_factor=0.0,
                total_return_pct=0.0,
                sharpe_ratio=wf.mean_sharpe,
                max_drawdown_pct=wf.worst_drawdown,
                precision_buy=wf.mean_precision_buy,
                recall_buy=0.0,
                f1_buy=0.0,
                accuracy=0.0,
            )
            g = grade_model(name, avg_metrics)
            _write(CACHE_DIR / f"backtest_{_safe(name)}.json", {
                "generated_at": _now(),
                "strategy_name": name,
                "mean_sharpe": wf.mean_sharpe,
                "mean_win_rate": wf.mean_win_rate,
                "mean_precision_buy": wf.mean_precision_buy,
                "worst_drawdown": wf.worst_drawdown,
                "grade": {
                    "model_name": g.model_name,
                    "grade": g.grade.value,
                    "composite_score": g.composite_score,
                    "metrics": _metrics_dict(g.metrics),
                },
                "folds": [
                    {
                        "fold": f.fold,
                        "train_start": f.train_start,
                        "train_end": f.train_end,
                        "test_start": f.test_start,
                        "test_end": f.test_end,
                        "n_trades": f.n_trades,
                        "metrics": _metrics_dict(f.metrics),
                    }
                    for f in wf.folds
                ],
            })
            logger.info(
                "    trades=%d  sharpe=%.3f  prec_buy=%.3f  grade=%s",
                total_trades, wf.mean_sharpe, wf.mean_precision_buy, g.grade.value,
            )
        except Exception as exc:
            logger.error("  FAILED %s: %s", name, exc)


def main() -> None:
    logger.info("=== Incremental precompute: 8 new strategies ===")
    df = load_training_data(PARQUET_DIR)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    step_new_backtests(df)

    # Reuse existing step_leaderboard and step_signals — they read all backtest_*.json
    # and run all 38 strategies respectively.
    from scripts.precompute_dashboard import step_leaderboard, step_signals
    logger.info("[2/3] leaderboard — aggregating all 38 backtest files")
    step_leaderboard()
    logger.info("[3/3] live signals — running all 38 strategies")
    step_signals()

    logger.info("=== Done. Run: git add data/cache/ && git commit && git push ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without import errors**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python -c "import scripts.precompute_new_strategies; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/precompute_new_strategies.py
git commit -m "feat: add incremental precompute script for 8 new strategies"
```

---

## Task 11: Run Backtest, Update Cache, Commit, Push

**Files:**
- Modify (generated): `data/cache/backtest_svm_strategy.json`
- Modify (generated): `data/cache/backtest_mlp_classifier.json`
- Modify (generated): `data/cache/backtest_adaboost.json`
- Modify (generated): `data/cache/backtest_knn_classifier.json`
- Modify (generated): `data/cache/backtest_ichimoku_cloud.json`
- Modify (generated): `data/cache/backtest_chaikin_money_flow.json`
- Modify (generated): `data/cache/backtest_aroon_oscillator.json`
- Modify (generated): `data/cache/backtest_vwap_cross.json`
- Modify (generated): `data/cache/leaderboard.json`
- Modify (generated): `data/cache/signals.json`

**Estimated runtime:** ~2–4 hours total. Rule-based strategies are fast (~minutes each via the stateless path). ML strategies (SVM, MLP, AdaBoost, KNN) take ~20–90 minutes each across ~740 folds. KNN is slow at predict time; SVM and MLP at fit time. Monitor the log for per-strategy completion.

- [ ] **Step 1: Run the incremental precompute script**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python scripts/precompute_new_strategies.py 2>&1 | Tee-Object -FilePath precompute_new.log
```

Monitor progress in `precompute_new.log`. Each strategy logs `trades=N sharpe=X prec_buy=Y grade=Z` when complete. Wait for `=== Done ===`.

- [ ] **Step 2: Verify all 8 new backtest files exist**

```
ls data/cache/backtest_svm_strategy.json
ls data/cache/backtest_mlp_classifier.json
ls data/cache/backtest_adaboost.json
ls data/cache/backtest_knn_classifier.json
ls data/cache/backtest_ichimoku_cloud.json
ls data/cache/backtest_chaikin_money_flow.json
ls data/cache/backtest_aroon_oscillator.json
ls data/cache/backtest_vwap_cross.json
```

- [ ] **Step 3: Verify the leaderboard has 38 entries**

```
$env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"; python -c "import json; lb = json.loads(open('data/cache/leaderboard.json').read()); print(f'{len(lb[\"grades\"])} strategies in leaderboard'); [print(f'  {g[\"model_name\"]}: {g[\"grade\"]} / {g[\"composite_score\"]:.3f}') for g in lb['grades'][:5]]"
```
Expected: `38 strategies in leaderboard`

- [ ] **Step 4: Commit cache and push**

```bash
git add data/cache/
git commit -m "data: precompute cache — 38 strategies, 151 tickers, history from 1980"
git push
```
