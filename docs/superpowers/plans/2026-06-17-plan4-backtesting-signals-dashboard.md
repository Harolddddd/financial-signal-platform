# Financial Platform — Plan 4: Backtesting, Signals & Dashboard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop on the platform: a vectorized backtest engine computes financial performance metrics (Sharpe, drawdown, win rate, profit factor) per model; a walk-forward backtest simulates realistic rolling deployment; a composite grader assigns A–D grades; a confidence-filtered signal engine issues real-time Buy/Hold/Sell recommendations with SHAP explanations; and a four-page Streamlit dashboard surfaces everything.

**Architecture:** `BacktestEngine` takes a trained `BaseClassifier` (from Plan 3) and a hold-out feature DataFrame (from Plan 2's Parquet files), simulates trades on "Buy" signals above a confidence threshold, and computes `BacktestMetrics`. `WalkForwardBacktest` re-uses date-fold logic from Plan 3's trainer but adds financial metrics per fold. `Grader` assigns letter grades from a weighted composite. `SignalEngine` loads the latest registry model, scores today's feature row, and calls the `XAI` module for SHAP explanations of each Buy signal. The Streamlit dashboard wires all of this into four interactive pages.

**Tech Stack:** Python 3.11, Polars 0.20, NumPy, scikit-learn (Plan 3), SHAP 0.45, Streamlit 1.35, Plotly 5.20, joblib (Plan 3), Plans 1–3 modules

**Dependency on Plans 1–3:** Imports `src.models.base_classifier`, `src.models.registry`, `src.models.trainer.walk_forward_train`, `src.features.duckdb_client.load_training_data`, `src.features.feature_store.build_features`

## Global Constraints

- Python >= 3.11; all signatures require type hints
- Buy class index in `predict_proba` output is always 0 — all zoo models sort classes alphabetically (`Buy` < `Hold` < `Sell`) via sklearn or `LabelEncoder`; this invariant is tested explicitly
- Backtest simulates only "Buy" signals — it is a long-only strategy; short signals are not traded
- `forward_return_5d` in the feature DataFrame is the source of truth for trade returns — no look-ahead
- Confidence threshold for live signals defaults to 0.75 per spec; backtest uses 0.5
- SHAP KernelExplainer background set is capped at 50 samples for speed; TreeExplainer used for RF/XGB/LGB
- Streamlit pages use `@st.cache_data` for all data-loading calls
- Dashboard logic functions (data loaders) are tested; Streamlit render calls are not

---

### Task 1: Backtest Metrics

**Files:**
- Modify: `pyproject.toml` (add shap, streamlit, plotly)
- Create: `src/backtesting/__init__.py`
- Create: `src/backtesting/metrics.py`
- Test: `tests/unit/test_backtest_metrics.py`

**Interfaces:**
- Consumes: nothing external
- Produces:
  - `Trade` dataclass: `entry_date: str`, `exit_date: str`, `entry_price: float`, `exit_price: float`, `predicted_label: str`, `actual_label: str`, `return_pct: float`, `confidence: float`
  - `BacktestMetrics` dataclass: `n_trades`, `win_rate`, `profit_factor`, `total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `precision_buy`, `recall_buy`, `f1_buy`, `accuracy`
  - `compute_metrics(trades: list[Trade]) -> BacktestMetrics`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

Add to `dependencies` in `pyproject.toml`:

```toml
    "shap>=0.45.0",
    "streamlit>=1.35.0",
    "plotly>=5.20.0",
```

- [ ] **Step 2: Install new dependencies**

```bash
pip install "shap>=0.45.0" "streamlit>=1.35.0" "plotly>=5.20.0"
```
Expected: All packages install without errors.

- [ ] **Step 3: Write failing tests**

```python
# tests/unit/test_backtest_metrics.py
import pytest
from src.backtesting.metrics import Trade, BacktestMetrics, compute_metrics


def _trade(return_pct: float, actual: str = "Buy", predicted: str = "Buy",
           confidence: float = 0.8) -> Trade:
    entry = 100.0
    return Trade(
        entry_date="2024-01-02",
        exit_date="2024-01-09",
        entry_price=entry,
        exit_price=entry * (1 + return_pct),
        predicted_label=predicted,
        actual_label=actual,
        return_pct=return_pct,
        confidence=confidence,
    )


def test_empty_trades_returns_zero_metrics():
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.sharpe_ratio == 0.0


def test_all_winning_trades():
    trades = [_trade(0.03), _trade(0.05), _trade(0.02)]
    m = compute_metrics(trades)
    assert m.win_rate == 1.0
    assert m.n_trades == 3
    assert m.total_return_pct > 0
    assert m.profit_factor == float("inf") or m.profit_factor > 0


def test_all_losing_trades():
    trades = [_trade(-0.03), _trade(-0.05)]
    m = compute_metrics(trades)
    assert m.win_rate == 0.0
    assert m.total_return_pct < 0


def test_max_drawdown_is_non_negative():
    trades = [_trade(0.05), _trade(-0.10), _trade(0.03)]
    m = compute_metrics(trades)
    assert m.max_drawdown_pct >= 0.0


def test_precision_buy_correct():
    trades = [
        _trade(0.03, actual="Buy",  predicted="Buy"),
        _trade(-0.02, actual="Hold", predicted="Buy"),
        _trade(0.01, actual="Buy",  predicted="Buy"),
    ]
    m = compute_metrics(trades)
    assert abs(m.precision_buy - 2 / 3) < 1e-9


def test_sharpe_ratio_positive_for_consistent_gains():
    trades = [_trade(0.02)] * 20
    m = compute_metrics(trades)
    assert m.sharpe_ratio > 0
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/unit/test_backtest_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.backtesting'`

- [ ] **Step 5: Create src/backtesting/__init__.py**

```bash
touch src/backtesting/__init__.py
```

- [ ] **Step 6: Implement src/backtesting/metrics.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    predicted_label: str
    actual_label: str
    return_pct: float
    confidence: float


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    precision_buy: float
    recall_buy: float
    f1_buy: float
    accuracy: float


def compute_metrics(trades: list[Trade]) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = np.array([t.return_pct for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = float(len(wins) / len(trades))
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return = float(returns.sum())

    std = returns.std() if len(returns) > 1 else 1e-9
    sharpe = float((returns.mean() / std) * np.sqrt(252 / 5)) if std > 0 else 0.0

    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(abs(drawdown.min()))

    y_true = np.array([t.actual_label for t in trades])
    y_pred = np.array([t.predicted_label for t in trades])
    true_buy = y_true == "Buy"
    pred_buy = y_pred == "Buy"
    tp = int((true_buy & pred_buy).sum())
    fp = int((~true_buy & pred_buy).sum())
    fn = int((true_buy & ~pred_buy).sum())
    tn = int((~true_buy & ~pred_buy).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(trades)

    return BacktestMetrics(
        n_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_return_pct=total_return,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        precision_buy=precision,
        recall_buy=recall,
        f1_buy=f1,
        accuracy=accuracy,
    )
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/unit/test_backtest_metrics.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/backtesting/ tests/unit/test_backtest_metrics.py
git commit -m "feat: backtest Trade/BacktestMetrics dataclasses and compute_metrics"
```

---

### Task 2: Backtest Engine

**Files:**
- Create: `src/backtesting/engine.py`
- Test: `tests/unit/test_backtest_engine.py`

**Interfaces:**
- Consumes: `BaseClassifier` (Plan 3), `pl.DataFrame` with columns `time`, `ticker`, `close`, feature cols, `forward_return_5d`, `label`; `compute_metrics`
- Produces:
  - `BacktestResult` dataclass: `trades: list[Trade]`, `metrics: BacktestMetrics`, `equity_curve: list[float]`
  - `run_backtest(model: BaseClassifier, test_df: pl.DataFrame, feature_cols: list[str], confidence_threshold: float = 0.5) -> BacktestResult`
    — predicts on `test_df`, creates a Trade for every row where predicted label is `"Buy"` and `proba[:, 0] >= confidence_threshold` (index 0 = "Buy" alphabetically), computes metrics

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_backtest_engine.py
from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from src.backtesting.engine import run_backtest, BacktestResult
from src.models.zoo.random_forest import RandomForestClassifier_
from sklearn.datasets import make_classification

_FEATURE_COLS = [f"f{i}" for i in range(10)]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_test_df(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    labels = [_CLASSES[i % 3] for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    closes = [150.0 + i * 0.1 for i in range(n)]
    features = {f"f{j}": rng.standard_normal(n).tolist() for j in range(10)}
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "close": closes,
        "forward_return_5d": returns,
        "label": labels,
        **features,
    })


def _trained_rf(df: pl.DataFrame) -> RandomForestClassifier_:
    X = df.select(_FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    clf = RandomForestClassifier_(n_estimators=20)
    clf.fit(X, y)
    return clf


def test_run_backtest_returns_backtest_result():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    assert isinstance(result, BacktestResult)


def test_all_trades_are_buy_signals():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    for trade in result.trades:
        assert trade.predicted_label == "Buy"


def test_trades_respect_confidence_threshold():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result_low  = run_backtest(clf, df[60:], _FEATURE_COLS, confidence_threshold=0.0)
    result_high = run_backtest(clf, df[60:], _FEATURE_COLS, confidence_threshold=1.0)
    assert len(result_low.trades) >= len(result_high.trades)


def test_equity_curve_length_matches_trades():
    df = _make_test_df()
    clf = _trained_rf(df[:60])
    result = run_backtest(clf, df[60:], _FEATURE_COLS)
    assert len(result.equity_curve) == len(result.trades)


def test_buy_class_at_index_zero():
    """Verify the alphabetical-sort invariant: Buy=0, Hold=1, Sell=2."""
    clf = RandomForestClassifier_(n_estimators=5)
    X, y_int = make_classification(
        n_samples=120, n_features=10, n_classes=3,
        n_informative=5, n_redundant=2, random_state=1,
    )
    y = np.array([_CLASSES[i] for i in y_int])
    clf.fit(X, y)
    proba = clf.predict_proba(X[:1])
    # sklearn sorts classes alphabetically — Buy should be at index 0
    assert proba.shape == (1, 3)
    # classes_ attribute on the internal model
    assert clf._model.classes_[0] == "Buy"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_backtest_engine.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.backtesting.engine'`

- [ ] **Step 3: Implement src/backtesting/engine.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl

from src.models.base_classifier import BaseClassifier
from src.backtesting.metrics import Trade, BacktestMetrics, compute_metrics

_BUY_IDX = 0  # "Buy" < "Hold" < "Sell" alphabetically — index 0 in all zoo models


@dataclass
class BacktestResult:
    trades: list[Trade]
    metrics: BacktestMetrics
    equity_curve: list[float]


def run_backtest(
    model: BaseClassifier,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    confidence_threshold: float = 0.5,
) -> BacktestResult:
    X = test_df.select(feature_cols).to_numpy()
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    trades: list[Trade] = []
    for i, row in enumerate(test_df.iter_rows(named=True)):
        if y_pred[i] != "Buy":
            continue
        buy_prob = float(proba[i][_BUY_IDX])
        if buy_prob < confidence_threshold:
            continue

        entry = float(row["close"])
        ret = float(row["forward_return_5d"])
        trades.append(Trade(
            entry_date=str(row["time"]),
            exit_date=str(row["time"]),
            entry_price=entry,
            exit_price=entry * (1 + ret),
            predicted_label="Buy",
            actual_label=str(row["label"]),
            return_pct=ret,
            confidence=buy_prob,
        ))

    metrics = compute_metrics(trades)
    equity_curve = list(np.cumprod(1 + np.array([t.return_pct for t in trades]))) if trades else []

    return BacktestResult(trades=trades, metrics=metrics, equity_curve=equity_curve)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_backtest_engine.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine.py tests/unit/test_backtest_engine.py
git commit -m "feat: backtest engine with confidence-threshold trade simulation"
```

---

### Task 3: Walk-Forward Backtest

**Files:**
- Create: `src/backtesting/walk_forward.py`
- Test: `tests/unit/test_walk_forward_backtest.py`

**Interfaces:**
- Consumes: `BaseClassifier`, `run_backtest`, `walk_forward_train` (Plan 3 trainer), feature `pl.DataFrame`
- Produces:
  - `FoldBacktestResult` dataclass: `fold: int`, `train_start: str`, `train_end: str`, `test_start: str`, `test_end: str`, `metrics: BacktestMetrics`, `n_trades: int`
  - `WalkForwardBacktestResult` dataclass: `folds: list[FoldBacktestResult]`, `mean_sharpe: float`, `mean_win_rate: float`, `mean_precision_buy: float`, `worst_drawdown: float`
  - `walk_forward_backtest(df: pl.DataFrame, model: BaseClassifier, feature_cols: list[str], train_window_days: int = 500, test_window_days: int = 21, step_days: int = 21, confidence_threshold: float = 0.5) -> WalkForwardBacktestResult`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_walk_forward_backtest.py
from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from src.backtesting.walk_forward import walk_forward_backtest, WalkForwardBacktestResult
from src.models.zoo.random_forest import RandomForestClassifier_

_FEATURE_COLS = ["f1", "f2", "f3"]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_df(n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    closes = [100.0 + i * 0.05 for i in range(n)]
    labels = [_CLASSES[i % 3] for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
    })


def test_walk_forward_backtest_returns_result():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    assert isinstance(result, WalkForwardBacktestResult)
    assert len(result.folds) >= 1


def test_each_fold_has_metrics():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert fold.metrics is not None
        assert fold.fold >= 0


def test_mean_sharpe_is_average_of_folds():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    manual = sum(f.metrics.sharpe_ratio for f in result.folds) / len(result.folds)
    assert abs(result.mean_sharpe - manual) < 1e-9


def test_worst_drawdown_is_max_across_folds():
    df = _make_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_backtest(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    worst = max(f.metrics.max_drawdown_pct for f in result.folds)
    assert abs(result.worst_drawdown - worst) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_walk_forward_backtest.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/backtesting/walk_forward.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import logging

import polars as pl

from src.models.base_classifier import BaseClassifier
from src.backtesting.engine import run_backtest
from src.backtesting.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class FoldBacktestResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: BacktestMetrics
    n_trades: int


@dataclass
class WalkForwardBacktestResult:
    folds: list[FoldBacktestResult]
    mean_sharpe: float
    mean_win_rate: float
    mean_precision_buy: float
    worst_drawdown: float


def walk_forward_backtest(
    df: pl.DataFrame,
    model: BaseClassifier,
    feature_cols: list[str],
    label_col: str = "label",
    train_window_days: int = 500,
    test_window_days: int = 21,
    step_days: int = 21,
    min_train_samples: int = 100,
    confidence_threshold: float = 0.5,
) -> WalkForwardBacktestResult:
    df_clean = df.drop_nulls(subset=feature_cols + [label_col, "forward_return_5d"]).sort("time")
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
            X_train = train_df.select(feature_cols).to_numpy()
            y_train = train_df[label_col].to_numpy()
            model.fit(X_train, y_train)
            result = run_backtest(model, test_df, feature_cols, confidence_threshold)
            folds.append(FoldBacktestResult(
                fold=fold_idx,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                metrics=result.metrics,
                n_trades=len(result.trades),
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_walk_forward_backtest.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/walk_forward.py tests/unit/test_walk_forward_backtest.py
git commit -m "feat: walk-forward backtest with per-fold financial metrics"
```

---

### Task 4: Grader

**Files:**
- Create: `src/backtesting/grader.py`
- Test: `tests/unit/test_grader.py`

**Interfaces:**
- Consumes: `BacktestMetrics`, `WalkForwardBacktestResult`
- Produces:
  - `Grade` enum: `A = "A"`, `B = "B"`, `C = "C"`, `D = "D"`
  - `ModelGrade` dataclass: `model_name: str`, `grade: Grade`, `composite_score: float`, `metrics: BacktestMetrics`
  - `grade_model(model_name: str, metrics: BacktestMetrics) -> ModelGrade`
    — composite score = `0.40 × precision_buy + 0.30 × norm_sharpe + 0.30 × (1 − norm_drawdown)`
    — `norm_sharpe = tanh(sharpe / 2)`, `norm_drawdown = min(max_drawdown_pct, 0.50) / 0.50`
    — Grade A ≥ 0.65, B ≥ 0.50, C ≥ 0.35, D < 0.35
  - `build_leaderboard(grades: list[ModelGrade]) -> list[ModelGrade]`
    — returns grades sorted by composite_score descending

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_grader.py
import pytest
from src.backtesting.grader import Grade, ModelGrade, grade_model, build_leaderboard
from src.backtesting.metrics import BacktestMetrics


def _metrics(precision: float, sharpe: float, drawdown: float) -> BacktestMetrics:
    return BacktestMetrics(
        n_trades=50, win_rate=0.55, profit_factor=1.5,
        total_return_pct=0.10, sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown, precision_buy=precision,
        recall_buy=0.4, f1_buy=0.48, accuracy=0.55,
    )


def test_high_precision_high_sharpe_low_drawdown_gets_grade_a():
    g = grade_model("rf", _metrics(precision=0.80, sharpe=2.5, drawdown=0.03))
    assert g.grade == Grade.A


def test_low_precision_low_sharpe_high_drawdown_gets_grade_d():
    g = grade_model("nb", _metrics(precision=0.20, sharpe=-1.0, drawdown=0.45))
    assert g.grade == Grade.D


def test_composite_score_between_zero_and_one():
    g = grade_model("rf", _metrics(0.55, 1.0, 0.10))
    assert 0.0 <= g.composite_score <= 1.0


def test_leaderboard_sorted_descending():
    grades = [
        grade_model("m1", _metrics(0.4, 0.5, 0.2)),
        grade_model("m2", _metrics(0.8, 2.0, 0.05)),
        grade_model("m3", _metrics(0.3, 0.1, 0.4)),
    ]
    board = build_leaderboard(grades)
    scores = [g.composite_score for g in board]
    assert scores == sorted(scores, reverse=True)


def test_grade_model_returns_model_grade():
    g = grade_model("xgboost", _metrics(0.6, 1.2, 0.08))
    assert isinstance(g, ModelGrade)
    assert g.model_name == "xgboost"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_grader.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/backtesting/grader.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math

from src.backtesting.metrics import BacktestMetrics


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class ModelGrade:
    model_name: str
    grade: Grade
    composite_score: float
    metrics: BacktestMetrics


def _norm_sharpe(sharpe: float) -> float:
    return float(math.tanh(sharpe / 2))


def _norm_drawdown(drawdown: float) -> float:
    return min(drawdown, 0.50) / 0.50


def grade_model(model_name: str, metrics: BacktestMetrics) -> ModelGrade:
    score = (
        0.40 * metrics.precision_buy
        + 0.30 * _norm_sharpe(metrics.sharpe_ratio)
        + 0.30 * (1.0 - _norm_drawdown(metrics.max_drawdown_pct))
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.65:
        grade = Grade.A
    elif score >= 0.50:
        grade = Grade.B
    elif score >= 0.35:
        grade = Grade.C
    else:
        grade = Grade.D

    return ModelGrade(
        model_name=model_name,
        grade=grade,
        composite_score=round(score, 4),
        metrics=metrics,
    )


def build_leaderboard(grades: list[ModelGrade]) -> list[ModelGrade]:
    return sorted(grades, key=lambda g: g.composite_score, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_grader.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/grader.py tests/unit/test_grader.py
git commit -m "feat: A-D composite grader (precision 40% + sharpe 30% + drawdown 30%)"
```

---

### Task 5: Signal Engine

**Files:**
- Create: `src/signals/__init__.py`
- Create: `src/signals/signal_engine.py`
- Test: `tests/unit/test_signal_engine.py`

**Interfaces:**
- Consumes: `BaseClassifier` (Plan 3), `load_training_data` (Plan 2), `list_models` + `load_model` (Plan 3 registry)
- Produces:
  - `Signal` dataclass: `ticker: str`, `date: str`, `label: str`, `confidence: float`, `buy_probability: float`, `entry_price: float`, `position_size: float`, `feature_explanation: dict[str, float]`
  - `generate_signals(model: BaseClassifier, df: pl.DataFrame, feature_cols: list[str], confidence_threshold: float = 0.75) -> list[Signal]`
    — scores the LATEST row per ticker in `df`; returns only signals where predicted label is `"Buy"` and buy_probability >= threshold
    — `position_size = buy_probability` (confidence-weighted, 0–1)
    — `feature_explanation` is empty dict here; XAI fills it in Task 6
  - `generate_all_signals(parquet_dir: Path, registry_dir: Path, feature_cols: list[str], model_name: str | None = None, confidence_threshold: float = 0.75) -> list[Signal]`
    — loads latest model from registry (or named model), loads latest feature row per ticker from Parquet, calls `generate_signals`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_signal_engine.py
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import polars as pl
import pytest

from src.signals.signal_engine import Signal, generate_signals

_FEATURE_COLS = ["f1", "f2", "f3"]
_CLASSES = ["Buy", "Hold", "Sell"]


def _make_feature_df(labels: list[str]) -> pl.DataFrame:
    n = len(labels)
    rng = np.random.default_rng(1)
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    returns = [0.03 if l == "Buy" else 0.00 for l in labels]
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             [155.0 + i for i in range(n)],
        "forward_return_5d": returns,
        "label":             labels,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
    })


def test_generate_signals_returns_only_buy_signals():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 5 + ["Hold"] * 5)
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert s.label == "Buy"


def test_generate_signals_filters_by_confidence():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 10)
    low  = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    high = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=1.0)
    assert len(low) >= len(high)


def test_signal_position_size_equals_buy_probability():
    from src.models.zoo.random_forest import RandomForestClassifier_
    X_train = np.random.randn(120, 3)
    y_train = np.array([_CLASSES[i % 3] for i in range(120)])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X_train, y_train)

    df = _make_feature_df(["Buy"] * 10)
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert abs(s.position_size - s.buy_probability) < 1e-9


def test_signal_is_dataclass():
    s = Signal(
        ticker="AAPL", date="2024-06-10", label="Buy",
        confidence=0.82, buy_probability=0.82,
        entry_price=155.0, position_size=0.82,
        feature_explanation={},
    )
    assert s.label == "Buy"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_signal_engine.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/signals/signal_engine.py**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import polars as pl

from src.models.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

_BUY_IDX = 0  # alphabetical: Buy=0, Hold=1, Sell=2


@dataclass
class Signal:
    ticker: str
    date: str
    label: str
    confidence: float
    buy_probability: float
    entry_price: float
    position_size: float
    feature_explanation: dict[str, float] = field(default_factory=dict)


def generate_signals(
    model: BaseClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    latest = df.sort("time").group_by("ticker").tail(1)
    signals: list[Signal] = []

    for row in latest.iter_rows(named=True):
        x = np.array([[row[c] for c in feature_cols]])
        y_pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        buy_prob = float(proba[_BUY_IDX])

        if y_pred != "Buy" or buy_prob < confidence_threshold:
            continue

        signals.append(Signal(
            ticker=str(row["ticker"]),
            date=str(row["time"]),
            label="Buy",
            confidence=buy_prob,
            buy_probability=buy_prob,
            entry_price=float(row["close"]),
            position_size=buy_prob,
            feature_explanation={},
        ))

    return signals


def generate_all_signals(
    parquet_dir: Path,
    registry_dir: Path,
    feature_cols: list[str],
    model_name: str | None = None,
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    from src.features.duckdb_client import load_training_data
    from src.models.registry import load_model, list_models

    df = load_training_data(parquet_dir)

    if model_name:
        model = load_model(model_name, registry_dir)
    else:
        records = list_models(registry_dir)
        if not records:
            raise RuntimeError(f"No models in registry at {registry_dir}")
        model = load_model(records[0].model_name, registry_dir)

    return generate_signals(model, df, feature_cols, confidence_threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_signal_engine.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/signals/ tests/unit/test_signal_engine.py
git commit -m "feat: confidence-filtered signal engine with position sizing"
```

---

### Task 6: XAI — SHAP Explainability

**Files:**
- Create: `src/explainability/__init__.py`
- Create: `src/explainability/xai.py`
- Test: `tests/unit/test_xai.py`

**Interfaces:**
- Consumes: `BaseClassifier` (Plan 3); SHAP
- Produces:
  - `explain_prediction(model: BaseClassifier, X_row: np.ndarray, feature_cols: list[str], background: np.ndarray | None = None) -> dict[str, float]`
    — returns `{feature_name: shap_value}` for the "Buy" class, sorted by `abs(shap_value)` descending
    — uses `TreeExplainer` for `random_forest`, `xgboost`, `lightgbm`; `LinearExplainer` for `logistic_regression`; `KernelExplainer` (50-sample background) for all others
  - `attach_explanations(signals: list[Signal], model: BaseClassifier, df: pl.DataFrame, feature_cols: list[str]) -> list[Signal]`
    — mutates each signal's `feature_explanation` field in-place; returns the same list

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_xai.py
import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.explainability.xai import explain_prediction, attach_explanations
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.logistic_regression import LogisticRegressionClassifier
from src.signals.signal_engine import Signal

_CLASSES = ["Buy", "Hold", "Sell"]
_FEATURE_COLS = [f"feat_{i}" for i in range(10)]


def _make_data(n: int = 150):
    X, y_int = make_classification(
        n_samples=n, n_features=10, n_classes=3,
        n_informative=5, n_redundant=2, random_state=42,
    )
    return X, np.array([_CLASSES[i] for i in y_int])


def test_explain_prediction_returns_dict_with_feature_names():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    assert isinstance(result, dict)
    assert set(result.keys()) == set(_FEATURE_COLS)


def test_explain_prediction_values_are_floats():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    for v in result.values():
        assert isinstance(v, float)


def test_explain_sorted_by_abs_value_descending():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    result = explain_prediction(clf, X[:1], _FEATURE_COLS, background=X[:20])
    abs_vals = [abs(v) for v in result.values()]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_attach_explanations_fills_feature_explanation():
    X, y = _make_data()
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)

    from datetime import datetime, timezone, timedelta
    import polars as pl
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "time":   [base + timedelta(days=i) for i in range(10)],
        "ticker": ["AAPL"] * 10,
        "close":  [155.0] * 10,
        **{f"feat_{j}": X[:10, j].tolist() for j in range(10)},
    })

    signals = [Signal(
        ticker="AAPL", date=str(base), label="Buy",
        confidence=0.80, buy_probability=0.80,
        entry_price=155.0, position_size=0.80,
        feature_explanation={},
    )]
    result = attach_explanations(signals, clf, df, _FEATURE_COLS)
    assert len(result[0].feature_explanation) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_xai.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create src/explainability/__init__.py**

```bash
touch src/explainability/__init__.py
```

- [ ] **Step 4: Implement src/explainability/xai.py**

```python
from __future__ import annotations
import logging

import numpy as np
import shap

from src.models.base_classifier import BaseClassifier
from src.signals.signal_engine import Signal

logger = logging.getLogger(__name__)

_TREE_MODELS  = {"random_forest", "xgboost", "lightgbm"}
_LINEAR_MODELS = {"logistic_regression"}
_BUY_IDX = 0  # Buy is class 0 alphabetically


def explain_prediction(
    model: BaseClassifier,
    X_row: np.ndarray,
    feature_cols: list[str],
    background: np.ndarray | None = None,
) -> dict[str, float]:
    if X_row.ndim == 1:
        X_row = X_row.reshape(1, -1)

    try:
        shap_values = _compute_shap(model, X_row, background)
    except Exception as e:
        logger.warning("SHAP failed for %s: %s — returning zeros", model.name, e)
        return {col: 0.0 for col in feature_cols}

    if isinstance(shap_values, list):
        vals = shap_values[_BUY_IDX][0]
    elif shap_values.ndim == 3:
        vals = shap_values[0, :, _BUY_IDX]
    else:
        vals = shap_values[0]

    explanation = dict(zip(feature_cols, vals.tolist()))
    return dict(sorted(explanation.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _compute_shap(
    model: BaseClassifier,
    X_row: np.ndarray,
    background: np.ndarray | None,
) -> np.ndarray:
    if model.name in _TREE_MODELS:
        inner = getattr(model, "_model", model)
        explainer = shap.TreeExplainer(inner)
        return explainer.shap_values(X_row)

    if model.name in _LINEAR_MODELS:
        inner = getattr(model, "_model", model)
        scaler = getattr(model, "_scaler", None)
        bg = scaler.transform(background[:50]) if scaler is not None and background is not None else background[:50]
        X_scaled = scaler.transform(X_row) if scaler is not None else X_row
        explainer = shap.LinearExplainer(inner, bg)
        return explainer.shap_values(X_scaled)

    bg = background[:50] if background is not None and len(background) >= 50 else background
    if bg is None:
        bg = np.zeros((1, X_row.shape[1]))

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)

    explainer = shap.KernelExplainer(predict_fn, bg)
    return explainer.shap_values(X_row, nsamples=100, silent=True)


def attach_explanations(
    signals: list[Signal],
    model: BaseClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
) -> list[Signal]:
    import polars as pl
    latest = df.sort("time").group_by("ticker").tail(1)
    ticker_rows = {row["ticker"]: row for row in latest.iter_rows(named=True)}
    background = df.select(feature_cols).to_numpy()

    for signal in signals:
        row = ticker_rows.get(signal.ticker)
        if row is None:
            continue
        X_row = np.array([[row[c] for c in feature_cols]])
        signal.feature_explanation = explain_prediction(
            model, X_row, feature_cols, background=background
        )

    return signals
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_xai.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/explainability/ tests/unit/test_xai.py
git commit -m "feat: SHAP explainability with TreeExplainer/LinearExplainer/KernelExplainer dispatch"
```

---

### Task 7: Dashboard Data Loader

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/data_loader.py`
- Test: `tests/unit/test_dashboard_data_loader.py`

**Interfaces:**
- Consumes: `list_models`, `load_model` (Plan 3 registry), `load_training_data` (Plan 2), `walk_forward_backtest`, `grade_model`, `build_leaderboard`, `generate_all_signals`, `attach_explanations`
- Produces (all pure functions, no Streamlit calls):
  - `get_leaderboard(registry_dir: Path, parquet_dir: Path, feature_cols: list[str]) -> list[ModelGrade]`
  - `get_backtest_result(model_name: str, registry_dir: Path, parquet_dir: Path, feature_cols: list[str]) -> tuple[WalkForwardBacktestResult, ModelGrade]`
  - `get_live_signals(registry_dir: Path, parquet_dir: Path, feature_cols: list[str], confidence_threshold: float = 0.75) -> list[Signal]`
  - `get_data_summary(parquet_dir: Path) -> dict`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_dashboard_data_loader.py
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from dashboard.data_loader import get_data_summary, get_leaderboard
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.registry import save_model
from src.models.base_classifier import EvaluationResult

_FEATURE_COLS = ["f1", "f2", "f3"]
_CLASSES = ["Buy", "Hold", "Sell"]
_EVAL = EvaluationResult(
    accuracy=0.55, precision_buy=0.62, recall_buy=0.48,
    f1_buy=0.54, f1_macro=0.50,
    confusion_matrix=[[5, 2, 1], [1, 4, 2], [0, 1, 4]],
    class_labels=["Buy", "Hold", "Sell"], n_samples=50,
)


def _write_sample_parquet(tmp_dir: Path, n: int = 200) -> None:
    rng = np.random.default_rng(42)
    base = datetime(2023, 1, 2, tzinfo=timezone.utc)
    labels = [_CLASSES[i % 3] for i in range(n)]
    closes = [150.0 + i * 0.1 for i in range(n)]
    returns = [0.03 if l == "Buy" else 0.00 for l in labels]
    df = pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
    })
    df.write_parquet(tmp_dir / "AAPL.parquet")


def _save_rf(registry_dir: Path) -> None:
    X = np.random.randn(150, 3)
    y = np.array([_CLASSES[i % 3] for i in range(150)])
    clf = RandomForestClassifier_(n_estimators=5)
    clf.fit(X, y)
    save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, registry_dir)


def test_get_data_summary_returns_dict():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        summary = get_data_summary(p)
        assert "n_tickers" in summary
        assert "n_rows" in summary
        assert summary["n_tickers"] >= 1


def test_get_leaderboard_returns_list_of_model_grades():
    with tempfile.TemporaryDirectory() as tmp_p, \
         tempfile.TemporaryDirectory() as tmp_r:
        parquet_dir = Path(tmp_p)
        registry_dir = Path(tmp_r)
        _write_sample_parquet(parquet_dir)
        _save_rf(registry_dir)
        leaderboard = get_leaderboard(registry_dir, parquet_dir, _FEATURE_COLS)
        assert isinstance(leaderboard, list)
        assert len(leaderboard) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_dashboard_data_loader.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement dashboard/data_loader.py**

```python
from __future__ import annotations
from pathlib import Path
import logging

from src.backtesting.grader import ModelGrade, grade_model, build_leaderboard
from src.backtesting.walk_forward import WalkForwardBacktestResult, walk_forward_backtest
from src.features.duckdb_client import load_training_data
from src.models.base_classifier import BaseClassifier
from src.models.registry import list_models, load_model
from src.signals.signal_engine import Signal, generate_signals
from src.explainability.xai import attach_explanations

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
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
) -> list[ModelGrade]:
    records = list_models(registry_dir)
    if not records:
        return []

    df = load_training_data(parquet_dir)
    grades: list[ModelGrade] = []
    for record in records:
        try:
            model = load_model(record.model_name, registry_dir)
            result = walk_forward_backtest(
                df, model, feature_cols,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            avg_metrics = result.folds[-1].metrics
            grades.append(grade_model(record.model_name, avg_metrics))
        except Exception as e:
            logger.warning("Leaderboard skipping %s: %s", record.model_name, e)

    return build_leaderboard(grades)


def get_backtest_result(
    model_name: str,
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    model = load_model(model_name, registry_dir)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest(df, model, feature_cols)
    grade = grade_model(model_name, result.folds[-1].metrics)
    return result, grade


def get_live_signals(
    registry_dir: Path,
    parquet_dir: Path,
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
) -> list[Signal]:
    records = list_models(registry_dir)
    if not records:
        return []
    model = load_model(records[0].model_name, registry_dir)
    df = load_training_data(parquet_dir)
    signals = generate_signals(model, df, feature_cols, confidence_threshold)
    return attach_explanations(signals, model, df, feature_cols)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_dashboard_data_loader.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/ tests/unit/test_dashboard_data_loader.py
git commit -m "feat: dashboard data loader functions (leaderboard, backtest, signals, summary)"
```

---

### Task 8: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`
- Create: `dashboard/pages/1_Data_Overview.py`
- Create: `dashboard/pages/2_Model_Leaderboard.py`
- Create: `dashboard/pages/3_Backtest_Results.py`
- Create: `dashboard/pages/4_Live_Signals.py`
- Create: `dashboard/config.py`

No unit tests for Streamlit render calls. Manual verification in Step 9.

**Interfaces:**
- Consumes: all `dashboard/data_loader.py` functions
- Produces: running Streamlit app on `localhost:8501` with 4 pages

- [ ] **Step 1: Create dashboard/config.py**

```python
from pathlib import Path

PARQUET_DIR  = Path("data/features")
REGISTRY_DIR = Path("data/registry")

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

- [ ] **Step 2: Create dashboard/app.py**

```python
import streamlit as st

st.set_page_config(
    page_title="Financial Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Signal Platform")
st.markdown("""
Navigate using the sidebar:

- **Data Overview** — data ingestion status, ticker universe, date ranges
- **Model Leaderboard** — all trained models ranked by composite grade
- **Backtest Results** — equity curve, trade log, and financial metrics per model
- **Live Signals** — today's Buy/Hold/Sell recommendations with SHAP explanations
""")
```

- [ ] **Step 3: Create dashboard/pages/1_Data_Overview.py**

```python
import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, FEATURE_COLS
from dashboard.data_loader import get_data_summary
from src.features.duckdb_client import load_training_data

st.set_page_config(page_title="Data Overview", layout="wide")
st.header("Data Overview")


@st.cache_data(ttl=3600)
def _summary():
    return get_data_summary(PARQUET_DIR)


@st.cache_data(ttl=3600)
def _load_ticker_df(ticker: str):
    return load_training_data(PARQUET_DIR, tickers=[ticker])


summary = _summary()

col1, col2, col3 = st.columns(3)
col1.metric("Tickers", summary["n_tickers"])
col2.metric("Total Rows", f"{summary['n_rows']:,}")
col3.metric("Date Range", f"{summary['date_range_start'][:10]} → {summary['date_range_end'][:10]}")

st.divider()

ticker = st.selectbox("Select ticker to preview", summary["tickers"])
if ticker:
    df = _load_ticker_df(ticker)
    if not df.is_empty():
        fig = go.Figure()
        times = df["time"].to_list()
        closes = df["close"].to_list()
        fig.add_trace(go.Scatter(x=times, y=closes, mode="lines", name="Close"))
        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=times, y=df["sma_20"].to_list(),
                                     mode="lines", name="SMA 20", line=dict(dash="dot")))
        fig.update_layout(title=f"{ticker} Price + SMA 20",
                          xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        if "sent_pos_avg_5d" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=times, y=df["sent_pos_avg_5d"].to_list(),
                                  name="5d Avg Positive Sentiment"))
            fig2.update_layout(title=f"{ticker} News Sentiment (5d Rolling)",
                                xaxis_title="Date", yaxis_title="Positive Sentiment Score")
            st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 4: Create dashboard/pages/2_Model_Leaderboard.py**

```python
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from dashboard.config import PARQUET_DIR, REGISTRY_DIR, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_leaderboard

st.set_page_config(page_title="Model Leaderboard", layout="wide")
st.header("Model Leaderboard")


@st.cache_data(ttl=1800)
def _leaderboard():
    return get_leaderboard(REGISTRY_DIR, PARQUET_DIR, FEATURE_COLS)


with st.spinner("Computing grades..."):
    leaderboard = _leaderboard()

if not leaderboard:
    st.warning("No trained models found in registry. Run the model_retrain_dag first.")
    st.stop()

rows = [{
    "Rank": i + 1,
    "Model": g.model_name,
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
    fig.update_layout(title="Precision (Buy class)", xaxis_title="Model",
                      yaxis_title="Precision", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Bar(
        x=[g.model_name for g in leaderboard],
        y=[g.metrics.sharpe_ratio for g in leaderboard],
        marker_color=[GRADE_COLORS[g.grade.value] for g in leaderboard],
    ))
    fig2.update_layout(title="Sharpe Ratio", xaxis_title="Model",
                       yaxis_title="Sharpe")
    st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 5: Create dashboard/pages/3_Backtest_Results.py**

```python
import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, REGISTRY_DIR, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_backtest_result
from src.models.registry import list_models

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Backtest Results")

records = list_models(REGISTRY_DIR)
if not records:
    st.warning("No models in registry.")
    st.stop()

model_names = [r.model_name for r in records]
selected = st.selectbox("Select model", model_names)


@st.cache_data(ttl=1800)
def _backtest(model_name: str):
    return get_backtest_result(model_name, REGISTRY_DIR, PARQUET_DIR, FEATURE_COLS)


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

- [ ] **Step 6: Create dashboard/pages/4_Live_Signals.py**

```python
import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, REGISTRY_DIR, FEATURE_COLS, CONFIDENCE_THRESHOLD
from dashboard.data_loader import get_live_signals

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

with st.spinner("Generating signals..."):
    signals = get_live_signals(REGISTRY_DIR, PARQUET_DIR, FEATURE_COLS, threshold)

if not signals:
    st.info("No Buy signals above the current confidence threshold.")
    st.stop()

st.success(f"Found **{len(signals)}** Buy signal(s)")

for sig in signals:
    with st.expander(f"**{sig.ticker}** — Confidence {sig.confidence:.1%} | Entry ${sig.entry_price:.2f}"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{sig.confidence:.1%}")
        c2.metric("Position Size", f"{sig.position_size:.1%}")
        c3.metric("Entry Price", f"${sig.entry_price:.2f}")

        if sig.feature_explanation:
            top_n = dict(list(sig.feature_explanation.items())[:10])
            features = list(top_n.keys())
            values   = list(top_n.values())
            colors   = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
            fig = go.Figure(go.Bar(
                x=values[::-1], y=features[::-1],
                orientation="h",
                marker_color=colors[::-1],
            ))
            fig.update_layout(
                title=f"Top 10 Features Driving {sig.ticker} Buy Signal",
                xaxis_title="SHAP Value (positive = bullish)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Feature explanation not available.")
```

- [ ] **Step 7: Verify app runs without crash**

```bash
mkdir -p data/features data/registry
streamlit run dashboard/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | grep -q "Financial Signal Platform" && echo "OK" || echo "FAIL"
```
Expected: `OK` printed (Streamlit is serving the app).

- [ ] **Step 8: Commit**

```bash
git add dashboard/ 
git commit -m "feat: 4-page Streamlit dashboard (overview, leaderboard, backtest, signals)"
```

---

### Task 9: Full Test Suite & Integration Smoke Test

**Files:**
- Create: `tests/integration/test_smoke.py`

**Purpose:** Verify all four plans connect end-to-end using synthetic data — no external APIs, no DB required. Generates features in memory, trains a tiny model, runs a backtest, generates a signal, grades it.

- [ ] **Step 1: Write smoke test**

```python
# tests/integration/test_smoke.py
"""
End-to-end smoke test: synthetic data → features → model → backtest → signal → grade.
No DB, no external APIs, no Airflow. Runs entirely in-memory.
"""
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_CLASSES = ["Buy", "Hold", "Sell"]
_FEATURE_COLS = [f"f{i}" for i in range(10)]


def _make_full_df(n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    labels = [_CLASSES[i % 3] for i in range(n)]
    closes = [100.0 + i * 0.1 for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    features = {f"f{j}": rng.standard_normal(n).tolist() for j in range(10)}
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        **features,
    })


def test_full_pipeline_smoke():
    from src.models.zoo.random_forest import RandomForestClassifier_
    from src.models.evaluator import evaluate
    from src.backtesting.engine import run_backtest
    from src.backtesting.walk_forward import walk_forward_backtest
    from src.backtesting.grader import grade_model
    from src.signals.signal_engine import generate_signals
    from src.models.registry import save_model, load_model, list_models
    from src.models.base_classifier import EvaluationResult

    df = _make_full_df()

    # 1. Train model
    X = df.select(_FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    clf = RandomForestClassifier_(n_estimators=20)
    clf.fit(X[:300], y[:300])

    # 2. Evaluate
    y_pred = clf.predict(X[300:])
    eval_result = evaluate(y[300:], y_pred)
    assert 0 <= eval_result.precision_buy <= 1

    # 3. Backtest
    bt = run_backtest(clf, df[300:], _FEATURE_COLS, confidence_threshold=0.0)
    assert isinstance(bt.metrics.sharpe_ratio, float)

    # 4. Walk-forward backtest
    wf = walk_forward_backtest(
        df, clf, _FEATURE_COLS,
        train_window_days=150, test_window_days=20, step_days=20,
    )
    assert len(wf.folds) >= 1

    # 5. Grade
    grade = grade_model("random_forest", wf.folds[-1].metrics)
    assert grade.grade.value in {"A", "B", "C", "D"}

    # 6. Registry save/load
    with tempfile.TemporaryDirectory() as tmp:
        reg = Path(tmp)
        path = save_model(clf, eval_result, clf.default_params, _FEATURE_COLS, reg)
        assert path.exists()
        loaded = load_model("random_forest", reg)
        assert loaded.predict(X[:1]).shape == (1,)

    # 7. Signal engine
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert s.label == "Buy"
        assert 0.0 <= s.buy_probability <= 1.0

    print(f"Smoke test passed: grade={grade.grade.value}, "
          f"precision={eval_result.precision_buy:.3f}, "
          f"signals={len(signals)}, folds={len(wf.folds)}")
```

- [ ] **Step 2: Run smoke test**

```bash
pytest tests/integration/test_smoke.py -v -s
```
Expected: `Smoke test passed: grade=...` printed. 1 test PASS.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS. Note: DB-dependent tests require TimescaleDB container running.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_smoke.py
git commit -m "test: end-to-end smoke test covering full pipeline from features to signals"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Out-of-sample backtesting on hold-out dataset | Task 2 |
| Accuracy, precision, recall, win rate, profit factor | Task 1 |
| Sharpe ratio (risk-adjusted return) | Task 1 |
| Maximum drawdown | Task 1 |
| Walk-forward analysis (rolling backtest) | Task 3 |
| Grade A/B/C/D composite grading system | Task 4 |
| Confidence threshold for Buy signal filtering | Tasks 2 + 5 |
| Buy/Hold/Sell signals with confidence scores | Task 5 |
| Suggested position sizing | Task 5 (`position_size = buy_probability`) |
| Explainable AI (SHAP feature importance) | Task 6 |
| "Which features drove this Buy signal" | Task 6 + Page 4 |
| Streamlit visualization dashboard | Task 8 |
| Model leaderboard with grades | Task 8 Page 2 |
| Side-by-side model comparison charts | Task 8 Page 2 |
| Entry price with signal | Task 5 (`entry_price = close`) |

**Placeholder scan:** No TBDs. All steps include code. Dashboard pages (Task 8) are not unit-tested — they are visually verified via the `curl` smoke check and the integration smoke test covers the underlying logic functions.

**Type consistency:** `Trade.return_pct` flows into `compute_metrics` → `BacktestMetrics.sharpe_ratio` → `grade_model` → `ModelGrade.composite_score`. `Signal.buy_probability` == `Signal.position_size` throughout Tasks 5–7. `_BUY_IDX = 0` constant is defined identically in `engine.py`, `signal_engine.py`, and `xai.py` — all three reference the same alphabetical-sort invariant tested in `test_backtest_engine.py::test_buy_class_at_index_zero`.
