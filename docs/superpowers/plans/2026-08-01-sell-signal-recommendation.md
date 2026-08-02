# Sell Signal Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give strategies the ability to actually emit a Sell signal (currently none of the 32 ever do, despite the `Signal` enum and training labels both being 3-class), extend the existing per-ticker rating aggregation with a sell-side score, and add a new dashboard page where a user picks a ticker they own (plus optional buy price/date) and gets a sell/hold recommendation with P&L context.

**Architecture:** A shared `buy_sell_hold_signal()` helper in `src/strategies/base.py` replaces the duplicated Buy-only threshold logic in 10 classifier-based strategies; `linear.py` (regression, not a classifier) gets a symmetric standalone fix. `dashboard/data_loader.py`'s existing `get_combined_ratings()` gains a parallel sell-side computation using the exact same per-ticker rows and weights it already assembles. A new page reuses that function directly. Real backtest/leaderboard/signal data gets regenerated for both markets since `predict()` changes affect walk-forward backtests too, not just live signals.

**Tech Stack:** Python, scikit-learn/xgboost/catboost, Streamlit, Polars, pytest.

## Global Constraints

- Only these 11 files under `src/strategies/statistical/` change: `random_forest.py`, `logistic.py`, `gaussian_nb_strategy.py`, `lda_strategy.py`, `knn_classifier.py`, `mlp_classifier.py`, `gradient_boosting.py`, `extra_trees.py`, `catboost_strategy.py`, `xgboost_strategy.py`, `linear.py`. The 20 rule-based strategies (`src/strategies/rule_based/`) and `isolation_forest_strategy.py` are explicitly out of scope — no natural sell condition, stay Buy/Hold-only.
- `adaboost.py`, `hist_gradient_boosting.py`, `lightgbm_strategy.py`, `ridge.py`, `stacking_ensemble.py`, `svm_strategy.py` under `src/strategies/statistical/` are **not** registered in `strategies.yaml` (dormant files, confirmed via `list_strategies()` returning exactly the 32 active names) — do not touch them.
- `confidence` continues to mean "confidence in the emitted signal" — for Hold rows specifically, it continues to report the Buy-class probability (or, for `linear.py`, the existing unchanged return-based formula) exactly as today. This preserves every existing cached Hold record's established meaning.
- `src/models/zoo/*` and `train_models.py`/`train_new_models.py` are a **separate, unrelated** code path (the registry-trained core models: random_forest/xgboost/lightgbm) — not touched by this plan. `scripts/precompute_dashboard.py`'s `step_backtests`/`step_signals` fit strategies fresh from `src.strategies.registry` on every run; they never load from the model registry. Only `scripts/precompute_dashboard.py --market <market>` needs re-running per market, not `train_models.py`.
- `config.markets.MARKETS` remains the sole source of truth for market selection in the new page — no hardcoded market list.
- Every `@st.cache_data`-wrapped function in the new page must take `market` as an explicit argument (same Streamlit cache-key rule as every other page).
- Real committed data used directly in tests where noted (no mocking of `config.markets` or real cache file shapes) — matches this repo's established convention.

---

### Task 1: Shared `buy_sell_hold_signal()` helper in `src/strategies/base.py`

**Files:**
- Modify: `src/strategies/base.py`
- Test: `tests/unit/test_strategy_signal_helper.py` (new file)

**Interfaces:**
- Produces: `buy_sell_hold_signal(proba, classes: list[str], threshold: float = 0.6) -> tuple[pd.Series, pd.Series]` — returns `(signal, confidence)`. Consumed by Task 2's 10 classifier strategies.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_strategy_signal_helper.py`:

```python
import numpy as np
import pandas as pd

from src.strategies.base import buy_sell_hold_signal


def test_high_buy_probability_emits_buy():
    proba = np.array([[0.7, 0.1, 0.2]])  # columns match classes order below
    signal, confidence = buy_sell_hold_signal(proba, classes=["Buy", "Hold", "Sell"])
    assert list(signal) == ["Buy"]
    assert confidence.iloc[0] == 0.7


def test_high_sell_probability_emits_sell():
    proba = np.array([[0.1, 0.1, 0.8]])
    signal, confidence = buy_sell_hold_signal(proba, classes=["Buy", "Hold", "Sell"])
    assert list(signal) == ["Sell"]
    assert confidence.iloc[0] == 0.8


def test_ambiguous_probability_emits_hold_with_buy_confidence():
    proba = np.array([[0.4, 0.3, 0.3]])
    signal, confidence = buy_sell_hold_signal(proba, classes=["Buy", "Hold", "Sell"])
    assert list(signal) == ["Hold"]
    assert confidence.iloc[0] == 0.4  # Hold rows report Buy-class probability


def test_missing_sell_class_never_emits_sell():
    proba = np.array([[0.7, 0.3]])  # only Buy/Hold in this row's classes
    signal, confidence = buy_sell_hold_signal(proba, classes=["Buy", "Hold"])
    assert list(signal) == ["Buy"]


def test_missing_buy_class_can_still_emit_sell():
    proba = np.array([[0.2, 0.8]])  # only Hold/Sell in this row's classes
    signal, confidence = buy_sell_hold_signal(proba, classes=["Hold", "Sell"])
    assert list(signal) == ["Sell"]
    assert confidence.iloc[0] == 0.8


def test_multiple_rows():
    proba = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.1, 0.8],
        [0.4, 0.3, 0.3],
    ])
    signal, confidence = buy_sell_hold_signal(proba, classes=["Buy", "Hold", "Sell"])
    assert list(signal) == ["Buy", "Sell", "Hold"]
    assert isinstance(signal, pd.Series)
    assert isinstance(confidence, pd.Series)
    assert len(signal) == 3
    assert len(confidence) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_strategy_signal_helper.py -v`
Expected: FAIL with `ImportError: cannot import name 'buy_sell_hold_signal'`.

- [ ] **Step 3: Add the helper to `src/strategies/base.py`**

Append to the end of `src/strategies/base.py` (after the `Strategy` class):

```python


def buy_sell_hold_signal(
    proba, classes: list[str], threshold: float = 0.6,
) -> tuple[pd.Series, pd.Series]:
    """Classify each row's class probabilities into Buy/Sell/Hold: the higher
    of the Buy/Sell class probabilities wins if it clears `threshold`,
    otherwise Hold. `confidence` reports the probability of whichever signal
    was emitted — the Buy-class probability for Hold rows, matching this
    platform's existing convention. Returns (signal, confidence)."""
    buy_idx = classes.index("Buy") if "Buy" in classes else None
    sell_idx = classes.index("Sell") if "Sell" in classes else None

    signal: list[str] = []
    confidence: list[float] = []
    for row in proba:
        b = float(row[buy_idx]) if buy_idx is not None else 0.0
        s = float(row[sell_idx]) if sell_idx is not None else 0.0
        if b >= threshold and b >= s:
            signal.append("Buy")
            confidence.append(b)
        elif s >= threshold and s > b:
            signal.append("Sell")
            confidence.append(s)
        else:
            signal.append("Hold")
            confidence.append(b)
    return pd.Series(signal), pd.Series(confidence)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_strategy_signal_helper.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/base.py tests/unit/test_strategy_signal_helper.py
git commit -m "feat: add buy_sell_hold_signal() helper for 3-class strategy signals"
```

---

### Task 2: Wire 10 classifier strategies to emit real Sell signals

**Files:**
- Modify: `src/strategies/statistical/random_forest.py`
- Modify: `src/strategies/statistical/logistic.py`
- Modify: `src/strategies/statistical/gaussian_nb_strategy.py`
- Modify: `src/strategies/statistical/lda_strategy.py`
- Modify: `src/strategies/statistical/knn_classifier.py`
- Modify: `src/strategies/statistical/mlp_classifier.py`
- Modify: `src/strategies/statistical/gradient_boosting.py`
- Modify: `src/strategies/statistical/extra_trees.py`
- Modify: `src/strategies/statistical/catboost_strategy.py`
- Modify: `src/strategies/statistical/xgboost_strategy.py`
- Modify (widen existing test): `tests/unit/test_random_forest_strategy.py`, `tests/unit/test_gaussian_nb_strategy.py`, `tests/unit/test_lda_strategy.py`, `tests/unit/test_knn_classifier.py`, `tests/unit/test_mlp_classifier.py`, `tests/unit/test_hist_gradient_boosting.py`, `tests/unit/test_extra_trees_strategy.py`, `tests/unit/test_catboost_strategy.py`, `tests/unit/test_xgboost_strategy.py`
- Test (new file, `logistic.py` has none today): `tests/unit/test_logistic_strategy.py`

**Interfaces:**
- Consumes: `src.strategies.base.buy_sell_hold_signal(proba, classes, threshold=0.6) -> tuple[pd.Series, pd.Series]` (Task 1).

All 10 files share the identical transformation. For each file, the `predict()` method's tail — from the `classes = ...` line (or `proba = self._model.predict_proba(X)` for `catboost_strategy.py`, which doesn't have a `classes = list(...)` line) through the final `return PredictionResult(...)` — is replaced.

- [ ] **Step 1: Widen the existing per-strategy tests to expect Sell as a valid signal (RED)**

In each of these 9 files, replace:

```python
def test_signals_are_buy_or_hold(sample_df):
```

through its body ending:

```python
    assert set(result.signal.unique()).issubset({"Buy", "Hold"})
```

with:

```python
def test_signals_are_buy_hold_or_sell(sample_df):
    s = <StrategyClassName>()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold", "Sell"})
```

Use the correct `<StrategyClassName>` per file (matches what that test file already imports and instantiates elsewhere): `RandomForestStrategy` (`test_random_forest_strategy.py`), `GaussianNBStrategy` (`test_gaussian_nb_strategy.py`), `LDAStrategy` (`test_lda_strategy.py`), `KNNStrategy` (`test_knn_classifier.py`), `MLPStrategy` (`test_mlp_classifier.py`), `GradientBoostingStrategy` (`test_hist_gradient_boosting.py`), `ExtraTreesStrategy` (`test_extra_trees_strategy.py`), `CatBoostStrategy` (`test_catboost_strategy.py`), `XGBoostStrategy` (`test_xgboost_strategy.py`).

Create `tests/unit/test_logistic_strategy.py` (this strategy has no dedicated test file today — mirror the exact fixture shape every other file in this list uses, e.g. `tests/unit/test_random_forest_strategy.py`):

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.logistic import LogisticStrategy

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
    s = LogisticStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_hold_or_sell(sample_df):
    s = LogisticStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold", "Sell"})


def test_confidence_in_unit_range(sample_df):
    s = LogisticStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_predict_before_fit_raises():
    s = LogisticStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0], "label": ["Buy"]}))


def test_data_source_is_features():
    assert LogisticStrategy.data_source == "features"
```

- [ ] **Step 2: Run the touched tests to verify they fail for the right reason**

Run: `python -m pytest tests/unit/test_random_forest_strategy.py tests/unit/test_gaussian_nb_strategy.py tests/unit/test_lda_strategy.py tests/unit/test_knn_classifier.py tests/unit/test_mlp_classifier.py tests/unit/test_hist_gradient_boosting.py tests/unit/test_extra_trees_strategy.py tests/unit/test_catboost_strategy.py tests/unit/test_xgboost_strategy.py tests/unit/test_logistic_strategy.py -v`
Expected: the widened `test_signals_are_buy_hold_or_sell` tests PASS already (issubset of a wider set than before still holds against current Buy/Hold-only output — this is expected, not a failure signal for this particular assertion). The NEW `tests/unit/test_logistic_strategy.py` file's tests should all PASS too once created, since `LogisticStrategy` isn't broken, just incomplete. There is no RED step that fails here — the point of this step is confirming the rename/widen didn't break anything, before Step 3 makes the underlying behavior real. Proceed to Step 3 regardless.

- [ ] **Step 3: Rewrite each of the 10 files' `predict()` tail**

For `random_forest.py`, `logistic.py`, `knn_classifier.py`, `mlp_classifier.py`, `extra_trees.py` — add the import and replace the tail identically:

Change the import line:
```python
from src.strategies.base import Strategy, PredictionResult
```
to:
```python
from src.strategies.base import Strategy, PredictionResult, buy_sell_hold_signal
```

Change the `predict()` method's tail from:
```python
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```
to:
```python
        classes = list(self._model.classes_)
        signal, confidence = buy_sell_hold_signal(proba, classes)
        return PredictionResult(confidence=confidence, signal=signal)
```

For `gradient_boosting.py` — same tail replacement as above (its `classes = list(self._model.classes_)` line is identical in shape), and the same import-line change.

For `gaussian_nb_strategy.py`, `lda_strategy.py`, `xgboost_strategy.py` — same import-line change, but the classes source is `self._le.classes_` not `self._model.classes_`. Change the tail from:
```python
        classes = list(self._le.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```
to:
```python
        classes = list(self._le.classes_)
        signal, confidence = buy_sell_hold_signal(proba, classes)
        return PredictionResult(confidence=confidence, signal=signal)
```

For `catboost_strategy.py` — same import-line change. This file has no `classes = list(...)` line (uses `self._classes` directly, already a list). Change the tail from:
```python
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
to:
```python
        proba = self._model.predict_proba(X)
        signal, confidence = buy_sell_hold_signal(proba, self._classes)
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run the touched tests to verify they pass**

Run: `python -m pytest tests/unit/test_random_forest_strategy.py tests/unit/test_gaussian_nb_strategy.py tests/unit/test_lda_strategy.py tests/unit/test_knn_classifier.py tests/unit/test_mlp_classifier.py tests/unit/test_hist_gradient_boosting.py tests/unit/test_extra_trees_strategy.py tests/unit/test_catboost_strategy.py tests/unit/test_xgboost_strategy.py tests/unit/test_logistic_strategy.py tests/unit/test_strategy_signal_helper.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/random_forest.py src/strategies/statistical/logistic.py src/strategies/statistical/gaussian_nb_strategy.py src/strategies/statistical/lda_strategy.py src/strategies/statistical/knn_classifier.py src/strategies/statistical/mlp_classifier.py src/strategies/statistical/gradient_boosting.py src/strategies/statistical/extra_trees.py src/strategies/statistical/catboost_strategy.py src/strategies/statistical/xgboost_strategy.py tests/unit/test_random_forest_strategy.py tests/unit/test_gaussian_nb_strategy.py tests/unit/test_lda_strategy.py tests/unit/test_knn_classifier.py tests/unit/test_mlp_classifier.py tests/unit/test_hist_gradient_boosting.py tests/unit/test_extra_trees_strategy.py tests/unit/test_catboost_strategy.py tests/unit/test_xgboost_strategy.py tests/unit/test_logistic_strategy.py
git commit -m "feat: wire 10 classifier strategies to emit real Sell signals via buy_sell_hold_signal()"
```

---

### Task 3: Fix `linear.py` (regression strategy) to emit Sell symmetrically

**Files:**
- Modify: `src/strategies/statistical/linear.py`
- Test: `tests/unit/test_linear_strategy.py` (new file — no dedicated test exists today)

**Interfaces:**
- Produces: `LinearStrategy(buy_threshold: float = 0.005, sell_threshold: float | None = None)` — `sell_threshold` defaults to `-buy_threshold` when not given.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_linear_strategy.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.strategies.statistical.linear import LinearStrategy

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
    data["forward_return_5d"] = rng.standard_normal(n) * 0.05  # wide spread so both thresholds are crossed
    return pd.DataFrame(data)


def test_fit_predict_correct_length(sample_df):
    s = LinearStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert len(result.signal) == len(sample_df)
    assert len(result.confidence) == len(sample_df)


def test_signals_are_buy_hold_or_sell(sample_df):
    s = LinearStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert set(result.signal.unique()).issubset({"Buy", "Hold", "Sell"})


def test_confidence_in_unit_range(sample_df):
    s = LinearStrategy()
    s.fit(sample_df)
    result = s.predict(sample_df)
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_sell_threshold_defaults_to_negative_buy_threshold():
    s = LinearStrategy(buy_threshold=0.02)
    assert s.sell_threshold == -0.02


def test_sell_threshold_explicit_override():
    s = LinearStrategy(buy_threshold=0.02, sell_threshold=-0.05)
    assert s.sell_threshold == -0.05


def test_predict_before_fit_raises():
    s = LinearStrategy()
    with pytest.raises(ValueError, match="fit"):
        s.predict(pd.DataFrame({"sma_10": [1.0]}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_linear_strategy.py -v`
Expected: FAIL — `LinearStrategy` doesn't accept `sell_threshold` yet, and can't emit `"Sell"`.

- [ ] **Step 3: Rewrite `src/strategies/statistical/linear.py`**

Replace the full file content with:

```python
# src/strategies/statistical/linear.py
from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class LinearStrategy(Strategy):
    data_source = "features"

    def __init__(self, buy_threshold: float = 0.005, sell_threshold: float | None = None) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold if sell_threshold is not None else -buy_threshold
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
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        pred_return = self._model.predict(X)

        def _label(r: float) -> str:
            if r >= self.buy_threshold:
                return "Buy"
            if r <= self.sell_threshold:
                return "Sell"
            return "Hold"

        signal = pd.Series([_label(r) for r in pred_return])
        # Map predicted return [-10%, +10%] → confidence [0, 1], unchanged for
        # Buy/Hold rows; for Sell rows, mirror the same mapping around zero so
        # a more negative predicted return means higher sell-confidence.
        confidence = pd.Series([
            ((-r) + 0.10) / 0.20 if sig == "Sell" else (r + 0.10) / 0.20
            for r, sig in zip(pred_return, signal)
        ]).clip(0, 1)
        return PredictionResult(confidence=confidence, signal=signal)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_linear_strategy.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/statistical/linear.py tests/unit/test_linear_strategy.py
git commit -m "feat: make linear_regression strategy emit Sell symmetrically around zero return"
```

---

### Task 4: Extend `get_combined_ratings()` with sell-side scoring

**Files:**
- Modify: `dashboard/data_loader.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Produces: `get_combined_ratings(market)`'s summary rows gain `sell_rating: float`, `net_rating: float`, `n_sell: int`; detail rows gain `sell_contribution: float`. All existing fields (`overall_rating`, `n_buy`, `n_strategies`, `date`, `entry_price`, and every existing detail field) are unchanged in meaning and value — purely additive. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_get_combined_ratings_computes_sell_and_net_rating(tmp_path, monkeypatch):
    import json
    from dashboard import data_loader

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "leaderboard.json").write_text(json.dumps({
        "grades": [
            {"model_name": "strat_a", "composite_score": 1.0},
            {"model_name": "strat_b", "composite_score": 1.0},
        ]
    }))
    (cache_dir / "signals.json").write_text(json.dumps({
        "signals": [
            {"ticker": "AAA", "date": "2026-01-01", "signal": "Buy", "confidence": 0.8,
             "entry_price": 10.0, "position_size": 0.8, "strategy": "strat_a"},
            {"ticker": "AAA", "date": "2026-01-01", "signal": "Sell", "confidence": 0.9,
             "entry_price": 10.0, "position_size": 0.9, "strategy": "strat_b"},
        ]
    }))
    monkeypatch.setattr(data_loader, "get_cache_dir", lambda market: cache_dir)

    summary_rows, detail_by_ticker = data_loader.get_combined_ratings(market="us")

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["ticker"] == "AAA"
    assert row["overall_rating"] == pytest.approx(40.0)   # 100 * (1.0*0.8) / 2.0
    assert row["sell_rating"] == pytest.approx(45.0)      # 100 * (1.0*0.9) / 2.0
    assert row["net_rating"] == pytest.approx(5.0)        # 100 * (0.9 - 0.8) / 2.0
    assert row["n_sell"] == 1

    detail = detail_by_ticker["AAA"]
    sell_row = next(d for d in detail if d["strategy"] == "strat_b")
    assert sell_row["sell_contribution"] == pytest.approx(0.9)
    assert sell_row["contribution"] == pytest.approx(0.0)
    buy_row = next(d for d in detail if d["strategy"] == "strat_a")
    assert buy_row["contribution"] == pytest.approx(0.8)
    assert buy_row["sell_contribution"] == pytest.approx(0.0)
```

Add `import pytest` at the top of the test file if not already present (check first — it is not currently imported in `tests/unit/test_dashboard_market_switching.py`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_get_combined_ratings_computes_sell_and_net_rating -v`
Expected: FAIL — `KeyError: 'sell_rating'`.

- [ ] **Step 3: Rewrite `get_combined_ratings()` in `dashboard/data_loader.py`**

Replace the full function (keep its existing docstring, only extend the body) from:

```python
def get_combined_ratings(market: str = "us") -> tuple[list[dict], dict[str, list[dict]]]:
    """Aggregate every strategy's live signal for each ticker into one
    overall Buy Rating, weighted by that strategy's leaderboard composite
    score — stronger track-record strategies count for more. Cache-only
    (no live-compute fallback): both signals.json and leaderboard.json must
    already exist.

    Returns (summary_rows, detail_by_ticker):
      summary_rows: one row per ticker — ticker, overall_rating (0-100),
        n_buy, n_strategies, date, entry_price — sorted by overall_rating
        descending.
      detail_by_ticker: ticker -> per-strategy contribution rows (each with
        its own date/entry_price too), sorted by contribution descending,
        for the drill-down view.
    """
    leaderboard = _load_cache(get_cache_dir(market) / "leaderboard.json")
    signals = _load_cache(get_cache_dir(market) / "signals.json")
    if not leaderboard or not signals:
        return [], {}

    weights = {g["model_name"]: g["composite_score"] for g in leaderboard["grades"]}

    by_ticker: dict[str, list[dict]] = {}
    for s in signals["signals"]:
        by_ticker.setdefault(s["ticker"], []).append(s)

    summary_rows: list[dict] = []
    detail_by_ticker: dict[str, list[dict]] = {}
    for ticker, rows in by_ticker.items():
        total_weight = 0.0
        weighted_buy = 0.0
        n_buy = 0
        detail: list[dict] = []
        latest_date = max((r["date"] for r in rows), default="")
        entry_price = next((r["entry_price"] for r in rows if r["date"] == latest_date), 0.0)
        for r in rows:
            w = weights.get(r["strategy"], 0.0)
            if w <= 0:
                continue
            is_buy = r["signal"] == "Buy"
            contribution = w * r["confidence"] if is_buy else 0.0
            total_weight += w
            weighted_buy += contribution
            n_buy += int(is_buy)
            detail.append({
                "strategy": r["strategy"],
                "weight": w,
                "signal": r["signal"],
                "confidence": r["confidence"],
                "contribution": contribution,
                "date": r["date"],
                "entry_price": r["entry_price"],
            })
        if total_weight <= 0:
            continue
        detail.sort(key=lambda d: d["contribution"], reverse=True)
        detail_by_ticker[ticker] = detail
        summary_rows.append({
            "ticker": ticker,
            "overall_rating": 100.0 * weighted_buy / total_weight,
            "n_buy": n_buy,
            "n_strategies": len(detail),
            "date": latest_date,
            "entry_price": entry_price,
        })

    summary_rows.sort(key=lambda r: r["overall_rating"], reverse=True)
    return summary_rows, detail_by_ticker
```

to:

```python
def get_combined_ratings(market: str = "us") -> tuple[list[dict], dict[str, list[dict]]]:
    """Aggregate every strategy's live signal for each ticker into one
    overall Buy Rating and a parallel Sell Rating, both weighted by that
    strategy's leaderboard composite score — stronger track-record
    strategies count for more. Cache-only (no live-compute fallback): both
    signals.json and leaderboard.json must already exist.

    Returns (summary_rows, detail_by_ticker):
      summary_rows: one row per ticker — ticker, overall_rating (0-100,
        buy-side), sell_rating (0-100, sell-side), net_rating (-100 to 100,
        sell_rating - overall_rating), n_buy, n_sell, n_strategies, date,
        entry_price — sorted by overall_rating descending.
      detail_by_ticker: ticker -> per-strategy rows (contribution = buy-side,
        sell_contribution = sell-side; each with its own date/entry_price
        too), sorted by contribution descending, for the drill-down view.
    """
    leaderboard = _load_cache(get_cache_dir(market) / "leaderboard.json")
    signals = _load_cache(get_cache_dir(market) / "signals.json")
    if not leaderboard or not signals:
        return [], {}

    weights = {g["model_name"]: g["composite_score"] for g in leaderboard["grades"]}

    by_ticker: dict[str, list[dict]] = {}
    for s in signals["signals"]:
        by_ticker.setdefault(s["ticker"], []).append(s)

    summary_rows: list[dict] = []
    detail_by_ticker: dict[str, list[dict]] = {}
    for ticker, rows in by_ticker.items():
        total_weight = 0.0
        weighted_buy = 0.0
        weighted_sell = 0.0
        n_buy = 0
        n_sell = 0
        detail: list[dict] = []
        latest_date = max((r["date"] for r in rows), default="")
        entry_price = next((r["entry_price"] for r in rows if r["date"] == latest_date), 0.0)
        for r in rows:
            w = weights.get(r["strategy"], 0.0)
            if w <= 0:
                continue
            is_buy = r["signal"] == "Buy"
            is_sell = r["signal"] == "Sell"
            contribution = w * r["confidence"] if is_buy else 0.0
            sell_contribution = w * r["confidence"] if is_sell else 0.0
            total_weight += w
            weighted_buy += contribution
            weighted_sell += sell_contribution
            n_buy += int(is_buy)
            n_sell += int(is_sell)
            detail.append({
                "strategy": r["strategy"],
                "weight": w,
                "signal": r["signal"],
                "confidence": r["confidence"],
                "contribution": contribution,
                "sell_contribution": sell_contribution,
                "date": r["date"],
                "entry_price": r["entry_price"],
            })
        if total_weight <= 0:
            continue
        detail.sort(key=lambda d: d["contribution"], reverse=True)
        detail_by_ticker[ticker] = detail
        summary_rows.append({
            "ticker": ticker,
            "overall_rating": 100.0 * weighted_buy / total_weight,
            "sell_rating": 100.0 * weighted_sell / total_weight,
            "net_rating": 100.0 * (weighted_sell - weighted_buy) / total_weight,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_strategies": len(detail),
            "date": latest_date,
            "entry_price": entry_price,
        })

    summary_rows.sort(key=lambda r: r["overall_rating"], reverse=True)
    return summary_rows, detail_by_ticker
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS, including the new test and every pre-existing one in this file (this change is purely additive to the function's return shape).

- [ ] **Step 5: Commit**

```bash
git add dashboard/data_loader.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: add sell_rating/net_rating to get_combined_ratings()"
```

---

### Task 5: New page `dashboard/pages/6_Should_I_Sell.py`

**Files:**
- Create: `dashboard/pages/6_Should_I_Sell.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.get_selected_market()` / `format_price()`, `dashboard.ui_config.get_paths(market)`, `dashboard.data_loader.get_combined_ratings(market)` (Task 4), `src.features.duckdb_client.load_training_data`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_should_i_sell_page_is_market_aware():
    source = Path("dashboard/pages/6_Should_I_Sell.py").read_text()
    assert "get_selected_market" in source
    assert "get_combined_ratings(market=market)" in source
    assert "format_price" in source


def test_should_i_sell_page_renders_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("dashboard/pages/6_Should_I_Sell.py", default_timeout=60)
    at.session_state["market"] = "us"
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_should_i_sell_page_is_market_aware tests/unit/test_dashboard_market_switching.py::test_should_i_sell_page_renders_without_exception -v`
Expected: FAIL — the file doesn't exist yet.

- [ ] **Step 3: Create `dashboard/pages/6_Should_I_Sell.py`**

```python
from datetime import date

import pandas as pd
import streamlit as st

from config.markets import get_market
from dashboard.data_loader import get_combined_ratings
from dashboard.market_state import get_selected_market, format_price
from dashboard.ui_config import get_paths
from src.features.duckdb_client import load_training_data

st.set_page_config(page_title="Should I Sell?", layout="wide")
st.header("Should I Sell?")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")

summary_rows, detail_by_ticker = get_combined_ratings(market=market)

if not summary_rows:
    st.warning(
        "No cached signals/leaderboard found. Run "
        f"`python scripts/precompute_dashboard.py --market {market}` first."
    )
    st.stop()

tickers = sorted(r["ticker"] for r in summary_rows)
ticker = st.selectbox("Which stock did you buy?", tickers)

col1, col2 = st.columns(2)
with col1:
    buy_price_input = st.number_input(
        "Buy price (optional)", min_value=0.0, value=0.0, step=0.01,
        help="Leave at 0 to use the buy date instead, or leave both blank for no P&L.",
    )
with col2:
    buy_date_input = st.date_input(
        "Buy date (optional)", value=None,
        help="Used to look up that day's closing price if buy price is left blank.",
    )

row = next(r for r in summary_rows if r["ticker"] == ticker)
current_price = row["entry_price"]

buy_price: float | None = buy_price_input if buy_price_input > 0 else None

if buy_price is None and buy_date_input is not None:
    parquet_dir, _ = get_paths(market)
    hist = load_training_data(parquet_dir, tickers=[ticker])
    if not hist.is_empty():
        as_of = hist.filter(hist["time"].dt.date() <= buy_date_input).sort("time")
        if len(as_of) > 0:
            buy_price = float(as_of["close"][-1])
        else:
            st.error(
                f"No trading data for {ticker} on or before {buy_date_input}. "
                "Try an earlier date or enter the buy price directly."
            )

if buy_price is not None:
    pnl_pct = (current_price - buy_price) / buy_price * 100.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Buy price", format_price(buy_price, market))
    c2.metric("Current price", format_price(current_price, market))
    c3.metric("Unrealized P&L", f"{pnl_pct:+.1f}%")
    st.divider()

net_rating = row["net_rating"]
if net_rating >= 12:
    label, color = "Strong Sell", "#e74c3c"
elif net_rating >= 4:
    label, color = "Consider Selling", "#e67e22"
elif net_rating > -4:
    label, color = "Hold", "#f1c40f"
else:
    label, color = "Keep Holding (Bullish)", "#2ecc71"

st.markdown(
    f"### Recommendation: <span style='color:{color};font-size:1.3em'>{label}</span>",
    unsafe_allow_html=True,
)
st.caption(
    f"Net signal score: {net_rating:+.1f} "
    f"(sell rating {row['sell_rating']:.1f}, buy rating {row['overall_rating']:.1f}, "
    f"from {row['n_strategies']} strategies — {row['n_buy']} Buy, {row['n_sell']} Sell)"
)

st.subheader(f"Per-strategy breakdown — {ticker}")
detail = detail_by_ticker.get(ticker, [])
ddf = pd.DataFrame(detail)
ddf["confidence"] = ddf["confidence"].round(3)
ddf["weight"] = ddf["weight"].round(3)
ddf["contribution"] = ddf["contribution"].round(3)
ddf["sell_contribution"] = ddf["sell_contribution"].round(3)
ddf = ddf[["strategy", "weight", "signal", "confidence", "contribution", "sell_contribution"]]
ddf.columns = ["Strategy", "Weight (composite score)", "Signal", "Confidence", "Buy Contribution", "Sell Contribution"]
st.dataframe(ddf, use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/6_Should_I_Sell.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: add Should I Sell? page (ticker + buy price/date -> sell recommendation)"
```

---

### Task 6: Real data regeneration for US market

**Files:**
- No source changes — this task runs the already-fixed pipeline for real.

- [ ] **Step 1: Run the real precompute command**

Run (this re-fits all 32 strategies and re-runs every walk-forward backtest for the US market's ~492 tickers — will take a while; do not interrupt it):

```bash
python scripts/precompute_dashboard.py --market us
```

Expected: exit code 0, log shows all 4 steps (`[1/4]` through `[4/4]`) completing.

- [ ] **Step 2: Verify real Sell signals now appear**

Run:
```bash
python -c "
import json
from collections import Counter
d = json.loads(open('markets/us/data/cache/signals.json', encoding='utf-8').read())
c = Counter(s['signal'] for s in d['signals'])
print(c)
assert c['Sell'] > 0, 'expected at least some real Sell signals after the fix'
print('OK')
"
```
Expected: prints a `Counter` with `Buy`, `Hold`, and now `Sell` all present, then `OK`.

- [ ] **Step 3: Commit the regenerated data**

```bash
git add markets/us/data/cache
git commit -m "data: regenerate US backtests/leaderboard/signals with real Sell classifications"
```

---

### Task 7: Real data regeneration for China market

**Files:**
- No source changes — this task runs the already-fixed pipeline for real.

**Context:** `precompute_dashboard.py --market china`'s live-signals step previously crashed repeatedly with a Rust/Polars allocator panic at full 500-ticker scale during the China full-universe-training plan; root cause was never fully pinned down. The working fallback from that plan was to compute live signals ticker-by-ticker instead of the framework's default all-tickers-then-concat/predict approach (that one-off script was deleted after use, since it wasn't meant to be permanent — see `docs/superpowers/plans/2026-07-31-china-full-universe-training.md`'s Deviations section for the full history).

- [ ] **Step 1: Attempt the real precompute command**

Run (this re-fits all 32 strategies and re-runs every walk-forward backtest for China's ~500 tickers — will take a while; do not interrupt it):

```bash
python scripts/precompute_dashboard.py --market china
```

Expected: exit code 0. **If this crashes during the live-signals step ([4/4]) with a `memory allocation ... failed` / `RUST_BACKTRACE` panic:** do not retry the same command. Steps [1/4]-[3/4] (data summary, all 32 backtests, leaderboard) do not have a history of crashing — only [4/4] (live signals) does. Proceed to Step 2.

- [ ] **Step 2 (only if Step 1's live-signals step crashed): Regenerate live signals ticker-by-ticker**

Confirm no stray Python processes are running first (`Get-Process python` via PowerShell should show nothing, or nothing consuming significant CPU/memory from a prior attempt). Steps [1/4]-[3/4]'s output (`data_summary.json`, all 32 `backtest_*.json`, `leaderboard.json`) will already be correct and in place from Step 1 of this task before the crash, since the crash is specific to [4/4] — only `signals.json` needs this workaround.

Create `scratch_china_signals_one_by_one.py` at the repo root (not part of the codebase — delete it after use, per Step 4):

```python
"""One-off driver: compute China live signals ticker-by-ticker to bound peak
memory (build_live_features()'s all-500-tickers-in-one-list-then-concat
approach has previously panicked with a Rust allocator OOM on this machine
at full CSI 500 scale). Produces the exact same
markets/china/data/cache/signals.json that step_signals(market="china")
would, just without ever holding more than one ticker's live-feature frame
in memory at a time. Not part of the codebase — delete after running.
"""
import gc
import logging

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS, OHLCV_COLS
from scripts.build_features import _TICKERS_BY_MARKET, build_features_for_ticker
from scripts.precompute_dashboard import _now, _write, _market_paths
from src.backtesting.strategy_runner import _is_stateless, _select_cols
from src.features.cross_asset_features import synthetic_vol_index
from src.features.duckdb_client import load_training_data
from src.strategies.registry import list_strategies, load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MARKET = "china"
feature_dir, cache_dir = _market_paths(MARKET)
market_cfg = get_market(MARKET)
raw_dir = market_cfg.data_root / "raw" / "ohlcv"

logger.info("Loading training data from %s ...", feature_dir)
df = load_training_data(feature_dir)
logger.info("Loaded %d rows", len(df))

logger.info("Fitting all strategies once ...")
fitted: dict[str, object] = {}
for name in list_strategies():
    strategy = load_strategy(name)
    if not _is_stateless(strategy):
        train_pd = _select_cols(df, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
        if not getattr(strategy, "handles_nan", False):
            train_pd = train_pd.dropna()
        if len(train_pd) < 100:
            logger.warning("  %s: too few rows after dropna, skipping", name)
            continue
        strategy.fit(train_pd)
        del train_pd
    fitted[name] = strategy
logger.info("Fitted %d/%d strategies", len(fitted), len(list_strategies()))

del df
gc.collect()

logger.info("Loading benchmark/vol-index once ...")
import polars as pl
benchmark_df = pl.read_parquet(raw_dir / f"{market_cfg.benchmark_ticker}.parquet")
if market_cfg.vol_index_ticker:
    vix_df = pl.read_parquet(raw_dir / f"{market_cfg.vol_index_ticker}.parquet")
else:
    vix_df = synthetic_vol_index(benchmark_df)

tickers = _TICKERS_BY_MARKET[MARKET]
all_signals: list[dict] = []
n_done = 0
for ticker in tickers:
    raw_path = raw_dir / f"{ticker}.parquet"
    if not raw_path.exists():
        continue
    try:
        t_pl = build_features_for_ticker(ticker, raw_dir, benchmark_df, vix_df, drop_label_nulls=False)
    except Exception as exc:
        logger.warning("  live features FAILED %s: %s", ticker, exc)
        continue

    for name, strategy in fitted.items():
        t_pd = _select_cols(t_pl, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
        if len(t_pd) == 0 or "time" not in t_pd.columns:
            continue
        try:
            result = strategy.predict(t_pd)
        except Exception as exc:
            logger.debug("  %s %s predict failed: %s", name, ticker, exc)
            continue
        pos = len(t_pd) - 1
        if pos >= len(result.signal):
            continue
        sig = str(result.signal.iloc[pos])
        conf = float(result.confidence.iloc[pos])
        close = float(t_pd["close"].iloc[pos]) if "close" in t_pd.columns else 0.0
        ticker_date = t_pd["time"].iloc[pos]
        all_signals.append({
            "ticker": ticker,
            "date": str(ticker_date),
            "signal": sig,
            "confidence": conf,
            "entry_price": close,
            "position_size": conf,
            "strategy": name,
        })
        del t_pd

    del t_pl
    n_done += 1
    if n_done % 25 == 0:
        gc.collect()
        logger.info("  processed %d/%d tickers, %d signals so far", n_done, len(tickers), len(all_signals))

buy_count = sum(1 for s in all_signals if s["signal"] == "Buy")
sell_count = sum(1 for s in all_signals if s["signal"] == "Sell")
logger.info("total signals: %d  buy: %d  sell: %d  (from %d tickers)", len(all_signals), buy_count, sell_count, n_done)
_write(cache_dir / "signals.json", {
    "generated_at": _now(),
    "signals": all_signals,
})
print("DONE")
```

Run it: `python scratch_china_signals_one_by_one.py`

- [ ] **Step 3: Verify real Sell signals now appear**

Run:
```bash
python -c "
import json
from collections import Counter
d = json.loads(open('markets/china/data/cache/signals.json', encoding='utf-8').read())
c = Counter(s['signal'] for s in d['signals'])
print(c)
assert c['Sell'] > 0, 'expected at least some real Sell signals after the fix'
print('OK')
"
```
Expected: prints a `Counter` with `Buy`, `Hold`, and now `Sell` all present, then `OK`.

- [ ] **Step 4: Delete `scratch_china_signals_one_by_one.py` if it was created in Step 2, then commit the regenerated data**

```bash
git add markets/china/data/cache
git commit -m "data: regenerate China backtests/leaderboard/signals with real Sell classifications"
```

---

### Task 8: Final validation

**Files:**
- No source changes — this task verifies the finished work end-to-end with real data.

- [ ] **Step 1: Run the full unit test suite**

Run: `python -m pytest tests/unit -q`
Expected: pass count at or above the pre-plan baseline (386 passed, 15 pre-existing failures unrelated to this plan, 2 skipped — per the dashboard market-switching plan's final state), plus the new tests added across Tasks 1-5.

- [ ] **Step 2: Integration check against real regenerated data — both markets**

Run (adjust ticker names to real ones present in each market's regenerated `signals.json` — pick any ticker whose `net_rating` is clearly positive or negative rather than near-zero, so the recommendation label is unambiguous):

```bash
python -c "
from dashboard.data_loader import get_combined_ratings

for market in ('us', 'china'):
    summary_rows, _ = get_combined_ratings(market=market)
    assert summary_rows, f'no summary rows for {market}'
    sell_leaning = [r for r in summary_rows if r['net_rating'] > 0]
    print(market, 'tickers with net_rating > 0:', len(sell_leaning), '/', len(summary_rows))
"
```
Expected: both markets show at least some tickers with `net_rating > 0` (i.e. the sell side genuinely contributes now, not just theoretically).

- [ ] **Step 3: AppTest-based recommendation check against real data**

Run a script (scratchpad, not committed) that drives `dashboard/pages/6_Should_I_Sell.py` via `AppTest`, selects a ticker known (from Step 2's output) to have a clearly positive `net_rating` in one market, and confirms the rendered recommendation text says "Sell" (either "Strong Sell" or "Consider Selling") rather than "Hold" or "Keep Holding" — a real end-to-end check that the new page's bucket logic matches the real computed `net_rating`, not just that the page renders without crashing.

- [ ] **Step 4: Manual verification**

Run: `streamlit run dashboard/app.py`

Checklist:
1. Navigate to "Should I Sell?" for each market. Pick a ticker, leave buy price/date blank — confirm a recommendation renders with no P&L section.
2. Enter a buy price — confirm P&L renders correctly (sign and magnitude sane for a price above/below current).
3. Leave price blank, enter a buy date in the past — confirm it resolves to a real historical price and P&L renders.
4. Enter a buy date before the ticker's earliest data — confirm the explicit error message renders (not a crash).
5. Switch markets on the home page, revisit "Should I Sell?" — confirm the ticker list and recommendation are for the newly-selected market (no cross-market leakage, same property verified for the other 5 pages in the prior plan).

- [ ] **Step 5: Commit** (only if Steps 1-4 surfaced a fix; otherwise this task has no commit of its own — Tasks 1-7's commits are the complete deliverable)
