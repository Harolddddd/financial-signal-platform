# Add 8 New Trading Strategies Design

**Date:** 2026-07-16  
**Status:** Approved

---

## Goal

Add 8 new trading strategies (4 ML + 4 rule-based), train them on the full dataset (151 tickers, 1980–2026), and update the dashboard cache. Total strategy count: 30 → 38.

## Architecture

All new strategies follow established project conventions exactly:

- **ML strategies**: subclass `Strategy`, set `data_source = "features"`, implement `fit()` and `predict()`, use `fillna(0.0)` for NaN (none of the 4 new ML models handle NaN natively), return `PredictionResult(confidence, signal)`, buy threshold `>= 0.6`
- **Rule-based strategies**: subclass `Strategy`, set `data_source = "ohlcv"`, stateless (no `fit()` override), implement `predict()` only
- All strategies registered in `src/strategies/strategies.yaml`
- Backtest cache is incremental: run walk-forward only for the 8 new strategies, leave existing `backtest_*.json` untouched, regenerate `leaderboard.json` and `signals.json` from all 38 files

## New ML Strategies (4)

### 1. `svm_strategy` — Support Vector Machine
- **File:** `src/strategies/statistical/svm_strategy.py`
- **Class:** `SVMStrategy(Strategy)`
- **Model:** `sklearn.svm.SVC(kernel="rbf", probability=True, class_weight="balanced")`
- `data_source = "features"`, no `handles_nan`
- `fillna(0.0)` in both `fit()` and `predict()`
- `_feature_cols` excludes `_META = {"time", "ticker", "label", "forward_return_5d"}`
- Buy signal: `proba[:, buy_idx] >= 0.6`
- Note: SVC with `probability=True` uses Platt scaling (slower on large datasets) — limit to 100k rows or use `max_iter=1000` to cap training time

### 2. `mlp_classifier` — Neural Network
- **File:** `src/strategies/statistical/mlp_classifier.py`
- **Class:** `MLPStrategy(Strategy)`
- **Model:** `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)`
- `data_source = "features"`, no `handles_nan`
- `fillna(0.0)` in both `fit()` and `predict()`
- Buy signal: `proba[:, buy_idx] >= 0.6`

### 3. `adaboost` — AdaBoost Classifier
- **File:** `src/strategies/statistical/adaboost.py`
- **Class:** `AdaBoostStrategy(Strategy)`
- **Model:** `sklearn.ensemble.AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")`
- `data_source = "features"`, no `handles_nan`
- `fillna(0.0)` in both `fit()` and `predict()`
- Buy signal: `proba[:, buy_idx] >= 0.6`

### 4. `knn_classifier` — K-Nearest Neighbors
- **File:** `src/strategies/statistical/knn_classifier.py`
- **Class:** `KNNStrategy(Strategy)`
- **Model:** `sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, weights="distance", n_jobs=-1)`
- `data_source = "features"`, no `handles_nan`
- `fillna(0.0)` in both `fit()` and `predict()`
- Buy signal: `proba[:, buy_idx] >= 0.6`

## New Rule-Based Strategies (4)

### 5. `ichimoku_cloud` — Ichimoku Cloud
- **File:** `src/strategies/rule_based/ichimoku_cloud.py`
- **Class:** `IchimokuCloud(Strategy)`
- `data_source = "ohlcv"`, stateless
- **Signal logic:** Buy when price (`close`) > `max(senkou_a, senkou_b)` AND `tenkan` > `kijun`
  - `tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2`
  - `kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2`
  - `senkou_a = ((tenkan + kijun) / 2).shift(26)`
  - `senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)`
- Confidence: `1.0` if Buy, `0.0` otherwise
- Requires columns: `open`, `high`, `low`, `close`, `volume`

### 6. `chaikin_money_flow` — Chaikin Money Flow
- **File:** `src/strategies/rule_based/chaikin_money_flow.py`
- **Class:** `ChaikinMoneyFlow(Strategy)`
- `data_source = "ohlcv"`, stateless
- **Signal logic:** Buy when `CMF(20) > 0.0`
  - `mfm = ((close - low) - (high - close)) / (high - low).replace(0, NaN)` (money flow multiplier)
  - `mfv = mfm * volume` (money flow volume)
  - `cmf = mfv.rolling(20).sum() / volume.rolling(20).sum()`
- Confidence: `clip(cmf, 0, 1)` when Buy, `0.0` otherwise

### 7. `aroon_oscillator` — Aroon Oscillator
- **File:** `src/strategies/rule_based/aroon_oscillator.py`
- **Class:** `AroonOscillator(Strategy)`
- `data_source = "ohlcv"`, stateless
- **Signal logic:** Buy when Aroon oscillator > `+50`
  - `period = 25`
  - `aroon_up   = (high.rolling(period+1).apply(lambda x: x.argmax()) / period) * 100`
  - `aroon_down = (low.rolling(period+1).apply(lambda x: x.argmin()) / period) * 100`
  - `aroon_osc  = aroon_up - aroon_down`
  - Note: `argmax()` returns 0-indexed position from oldest (left), so position=period means newest bar → aroon_up=100
- Confidence: `clip(aroon_osc / 100, 0, 1)` when Buy, `0.0` otherwise

### 8. `vwap_cross` — VWAP Cross
- **File:** `src/strategies/rule_based/vwap_cross.py`
- **Class:** `VWAPCross(Strategy)`
- `data_source = "ohlcv"`, stateless
- **Signal logic:** Buy when `close` crosses above rolling VWAP (20-day window)
  - `typical_price = (high + low + close) / 3`
  - `vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()`
  - Signal: Buy when `close > vwap` (price above VWAP — trend continuation, more signals than a one-bar crossover)
- Confidence: `clip((close - vwap) / vwap, 0, 0.1) * 10` (percentage gap above VWAP, scaled to [0,1]) when Buy, `0.0` otherwise

## strategies.yaml Additions

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
    params: {}

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

## Tests

Each strategy gets a unit test file in `tests/unit/` with 6 tests (same pattern as existing ML test files):
1. `fit()` + `predict()` correct output length
2. Signals in `{"Buy", "Hold"}`
3. Confidence in `[0.0, 1.0]`
4. `predict()` before `fit()` raises `ValueError` (ML only; rule-based N/A)
5. `data_source == "features"` or `"ohlcv"`
6. `handles_nan` attribute check (ML strategies: absent or False)

## Incremental Backtest Script

A new script `scripts/precompute_new_strategies.py` runs only the 8 new strategies (not all 38), then regenerates leaderboard and signals. This avoids re-running ~3 hours of RF/ExtraTrees backtest.

The script:
1. Calls `step_backtests()` filtered to new strategy names
2. Calls `step_leaderboard()` (reads all 38 `backtest_*.json`)
3. Calls `step_signals()` (runs all 38 for live signals)

## Global Constraints

- Buy signal threshold: `>= 0.6` (all strategies, consistent)
- `_META = {"time", "ticker", "label", "forward_return_5d"}` (excluded from feature columns)
- Walk-forward: 400-day train, 21-day test, 21-day step (unchanged)
- Grading formula: `0.40 × precision_buy + 0.30 × tanh(sharpe/2) + 0.30 × (1 − min(drawdown,50%)/50%)`
- No new data ingestion — use existing 151-ticker dataset as-is
- `PYTHONPATH="c:/Users/h1810/.vscode/EXP"` for all script execution
- SVC training note: SVC can be slow on large folds — use `max_iter=1000` to cap, acceptable to have some non-converged folds
