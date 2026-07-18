# Expand Data + Add 5 Strategies Design

## Goal

Refresh the feature dataset to today (2026-07-18), add 5 new trading strategies (2 ML + 3 rule-based), retrain all 43 strategies via walk-forward backtest, and update the dashboard cache.

## Architecture

Three sequential stages: (1) incremental data refresh for 151 tickers, (2) implement and register 5 new strategies, (3) run full precompute and refresh dashboard cache. Stages are independent in implementation but must execute in order at runtime.

## Tech Stack

- Data: `yfinance`, `polars`, parquet
- ML: `catboost`, `scikit-learn` (`HistGradientBoostingClassifier`)
- Rule-based: pure pandas/numpy (same pattern as existing rule-based strategies)
- Dashboard: Streamlit reading from `data/cache/*.json`

---

## Global Constraints

- Python 3.14, scikit-learn 1.9+, polars, pandas
- All strategy files follow the existing pattern: inherit `Strategy`, implement `fit()`/`predict()` returning `PredictionResult`
- ML strategies: `data_source = "features"`, `fillna(0.0)` in fit/predict (or `handles_nan = True` for native NaN support)
- Rule-based strategies: `data_source = "ohlcv"`, stateless (no `fit()` override), buy signal `confidence` in `[0, 1]`
- `_META = {"time", "ticker", "label", "forward_return_5d"}` — excluded from ML feature columns
- Buy signal threshold: `>= 0.6` on `predict_proba` confidence for ML; natural `[0,1]` for rule-based
- 6 unit tests per strategy file (same structure as existing tests)
- Walk-forward: 400-day train, 21-day test, 21-day step
- `PYTHONPATH=c:/Users/h1810/.vscode/EXP` for all script runs
- Skip-if-exists guard in precompute script (existing 38 JSONs not re-run)
- Commit cache after precompute completes

---

## Stage 1 — Incremental Data Refresh

### What

Fetch the missing tail (2026-07-08 → 2026-07-18) for all 151 tickers in `data/raw/ohlcv/`, append to existing raw parquets, then rebuild `data/features/*.parquet` via `scripts/build_features.py`.

### Script: `scripts/refresh_data.py`

New script that:
1. Reads each existing `data/raw/ohlcv/{ticker}.parquet`, finds its `max(time)`
2. Fetches from `max_date + 1 day` to `today` via `src.ingestion.historical_collector.fetch_ohlcv`
3. Appends new rows (deduplicating on `time`) and writes back
4. Also refreshes `SPY.parquet` and `^VIX.parquet` (needed by feature builder)
5. After all tickers updated, calls `build_features_for_ticker` from `scripts/build_features.py` for each ticker

Tickers list: same `_STOCK_TICKERS` list as in `scripts/build_features.py` (151 entries), plus `SPY` and `^VIX`.

If a ticker fetch returns empty (market closed, delisted), log a warning and skip — do not abort.

---

## Stage 2 — New Strategies

### 2a. CatBoostClassifier (`src/strategies/statistical/catboost_strategy.py`)

```python
from catboost import CatBoostClassifier

class CatBoostStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, iterations: int = 200, depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            loss_function="MultiClass", eval_metric="Accuracy",
            random_seed=42, verbose=0,
        )
        self._classes: list[str] = []
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)
        self._classes = list(self._model.classes_)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        if "Buy" not in self._classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0]*n), signal=pd.Series(["Hold"]*n))
        buy_idx = self._classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

Test file: `tests/unit/test_catboost_strategy.py` — 6 tests matching the pattern of `test_xgboost_strategy.py`.

### 2b. HistGradientBoosting (`src/strategies/statistical/hist_gradient_boosting.py`)

```python
from sklearn.ensemble import HistGradientBoostingClassifier

class HistGradientBoostingStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, max_iter: int = 200, max_depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=max_iter, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42,
        )
        self._le = LabelEncoder()
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = self._le.fit_transform(df["label"].to_numpy())
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._le.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0]*n), signal=pd.Series(["Hold"]*n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
```

Test file: `tests/unit/test_hist_gradient_boosting.py` — 6 tests.

### 2c. SuperTrend (`src/strategies/rule_based/supertrend.py`)

ATR-based trend line. Parameters: `atr_period=10`, `multiplier=3.0`.

```
TR = max(high-low, |high-prev_close|, |low-prev_close|)
ATR = TR.rolling(atr_period).mean()
upper_band = (high+low)/2 + multiplier * ATR
lower_band = (high+low)/2 - multiplier * ATR

supertrend: starts at lower_band
  if close > supertrend[-1]: supertrend = max(lower_band, supertrend[-1])  # uptrend
  else: supertrend = min(upper_band, supertrend[-1])               # downtrend

buy = close > supertrend
confidence = ((close - supertrend) / close).clip(0, 0.05) / 0.05
```

`data_source = "ohlcv"`, stateless. Returns `PredictionResult`.
Test file: `tests/unit/test_supertrend.py` — 6 tests.

### 2d. ADX Trend Filter (`src/strategies/rule_based/adx_trend.py`)

Average Directional Index. Parameters: `period=14`, `adx_threshold=25`.

```
+DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
-DM = max(prev_low - low,   0) if prev_low - low > high - prev_high else 0
TR = max(high-low, |high-prev_close|, |low-prev_close|)

smoothed with Wilder's moving average (EWM span = 2*period - 1):
+DI = 100 * (+DM_smooth / TR_smooth)
-DI = 100 * (-DM_smooth / TR_smooth)
DX  = 100 * |+DI - -DI| / (+DI + -DI)
ADX = DX.ewm(span=2*period-1).mean()

buy = (ADX > adx_threshold) AND (+DI > -DI)
confidence = (ADX / 100).clip(0, 1) * ((+DI - -DI) / 100).clip(0, 1)
```

`data_source = "ohlcv"`, stateless.
Test file: `tests/unit/test_adx_trend.py` — 6 tests.

### 2e. Money Flow Index (`src/strategies/rule_based/money_flow_index.py`)

Volume-weighted RSI. Parameters: `period=14`, `oversold=30`, `overbought=70`.

```
typical_price = (high + low + close) / 3
raw_money_flow = typical_price * volume
positive_flow = raw_money_flow where typical_price > prev_typical_price else 0
negative_flow = raw_money_flow where typical_price < prev_typical_price else 0
MFR = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
MFI = 100 - (100 / (1 + MFR))

buy = MFI < oversold  (oversold recovery signal)
confidence = ((oversold - MFI) / oversold).clip(0, 1)  — higher when more oversold
```

`data_source = "ohlcv"`, stateless.
Test file: `tests/unit/test_money_flow_index.py` — 6 tests.

### 2f. Registry update (`src/strategies/strategies.yaml`)

Append 5 entries:
```yaml
- name: catboost_strategy
  class: src.strategies.statistical.catboost_strategy.CatBoostStrategy
- name: hist_gradient_boosting
  class: src.strategies.statistical.hist_gradient_boosting.HistGradientBoostingStrategy
- name: supertrend
  class: src.strategies.rule_based.supertrend.SuperTrend
- name: adx_trend
  class: src.strategies.rule_based.adx_trend.ADXTrend
- name: money_flow_index
  class: src.strategies.rule_based.money_flow_index.MoneyFlowIndex
```

Total strategies: **43**.

---

## Stage 3 — Precompute & Dashboard Update

### Script update: `scripts/precompute_new_strategies.py`

Add the 5 new strategy names to `_NEW_STRATEGIES`. The skip-if-exists guard already in place means the 38 existing JSONs are untouched. Only the 5 new ones get backtested.

After all 5 complete, the existing `step_leaderboard()` + `step_signals()` calls regenerate the full leaderboard and signals across all 43 strategies.

### Dashboard

No code changes needed. The Streamlit pages read from `data/cache/leaderboard.json` and `data/cache/signals.json` which are refreshed by the precompute script.

After precompute: `git add data/ && git commit -m "data: refresh to 2026-07-18, 43 strategies"`.

---

## Execution Order

1. `python scripts/refresh_data.py` — fetch missing OHLCV tail + rebuild features (~5 min)
2. Implement 5 new strategies + tests, register in YAML
3. `python scripts/precompute_new_strategies.py` — backtest 5 new, update leaderboard + signals
4. Commit cache + push
