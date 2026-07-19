# Add 5 Strategies Wave 2 Design

## Goal

Add 5 new trading strategies (2 ML + 3 rule-based) to the platform (currently 43), retrain via walk-forward backtest over 151 tickers, and update the dashboard cache. No data refresh needed — features are current to 2026-07-18.

## Architecture

Same two-stage pipeline as wave 1: implement strategy classes + unit tests, register in `strategies.yaml` (43 → 48), add to `_NEW_STRATEGIES` in precompute script, run walk-forward backtest, regenerate leaderboard + signals.

## Tech Stack

- ML: `scikit-learn` (`GaussianNB`, `LinearDiscriminantAnalysis`, `LabelEncoder`)
- Rule-based: pure pandas/numpy
- Dashboard: Streamlit reading from `data/cache/*.json`

---

## Global Constraints

- Python 3.14, scikit-learn 1.9+, polars, pandas
- ML strategies: `data_source = "features"`, `fillna(0.0)` in fit/predict (GaussianNB and LDA do not support NaN natively)
- Rule-based strategies: `data_source = "ohlcv"`, stateless (no `fit()` override), `confidence = 0.0` on Hold rows
- `_META = {"time", "ticker", "label", "forward_return_5d"}` — excluded from ML feature columns
- Buy signal threshold: `>= 0.6` on `predict_proba` for ML
- 6 unit tests per strategy file
- Walk-forward: 400-day train, 21-day test, 21-day step
- `PYTHONPATH=c:/Users/h1810/.vscode/EXP` for all script runs
- Skip-if-exists guard in precompute script (existing 43 JSONs not re-run)

---

## Strategy Specs

### ML 1: GaussianNB (`src/strategies/statistical/gaussian_nb_strategy.py`)

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

class GaussianNBStrategy(Strategy):
    data_source = "features"

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self._model = GaussianNB(var_smoothing=var_smoothing)
        self._le = LabelEncoder()
        self._feature_cols: list[str] = []

    def fit(self, df):
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = self._le.fit_transform(df["label"].to_numpy())
        self._model.fit(X, y)

    def predict(self, df):
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
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

Test file: `tests/unit/test_gaussian_nb_strategy.py` — 6 tests matching xgboost pattern (fit_predict_correct_length, signals_buy_or_hold, confidence_in_unit_range, predict_before_fit_raises, data_source_is_features, no handles_nan attribute needed).

### ML 2: LDA (`src/strategies/statistical/lda_strategy.py`)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAStrategy(Strategy):
    data_source = "features"

    def __init__(self, solver: str = "svd") -> None:
        self._model = LinearDiscriminantAnalysis(solver=solver)
        self._le = LabelEncoder()
        self._feature_cols: list[str] = []

    # fit/predict identical pattern to GaussianNBStrategy
```

Test file: `tests/unit/test_lda_strategy.py` — 6 tests.

### Rule-based 1: StochRSI (`src/strategies/rule_based/stoch_rsi.py`)

Parameters: `rsi_period=14`, `stoch_period=14`, `oversold=0.2`

```
RSI = 100 - 100 / (1 + avg_gain.ewm(span=rsi_period).mean() / avg_loss.ewm(span=rsi_period).mean())
stoch_rsi = (RSI - RSI.rolling(stoch_period).min()) / (RSI.rolling(stoch_period).max() - RSI.rolling(stoch_period).min())
stoch_rsi = stoch_rsi.clip(0, 1)

buy = stoch_rsi < oversold
confidence = ((oversold - stoch_rsi) / oversold).clip(0, 1)  on Buy, 0.0 on Hold
```

Guard: replace 0 in denominator with NaN before dividing.
Test file: `tests/unit/test_stoch_rsi.py` — 6 tests.

### Rule-based 2: HullMA (`src/strategies/rule_based/hull_ma.py`)

Parameters: `period=20`

```
WMA(series, n) = series.rolling(n).apply(lambda x: np.average(x, weights=np.arange(1, n+1)))
HMA = WMA(2 * WMA(close, period//2) - WMA(close, period), int(np.sqrt(period)))

buy = close > HMA
confidence = ((close - HMA) / close).clip(0, 0.02) / 0.02  on Buy, 0.0 on Hold
```

Test file: `tests/unit/test_hull_ma.py` — 6 tests.

### Rule-based 3: TRIX (`src/strategies/rule_based/trix.py`)

Parameters: `period=15`

```
ema1 = close.ewm(span=period, adjust=False).mean()
ema2 = ema1.ewm(span=period, adjust=False).mean()
ema3 = ema2.ewm(span=period, adjust=False).mean()
trix = (ema3 / ema3.shift(1) - 1) * 100  # percent change

buy = trix > 0
confidence = (trix / 2.0).clip(0, 1)  on Buy, 0.0 on Hold
```

Test file: `tests/unit/test_trix.py` — 6 tests.

---

## Registry Update

Append to `src/strategies/strategies.yaml`:

```yaml
  - name: gaussian_nb
    class: src.strategies.statistical.gaussian_nb_strategy.GaussianNBStrategy
    params:
      var_smoothing: 1.0e-9

  - name: lda_strategy
    class: src.strategies.statistical.lda_strategy.LDAStrategy
    params:
      solver: svd

  - name: stoch_rsi
    class: src.strategies.rule_based.stoch_rsi.StochRSI
    params:
      rsi_period: 14
      stoch_period: 14
      oversold: 0.2

  - name: hull_ma
    class: src.strategies.rule_based.hull_ma.HullMA
    params:
      period: 20

  - name: trix
    class: src.strategies.rule_based.trix.TRIX
    params:
      period: 15
```

Total: 48 strategies.

---

## Precompute

Add 5 new names to `_NEW_STRATEGIES` in `scripts/precompute_new_strategies.py`. Skip-if-exists guard ensures 43 existing JSONs are untouched. After all 5 complete, `step_leaderboard()` + `step_signals()` regenerate the full cache. Commit cache.
