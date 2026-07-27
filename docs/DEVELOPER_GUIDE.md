# Financial Signal Platform — Developer Guide

All commands in this document have been tested and run from `C:\Users\h1810\.vscode\EXP`
in a Python terminal (`python` or `python -c "..."`). Every snippet is copy-paste ready.

---

## Architecture Overview

Data flows through four independent layers. You can extend any one without touching the others.

```
Plan 1 — Ingestion
  ↓  raw OHLCV + news → PostgreSQL
Plan 2 — Feature Engineering
  ↓  22 features + Buy/Hold/Sell labels → Parquet / DuckDB
Plan 3 — Training & Tuning
  ↓  8 models, Optuna search → joblib registry
Plan 4 — Backtest + Signals + Dashboard
  ↓  A–D grades, live signals → Streamlit at localhost:8501
```

---

## Running the Project

### Dashboard only (no database needed)

Run this in a PowerShell terminal from the project root:

```powershell
streamlit run dashboard/app.py --server.headless true --server.port 8501
```

Open your browser to `http://localhost:8501`.
Synthetic data is already in `data/features/` (5 tickers × 800 rows).
Trained models are already in `data/registry/` (xgboost, random_forest, lightgbm).

### Run the test suite

```powershell
pytest tests/unit/ tests/integration/ -q
```

Expected output: `162 passed`.

### Full pipeline (requires PostgreSQL)

```powershell
# 1. Start the database
docker compose up -d postgres

# 2. Ingest OHLCV for a ticker
python -c "
from datetime import datetime, timezone
from src.ingestion.collector import collect_ohlcv
df = collect_ohlcv('AAPL', datetime(2020,1,1,tzinfo=timezone.utc), datetime(2024,1,1,tzinfo=timezone.utc))
print(df.shape)
"

# 3. Collect news
python -c "
from src.ingestion.news_collector import collect_rss
articles = collect_rss('AAPL', max_items=20)
print(len(articles), 'articles')
"

# 4. Build features and export parquet
python -c "
from datetime import datetime, timezone
from pathlib import Path
from src.features.feature_store import build_features, export_parquet
df = build_features('AAPL', datetime(2020,1,1,tzinfo=timezone.utc), datetime(2024,1,1,tzinfo=timezone.utc))
export_parquet(df, 'AAPL', Path('data/features'))
print('Exported', len(df), 'rows')
"

# 5. Load features and list models
python -c "
from pathlib import Path
from src.features.duckdb_client import load_training_data
from src.models.registry import list_models
df = load_training_data(Path('data/features'))
print('Feature data:', df.shape)
models = list_models(Path('data/registry'))
print('Models:', [m.model_name for m in models])
"

# 6. Load a model and grade it manually
python -c "
from pathlib import Path
from src.models.registry import load_model
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics
m = load_model('xgboost', Path('data/registry'))
metrics = BacktestMetrics(50, 0.55, 1.2, 0.08, 1.5, 0.12, 0.60, 0.50, 0.54, 0.65)
g = grade_model(m.name, metrics)
print(f'Grade: {g.grade.value}, score: {g.composite_score}')
"
```

---

## Part 1 — Data Ingestion

**Files:** `src/ingestion/`, `dags/historical_data_dag.py`, `dags/news_sentiment_dag.py`

**What it does:**
- `collect_ohlcv(ticker, start, end)` — tries yfinance → Alpha Vantage → FMP, returns a Polars DataFrame
- `collect_rss / collect_newsapi / collect_finnhub` — news from three sources, deduplicated by URL
- Sentiment (FinBERT or VADER) is applied in `src/ingestion/sentiment_processor.py`
- Everything written to PostgreSQL `ohlcv` and `news_articles` tables

### Verify ingestion works right now

```python
from datetime import datetime, timezone
from src.ingestion.collector import collect_ohlcv

df = collect_ohlcv(
    'MSFT',
    datetime(2024, 1, 1, tzinfo=timezone.utc),
    datetime(2024, 6, 1, tzinfo=timezone.utc),
)
print(df.shape)          # (N, 10)
print(df.columns)        # time, ticker, open, high, low, close, volume, adj_close, dividends, stock_splits
```

### Add a new data source

1. Create `src/ingestion/my_client.py` with a function `fetch_ohlcv_mine(ticker, start, end) -> pl.DataFrame`
   returning the same columns as the yfinance result above.

2. Add it to the fallback chain in `src/ingestion/collector.py`:

```python
sources = [
    ("yfinance",       lambda: fetch_ohlcv(ticker, start, end, interval)),
    ("alpha_vantage",  lambda: fetch_ohlcv_av(ticker, start, end)),
    ("fmp",            lambda: fetch_ohlcv_fmp(ticker, start, end)),
    ("mine",           lambda: fetch_ohlcv_mine(ticker, start, end)),  # add here
]
```

3. Add any API key to `config/settings.py` and your `.env` file.

### Key knobs

| File | Variable | Effect |
|---|---|---|
| `config/settings.py` | `HISTORICAL_DAYS = 3650` | Years of history pulled (default 10y) |
| `config/settings.py` | `STOCK_UNIVERSE = "sp500"` | Which tickers to process |
| `src/ingestion/sentiment_processor.py` | top of file | Switch between FinBERT and VADER |

---

## Part 2 — Feature Engineering

**Files:** `src/features/`, `dags/feature_engineering_dag.py`

**What it does:**
Pulls OHLCV + sentiment from PostgreSQL, computes 22 features, generates Buy/Hold/Sell labels,
exports one Parquet file per ticker to `data/features/`.

### The 22 feature columns

| Group | Columns |
|---|---|
| Moving averages | `sma_10`, `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26` |
| Momentum | `rsi_14`, `macd`, `macd_signal`, `macd_hist` |
| Volatility | `bb_upper`, `bb_lower`, `bb_width`, `atr_14`, `hist_vol_21` |
| Sentiment | `sent_pos_avg_3d`, `sent_pos_avg_5d`, `sent_pos_avg_10d`, `sent_pos_mom_3d`, `news_vol_spike` |
| Cross-asset | `rel_strength_spy`, `vix_level` |

### Add a new technical indicator

Open `src/features/technical_indicators.py`. All indicators follow the same pattern
— add a private function and call it from `add_technical_indicators()`:

```python
# src/features/technical_indicators.py

def _add_stochastic(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    low_n  = pl.col("low").rolling_min(window_size=period)
    high_n = pl.col("high").rolling_max(window_size=period)
    k = ((pl.col("close") - low_n) / (high_n - low_n) * 100).alias("stoch_k")
    return df.with_columns(k)

def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = _add_moving_averages(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger_bands(df)
    df = _add_atr(df)
    df = _add_hist_vol(df)
    df = _add_stochastic(df)   # <-- add here
    return df
```

Then register the new column name in **two places**:

1. The `_FEATURE_COLS` list at the top of `src/features/feature_store.py` (and add it to the SQL INSERT below)
2. The `FEATURE_COLS` list in `dashboard/config.py`

The model training and dashboard pick it up automatically after that.

### Add a sentiment indicator

`src/features/sentiment_features.py` receives aggregated daily news scores.
You can add rolling divergence, topic-specific scores (macro vs. earnings),
or Reddit/Twitter signals by computing new columns in `add_sentiment_features()`.
Register the column name in the same two places as above.

### Change the label logic

`src/features/label_generator.py`:

```python
def add_labels(
    df,
    forward_days: int = 5,       # horizon: predict 5-day forward return
    buy_threshold: float = 0.02,  # >+2% = Buy
    sell_threshold: float = -0.02 # <-2% = Sell
):
```

Examples of changes and their effect:

| Change | Effect |
|---|---|
| `forward_days=10` | 2-week signal, fewer labels, longer holding period |
| `buy_threshold=0.03` | Stricter Buy — rarer but higher-quality labels |
| `sell_threshold=-0.01` | Wider Sell net — more balanced class distribution |

After changing, re-export Parquet files so models train on the new labels.

### Test label generation in isolation

```python
from datetime import datetime, timedelta
import polars as pl
from src.features.label_generator import add_labels

df = pl.DataFrame({
    "time":  [datetime(2024,1,1) + timedelta(days=i) for i in range(20)],
    "close": [100.0 + i * 0.5 for i in range(20)],
})
out = add_labels(df, forward_days=5, buy_threshold=0.02)
print(out.select(["time", "close", "forward_return_5d", "label"]))
```

---

## Part 3 — Model Training & Tuning

**Files:** `src/models/`, `dags/model_retrain_dag.py`

**What it does:**
- Walk-forward training: retrain on a rolling window, test on the next period, repeat fold by fold
- Optuna hyperparameter search: each model has a `_PARAM_SPACES` entry defining the search space
- Best model saved to `data/registry/<model_name>/model.joblib`

### The 8-model zoo

| Model | File | Notes |
|---|---|---|
| Logistic Regression | `src/models/zoo/logistic_regression.py` | Fast baseline |
| Random Forest | `src/models/zoo/random_forest.py` | Balanced class weights |
| XGBoost | `src/models/zoo/xgboost_model.py` | Usually top performer |
| LightGBM | `src/models/zoo/lightgbm_model.py` | Fastest tree model |
| SVM | `src/models/zoo/svm_model.py` | Good on small datasets |
| Naive Bayes | `src/models/zoo/naive_bayes.py` | Useful as diversity check |
| MLP | `src/models/zoo/mlp_model.py` | Feedforward neural net |
| LSTM | `src/models/zoo/lstm_model.py` | Sequence model for time-series |

### Add a new model

Every model must implement the `BaseClassifier` contract in `src/models/base_classifier.py`:

```python
# src/models/zoo/gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
from src.models.base_classifier import BaseClassifier

class GradientBoostingClassifier_(BaseClassifier):

    def __init__(self, **params):
        self._model = GradientBoostingClassifier(**{**self.default_params, **params})

    @property
    def name(self) -> str:
        return "gradient_boosting"

    @property
    def default_params(self) -> dict:
        return {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
        }

    def fit(self, X, y):          self._model.fit(X, y)
    def predict(self, X):         return self._model.predict(X)
    def predict_proba(self, X):   return self._model.predict_proba(X)
```

Then add its hyperparameter search space in `src/models/tuner.py`:

```python
def _gbt_space(trial):
    return {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 400),
        "max_depth":     trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }

_PARAM_SPACES["gradient_boosting"] = _gbt_space
```

Finally register it in `dags/model_retrain_dag.py` by adding it to the `MODEL_ZOO` dict.

### Change hyperparameter search ranges

Each `_*_space` function in `src/models/tuner.py` defines what Optuna explores.
Example — widening the random forest search:

```python
def _rf_space(trial):
    return {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 1000),   # was 500
        "max_depth":       trial.suggest_int("max_depth", 3, 30),          # was 20
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 20),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    }
```

To change what the tuner optimises (currently `precision_buy`), edit the `objective()` return
value inside `tune()` in `src/models/tuner.py`.

### Change walk-forward window sizes

In `dags/model_retrain_dag.py`, look for the call to `walk_forward_train()`:

```python
walk_forward_train(
    df, model, feature_cols,
    train_window_days=500,   # 2 years of training history per fold
    test_window_days=21,     # 1-month test window
    step_days=21,            # fold advances monthly
)
```

Longer `train_window_days` → more history per fold, fewer folds overall.
Shorter `test_window_days` → more folds, noisier per-fold metrics.

### Train and evaluate a model interactively

```python
from pathlib import Path
import numpy as np
from src.features.duckdb_client import load_training_data
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.evaluator import evaluate
from src.models.registry import save_model

FEATURE_COLS = [
    "sma_10","sma_20","sma_50","sma_200","ema_12","ema_26",
    "rsi_14","macd","macd_signal","macd_hist",
    "bb_upper","bb_lower","bb_width","atr_14","hist_vol_21",
    "sent_pos_avg_3d","sent_pos_avg_5d","sent_pos_avg_10d",
    "sent_pos_mom_3d","news_vol_spike","rel_strength_spy","vix_level",
]

df = load_training_data(Path("data/features"))
X = df.select(FEATURE_COLS).to_numpy()
y = df["label"].to_numpy()

clf = RandomForestClassifier_(n_estimators=100)
clf.fit(X[:3200], y[:3200])

result = evaluate(y[3200:], clf.predict(X[3200:]))
print(f"precision_buy={result.precision_buy:.3f}  f1_macro={result.f1_macro:.3f}")

save_model(clf, result, clf.default_params, FEATURE_COLS, Path("data/registry"))
```

---

## Part 4 — Grading System

**File:** `src/backtesting/grader.py`

### How the grade is computed

```python
score = (
    0.40 * metrics.precision_buy                              # signal accuracy
    + 0.30 * tanh(metrics.sharpe_ratio / 2)                  # risk-adjusted return
    + 0.30 * (1.0 - min(metrics.max_drawdown_pct, 0.50)/0.50) # capital preservation
)
```

| Score range | Grade |
|---|---|
| ≥ 0.65 | A |
| ≥ 0.50 | B |
| ≥ 0.35 | C |
| < 0.35 | D |

### Change grade weights

Open `src/backtesting/grader.py`, edit `grade_model()`. Weights must sum to 1.0:

```python
score = (
    0.30 * metrics.precision_buy                       # reduced — less focus on raw precision
    + 0.25 * _norm_sharpe(metrics.sharpe_ratio)
    + 0.25 * (1.0 - _norm_drawdown(metrics.max_drawdown_pct))
    + 0.20 * metrics.win_rate                          # new — win rate now matters
)
```

### Change grade thresholds

Edit the `if/elif` block at the end of `grade_model()`:

```python
if score >= 0.70:     # raise the bar for A
    grade = Grade.A
elif score >= 0.55:
    grade = Grade.B
elif score >= 0.40:
    grade = Grade.C
else:
    grade = Grade.D
```

### All metrics available for grading

`BacktestMetrics` in `src/backtesting/metrics.py` already computes all of these:

```
n_trades          win_rate           profit_factor
total_return_pct  sharpe_ratio       max_drawdown_pct
precision_buy     recall_buy         f1_buy
accuracy
```

Any of them can be added to the grade formula immediately — no extra code needed.

### Add a new metric

Example — Calmar Ratio (annualized return / max drawdown):

Step 1: add to the `BacktestMetrics` dataclass in `src/backtesting/metrics.py`:

```python
@dataclass
class BacktestMetrics:
    ...
    calmar_ratio: float     # add this field
```

Step 2: compute it in `compute_metrics()` in the same file:

```python
annual_return = total_return * (252 / len(trades))
calmar = annual_return / max_dd if max_dd > 0 else 0.0
return BacktestMetrics(..., calmar_ratio=calmar)
```

Step 3: use it in `grade_model()`:

```python
score = (
    0.35 * metrics.precision_buy
    + 0.25 * _norm_sharpe(metrics.sharpe_ratio)
    + 0.25 * (1.0 - _norm_drawdown(metrics.max_drawdown_pct))
    + 0.15 * min(metrics.calmar_ratio / 3.0, 1.0)   # normalise calmar to [0,1]
)
```

### Grade a model interactively

```python
from pathlib import Path
from src.models.registry import load_model
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics

model = load_model("xgboost", Path("data/registry"))

# Replace with real backtest output once you have live data
metrics = BacktestMetrics(
    n_trades=120, win_rate=0.55, profit_factor=1.4,
    total_return_pct=0.18, sharpe_ratio=1.6, max_drawdown_pct=0.09,
    precision_buy=0.62, recall_buy=0.48, f1_buy=0.54, accuracy=0.67,
)
grade = grade_model(model.name, metrics)
print(f"Grade: {grade.grade.value}  Score: {grade.composite_score}")
```

---

## Part 5 — Dashboard Pages

**Files:** `dashboard/pages/`

| Page | File | What it shows |
|---|---|---|
| Data Overview | `1_Data_Overview.py` | Price chart + SMA20, sentiment bar |
| Model Leaderboard | `2_Model_Leaderboard.py` | A–D grade table, precision / Sharpe bars |
| Backtest Results | `3_Backtest_Results.py` | Walk-forward fold precision, Sharpe, trade count |
| Live Signals | `4_Live_Signals.py` | Confidence-filtered Buy signals with SHAP explanations |

To add a new page, create `dashboard/pages/5_My_Page.py`.
Streamlit picks it up automatically from the filename — no registration needed.

Cached data loaders are in `dashboard/data_loader.py`. All use `@st.cache_data(ttl=...)`.
Lower the TTL to see changes from a live retrain session immediately.

---

## Quick Reference — Where to Touch for Each Goal

| Goal | File(s) |
|---|---|
| Add a price/volatility indicator | `src/features/technical_indicators.py` + `FEATURE_COLS` in `dashboard/config.py` |
| Add a sentiment indicator | `src/features/sentiment_features.py` + same `FEATURE_COLS` registration |
| Add a cross-asset feature | `src/features/cross_asset_features.py` + same `FEATURE_COLS` registration |
| Add a new data feed | new file in `src/ingestion/`, add to fallback chain in `collector.py` |
| Add a new model | new file in `src/models/zoo/`, space in `tuner.py`, entry in retrain DAG |
| Change prediction horizon | `src/features/label_generator.py` → `forward_days` |
| Change Buy/Sell thresholds | `src/features/label_generator.py` → `buy_threshold`, `sell_threshold` |
| Change grade weights | `src/backtesting/grader.py` → `grade_model()` score formula |
| Change grade thresholds (A/B/C/D) | `src/backtesting/grader.py` → `if/elif` block |
| Add a new grade metric | `src/backtesting/metrics.py` dataclass + `compute_metrics()`, then `grader.py` |
| Widen hyperparameter search | `src/models/tuner.py` → `_*_space` functions |
| Change optimisation target | `src/models/tuner.py` → `objective()` return value |
| Add a dashboard page | new `5_*.py` in `dashboard/pages/` |
| Change data cache TTL | `dashboard/data_loader.py` → `@st.cache_data(ttl=...)` |
