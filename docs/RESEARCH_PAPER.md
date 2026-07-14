# Research Paper: Financial Signal Platform — Model Design & Ensemble Strategy

**Author:** Harold  
**Started:** 2026-06-17  
**Last Updated:** 2026-07-13  

---

## Abstract

This document records the design rationale, architectural decisions, experimental thoughts, and future research directions for an end-to-end equity signal platform targeting S&P 500 stocks. The platform ingests multi-source market data, engineers technical and sentiment features, trains a zoo of ML classifiers, backtests signals with walk-forward validation, and surfaces live Buy/Hold/Sell signals through a Streamlit dashboard.

---

## Project Description

**One-liner:** End-to-end equity signal platform for S&P 500 stocks covering survivorship-bias-free data ingestion, technical and sentiment feature engineering, multi-model ML classification, walk-forward backtesting, and live Buy/Hold/Sell signal generation with a Streamlit leaderboard dashboard.

**Key Features:**
- Multi-source OHLCV ingestion with automatic fallback chain (yFinance → Alpha Vantage → FMP)
- Survivorship-bias-free index composition via versioned CSV snapshots
- **92-ticker universe** covering mega-cap tech, financials, healthcare, consumer, energy, industrials, media/telecom, cybersecurity — data spanning **1990-01-01 to present** (~36 years, ~700K rows)
- 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, HistVol) + sentiment rolling features + cross-asset context (SPY relative strength, VIX)
- **26 strategies** across four families: individual rule-based, combination rule-based, and statistical (linear/ML-based)
- 8-model zoo (LogisticRegression, RandomForest, XGBoost, LightGBM, SVM, NaiveBayes, MLP, LSTM) sharing a common `BaseClassifier` interface; XGBoost leads at acc=44.6%, prec_buy=39.9%
- Walk-forward trainer with Optuna hyperparameter tuning maximizing Buy-class precision
- Backtest engine with confidence-threshold trade simulation and A–D composite grading (precision × Sharpe × drawdown)
- Airflow DAGs for daily ingestion, feature engineering, and weekly model retraining
- dbt lineage + DuckDB Parquet export for reproducible ML training pipelines
- Streamlit dashboard deployed on Render: Data Overview, Model Leaderboard, Backtest Results, Live Signals; cache pre-computed locally and committed to git for instant cold-start

**Data Types:**
- OHLCV price bars (daily, TimescaleDB hypertables)
- News articles with headline sentiment scores
- Engineered features (technical, sentiment, cross-asset, labels)
- Fundamental ratios
- Corporate actions and index compositions
- Model artifacts (joblib + JSON metadata, ISO-8601 versioned)
- Backtest trade logs and performance metrics (Sharpe, drawdown, win rate, profit factor)
- Live signals with confidence scores and position sizes

---

## Architecture

```
Plan 1 — Ingestion
  ↓  raw OHLCV + news → PostgreSQL / TimescaleDB
Plan 2 — Feature Engineering
  ↓  22 features + Buy/Hold/Sell labels → Parquet / DuckDB
Plan 3 — Training & Tuning
  ↓  8 models, Optuna search → joblib registry
Plan 4 — Backtest + Signals + Dashboard
  ↓  A–D grades, live signals → Streamlit at localhost:8501
```

**Label Schema:** 5-day forward return → Buy (>+2%) / Hold / Sell (<-2%)  
**Grading Formula:** `precision_buy × 0.40 + tanh(sharpe/2) × 0.30 + (1 − norm_drawdown) × 0.30`

---

## Current Strategies

26 strategies are active across three families. All are walk-forward backtested with a 400-day training window, 21-day test window, and 21-day step.

### Individual Rule-Based (18)
| ID | Strategy | Key Parameters | Signal Logic |
|----|----------|----------------|--------------|
| 1  | MA Crossover | SMA 20 / 50 | Fast SMA > Slow SMA |
| 2  | RSI Threshold | period=14, oversold=30 | RSI < 30 |
| 3  | MACD Signal | 12/26/9 | Histogram > 0 |
| 4  | Bollinger Bounce | window=20, 2σ | %B < 0.20 |
| 5  | Stochastic Oscillator | k=14, d=3, oversold=20 | %K < 20 and %K > %D |
| 6  | CCI | period=20, threshold=−100 | CCI < −100 |
| 7  | Williams %R | period=14, threshold=−80 | %R < −80 |
| 8  | Donchian Breakout | period=20 | Close > prior 20-day high |
| 9  | OBV Trend | obv_window=20, price_window=5 | OBV above MA + price below MA |
| 10 | Mean Reversion | window=60, z=−1.5 | Z-score < −1.5 |
| 11 | Momentum | period=20, threshold=3% | 20-day ROC > 3% |
| 12 | Triple MA Filter | 10 / 50 / 200 | SMA10 > SMA50 > SMA200 |
| 13 | Parabolic SAR | AF 0.02/0.02/0.20 | Price flips above trailing SAR |
| 14 | Keltner Breakout | EMA=20, ATR=14, mult=2.0 | Close > EMA + 2×ATR |
| 15 | DEMA Crossover | fast=10, slow=30 | DEMA(10) > DEMA(30) |
| 16 | Elder Ray | period=13 | Bull Power > 0, Bear Power rising |
| 17 | PVT Trend | pvt_window=20, price_window=5 | PVT above MA + price dip |
| 18 | ATR Breakout | lookback=10, 0.5×ATR | Close > prior 10-day high + 0.5×ATR |

### Combination Rule-Based (4)
| ID | Strategy | Components | Logic |
|----|----------|------------|-------|
| 19 | RSI + MACD Combo | RSI(14), MACD(12/26/9) | RSI < 35 AND histogram > 0 |
| 20 | Bollinger + RSI Combo | BB(20,2σ), RSI(14) | %B < 0.25 AND RSI < 45 |
| 21 | MA + Volume Confirm | EMA(20/50), volume | EMA uptrend AND volume > 1.5× avg |
| 22 | Trend Pullback | Triple MA + RSI | SMA10>SMA50>SMA200 AND RSI in 35–50 |

### Statistical (4)
| ID | Strategy | Model | Data Source |
|----|----------|-------|-------------|
| 23 | Logistic Regression | sklearn LogisticRegression (C=1.0) | Feature set |
| 24 | Linear Regression | sklearn LinearRegression | Feature set (return forecast) |
| 25 | Ridge Regression | sklearn Ridge (α=1.0) | Feature set (return forecast) |
| 26 | Gradient Boosting | HistGradientBoostingClassifier (100 iter) | Feature set |

### ML Model Zoo (inference only — not walk-forward strategy)
| Model | Accuracy | Prec (Buy) | F1 Macro |
|-------|----------|------------|----------|
| Random Forest | 40.9% | 36.2% | 38.5% |
| XGBoost | **44.6%** | **39.9%** | 33.7% |
| LightGBM | 39.7% | 34.6% | **38.1%** |

---

## Research Timeline

### 2026-06-17 — Core Architecture Planned
- Wrote Plans 1–4 covering all four layers of the platform
- Defined `BaseClassifier` interface shared by all 8 models
- Designed walk-forward trainer and A–D composite grader
- Designed SHAP explainability layer (TreeExplainer / LinearExplainer / KernelExplainer)

### 2026-06-18 — Extensions Beyond Plans
- Added Top-20 scrape pipeline (news/sentiment for top movers)
- Added simple strategy replacement subsystem: rule-based (MA, RSI, MACD, Bollinger) and statistical (logistic, linear regression)
- Deployed to Render (render.yaml, pyyaml fix, PYTHONPATH fix)

### 2026-07-07 — Project Review & Ensemble Research
- Conducted full project review: confirmed all planned modules are fully implemented
- Identified one open gap: intraday data fallback (AV and FMP cover daily only)
- Began research into model combination / ensemble methods (see section below)

### 2026-07-12 — Universe & Data Range Expansion
- Expanded stock universe from 20 → **87 tickers** across 7 sectors: mega-cap tech, financials, healthcare, consumer/retail, energy/industrials, media/telecom, cybersecurity/cloud
- Extended historical data range: `2000-01-01 → 2020-12-31` → `1990-01-01 → today` (~36 years per ticker, up to 9,197 rows each)
- Added 20 new strategies (stochastic, CCI, Williams %R, Donchian, OBV trend, mean reversion, momentum, triple MA, ridge regression, HistGradientBoosting), bringing total to 16
- Fixed `dashboard/config.py` missing `REGISTRY_DIR`; fixed `GradientBoostingClassifier` NaN issue by switching to `HistGradientBoostingClassifier`
- Retrained RF/XGBoost/LightGBM on 550K+ rows; XGBoost leads at acc=44.6%, prec_buy=39.9%
- Refreshed all dashboard cache; committed and deployed to Render

### 2026-07-13 — Second Strategy Expansion + Dashboard Fix
- Added 10 more strategies (combinations and new indicators), bringing total to **26 strategies**:
  - New combinations: RSI+MACD, Bollinger+RSI, MA+Volume, Trend Pullback
  - New methods: Parabolic SAR, Keltner Breakout, DEMA Crossover, Elder Ray, PVT Trend, ATR Breakout
- Expanded stock universe to **92 tickers** (data_summary confirmed)
- Fixed critical cache-stale bug: `precompute_dashboard.py` was calling `get_data_summary()` which hit the old cache and wrote it back unchanged; fixed by reading parquet files directly in the precompute step
- Data Overview now correctly shows 92 tickers, 1990–2026 range
- All 26 strategy backtest cache files committed and deployed

---

## Research Thread: Model Combination & Ensemble Methods

**Date:** 2026-07-07  
**Question:** What simple method combinations can form a stronger model? Can high-similarity models be combined?

### Core Insight: Diversity Beats Similarity

The fundamental principle of ensemble learning:

> **Ensemble error = average individual error − benefit from disagreement.**  
> If two models always agree (high similarity / high correlation), combining them gains nothing — you are duplicating the same mistakes. The benefit comes from diverse models that are *wrong on different samples*.

High model correlation → small ensemble gain.  
Low model correlation → large ensemble gain.

**Exception:** Identical algorithms trained on different data subsets (bagging) still reduce variance even though the method is the same. This is how Random Forest works internally.

### Method Survey

#### 1. Soft Voting / Probability Averaging
Run multiple models, average their `P(Buy)` probabilities, then classify.
```
ensemble_prob = mean([p_lr, p_rf, p_xgb, ...])
signal = "Buy" if ensemble_prob > threshold else "Hold"
```
- Simplest approach
- Works best when models have different failure modes
- Easily implemented as a new `BaseClassifier`-compatible wrapper

#### 2. Hard Voting
Majority class wins. Less expressive than soft voting because it discards probability information.

#### 3. Stacking (Meta-Learning)
Use base model predictions as features for a second-level meta-learner:
```
Layer 1: RF, XGB, LGBM each output P(Buy)
Layer 2: Logistic Regression takes [p_rf, p_xgb, p_lgb] → final label
```
The meta-learner learns *which base model to trust* in which market conditions. More powerful than voting; requires a held-out validation set for training the meta-learner.

#### 4. Blending
Simplified stacking — train base models on 80% of data, generate predictions on held-out 20%, train blender on those predictions. Avoids data leakage more simply than full stacking.

#### 5. Bagging (already embedded)
Train same model on different bootstrap samples. Reduces variance. Already used internally by RandomForest.

#### 6. Boosting (already embedded)
Train models sequentially, each correcting errors of the previous. Already used by XGBoost and LightGBM.

### Practical Recommendations for This Platform

| Approach | Effort | Expected Gain | Notes |
|----------|--------|---------------|-------|
| Soft vote: RF + XGB + LGBM | Low | Solid | Tree models are correlated but have different hyperparameter regimes |
| Soft vote: RF + LR + NaiveBayes | Low | Better diversity | Linear + tree + probabilistic = different failure modes |
| Stack: (RF, XGB, LGBM) → LR meta | Medium | Usually beats raw voting | LR meta is conservative, good for precision-focused grader |
| Stack: all 8 models → LR meta | Medium | Diminishing returns past ~5 | Too many correlated base models |

**Recommended first experiment:** Soft vote over RF + LR + NaiveBayes (maximum diversity across model families).  
**Recommended second experiment:** Stack (RF, XGB, LGBM) → Logistic Regression meta with `class_weight='balanced'`.

---

## Open Questions / Future Research Directions

- [x] **Strategy expansion:** Added 20 new strategies (total 26), covering oscillators, breakout, volume, combination, and statistical families *(completed 2026-07-13)*
- [x] **Universe expansion:** Grew from 20 → 92 tickers across 7 sectors *(completed 2026-07-13)*
- [x] **Historical depth:** Extended from 2000–2020 to 1990–present *(completed 2026-07-12)*
- [ ] **Ensemble implementation:** Add `VotingEnsemble` and `StackingEnsemble` as new `BaseClassifier`-compatible wrappers in the model zoo
- [ ] **Correlation audit:** Compute pairwise prediction correlation across the 8 models on walk-forward validation sets to identify the most complementary pairs
- [ ] **Intraday fallback:** AV and FMP only cover daily resolution — intraday data fallback path is incomplete
- [ ] **Feature selection:** Which of the 22 features contribute most per model? (SHAP already wired; run aggregation)
- [ ] **Signal calibration:** Are predicted probabilities well-calibrated? Platt scaling or isotonic regression as a post-processing step
- [ ] **Regime detection:** Do models degrade in bear markets vs bull markets? Add market-regime conditioning (e.g., VIX threshold, SPY 200-day MA)
- [ ] **Walk-forward ensemble:** Re-stack ensemble predictions through the walk-forward backtest and compare A–D grades against individual models
- [ ] **Per-strategy grade summary:** Aggregate and rank all 26 walk-forward backtests by composite score; identify top-3 strategies per sector

---

## Key Design Decisions (Recorded)

| Decision | Rationale |
|----------|-----------|
| Buy-class precision as primary Optuna objective | Minimizes false Buy signals; better for a long-only strategy |
| 5-day forward return label | Short enough to capture momentum, long enough to filter noise |
| >+2% / <-2% thresholds | Filters micro-moves; forces the model to identify meaningful moves |
| Walk-forward (not simple train/test split) | Prevents look-ahead; mimics real deployment conditions |
| Composite A–D grader | Balances raw signal quality (precision) with risk-adjusted return (Sharpe) and downside protection (drawdown) |
| Polars for all DataFrames | 3–10× faster than Pandas for feature engineering at S&P 500 scale |
| Survivorship-bias-free composition | Prevents the model from only ever seeing winning stocks |
| All API keys from environment only | Security; required for Render deployment |
| UTC timestamps everywhere | Avoids DST ambiguity across global data sources |
