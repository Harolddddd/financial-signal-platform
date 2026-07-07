# Research Paper: Financial Signal Platform — Model Design & Ensemble Strategy

**Author:** Harold  
**Started:** 2026-06-17  
**Last Updated:** 2026-07-07  

---

## Abstract

This document records the design rationale, architectural decisions, experimental thoughts, and future research directions for an end-to-end equity signal platform targeting S&P 500 stocks. The platform ingests multi-source market data, engineers technical and sentiment features, trains a zoo of ML classifiers, backtests signals with walk-forward validation, and surfaces live Buy/Hold/Sell signals through a Streamlit dashboard.

---

## Project Description

**One-liner:** End-to-end equity signal platform for S&P 500 stocks covering survivorship-bias-free data ingestion, technical and sentiment feature engineering, multi-model ML classification, walk-forward backtesting, and live Buy/Hold/Sell signal generation with a Streamlit leaderboard dashboard.

**Key Features:**
- Multi-source OHLCV ingestion with automatic fallback chain (yFinance → Alpha Vantage → FMP)
- Survivorship-bias-free index composition via versioned CSV snapshots
- 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, HistVol) + sentiment rolling features + cross-asset context (SPY relative strength, VIX)
- 8-model zoo (LogisticRegression, RandomForest, XGBoost, LightGBM, SVM, NaiveBayes, MLP, LSTM) sharing a common `BaseClassifier` interface
- Walk-forward trainer with Optuna hyperparameter tuning maximizing Buy-class precision
- Backtest engine with confidence-threshold trade simulation and A–D composite grading (precision × Sharpe × drawdown)
- Rule-based strategy layer (MA crossover, RSI, MACD, Bollinger) alongside statistical strategies
- Airflow DAGs for daily ingestion, feature engineering, and weekly model retraining
- dbt lineage + DuckDB Parquet export for reproducible ML training pipelines
- Streamlit dashboard: Data Overview, Model Leaderboard, Backtest Results, Live Signals

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

### Rule-Based
| ID | Strategy | Logic |
|----|----------|-------|
| 1 | MA Crossover | Short SMA crosses above long SMA → Buy |
| 2 | RSI | Below 30 → Buy; Above 70 → Sell |
| 3 | MACD | MACD line crosses signal line |
| 4 | Bollinger Bands | Price touches/crosses lower band → Buy |

### Statistical
| ID | Strategy | Logic |
|----|----------|-------|
| 5 | Logistic Regression | Probability-based 3-class classification |
| 6 | Linear Regression | Return forecast thresholded to signal |

### ML Model Zoo
| ID | Model | Type |
|----|-------|------|
| 7  | LogisticRegression | Linear |
| 8  | RandomForest | Tree ensemble (bagging) |
| 9  | XGBoost | Tree ensemble (boosting) |
| 10 | LightGBM | Tree ensemble (boosting) |
| 11 | SVM | Kernel-based |
| 12 | NaiveBayes | Probabilistic |
| 13 | MLP | Neural network |
| 14 | LSTM | Sequential neural network |

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

- [ ] **Ensemble implementation:** Add `VotingEnsemble` and `StackingEnsemble` as new `BaseClassifier`-compatible wrappers in the model zoo
- [ ] **Correlation audit:** Compute pairwise prediction correlation across the 8 models on walk-forward validation sets to identify the most complementary pairs
- [ ] **Intraday fallback:** AV and FMP only cover daily resolution — intraday data fallback path is incomplete
- [ ] **Feature selection:** Which of the 22 features contribute most per model? (SHAP already wired; run aggregation)
- [ ] **Signal calibration:** Are predicted probabilities well-calibrated? Platt scaling or isotonic regression as a post-processing step
- [ ] **Regime detection:** Do models degrade in bear markets vs bull markets? Add market-regime conditioning
- [ ] **Walk-forward ensemble:** Re-stack ensemble predictions through the walk-forward backtest and compare A–D grades against individual models

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
