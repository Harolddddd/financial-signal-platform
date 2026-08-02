# Sell Signal Recommendation — Design

## Problem

The platform's 32 strategies classify every ticker into Buy/Hold/Sell (the `Signal` enum defines all three, and training labels are genuinely 3-class: `forward_return_5d > +2% → Buy`, `< -2% → Sell`, else `Hold`), but **no strategy ever actually emits a Sell signal today**. Every classifier-based strategy's `predict()` only checks the Buy-class probability (`buy_idx = classes.index("Buy")`) and collapses everything else — including cases where the model is confident about Sell — into Hold. The 20 rule-based strategies were never given sell conditions at all. Confirmed empirically: `markets/{us,china}/data/cache/signals.json` contain zero `"signal": "Sell"` records across both markets' full history.

The user wants to select a stock they own (ticker + optional buy price/date) and get a sell/hold recommendation, driven by the strategies — which first requires the strategies to be capable of saying "Sell" at all.

## Goals

- Fix the strategies that can meaningfully support a Sell classification so they actually emit one.
- Extend the existing per-ticker signal-aggregation logic to compute a sell-side score alongside the existing buy-side one, using the same weighting scheme already in production (Combined Signal page).
- A new dashboard page: pick a ticker (in the currently-selected market), optionally enter what you paid and/or when you bought, see today's sell/hold recommendation plus unrealized P&L.
- Regenerate real backtest/leaderboard/signal data for both markets so the fix is reflected everywhere, not just in new code paths.

## Non-Goals

- No changes to the 20 rule-based strategies (`src/strategies/rule_based/`) — they have no natural sell condition and stay Buy/Hold-only, exactly like today.
- No changes to `isolation_forest_strategy.py` — it's an anomaly-buy detector, not a 3-class classifier; no natural sell condition either.
- No portfolio/position persistence, no login, no database — buy price/date are per-session form inputs only, matching this platform's existing no-auth architecture.
- No changes to the training pipeline itself (`add_labels()`, feature engineering, model architectures) — the 3-class labels already exist and are already used for training; this only fixes the *prediction*-time collapse to Buy/Hold.
- No changes to `train_new_models.py`'s separate US-only zoo (logistic_regression/naive_bayes/mlp/svm/lstm as *models*, distinct from the same-named *strategies* under `src/strategies/statistical/`) — out of scope, unrelated code path.

## Architecture

### 1. Strategy fix — 11 files under `src/strategies/statistical/`

**10 classifier-based strategies** share an identical current pattern (verified by reading all 10): `logistic.py`, `gaussian_nb_strategy.py`, `lda_strategy.py`, `knn_classifier.py`, `mlp_classifier.py`, `gradient_boosting.py`, `extra_trees.py`, `catboost_strategy.py`, `xgboost_strategy.py`, `random_forest.py`. Each gets the same symmetric change to `predict()`:

```python
# before (every one of the 10 files, same shape):
if "Buy" not in classes:
    ...
buy_idx = classes.index("Buy")
confidence = pd.Series(proba[:, buy_idx])
signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
return PredictionResult(confidence=confidence, signal=signal)

# after:
buy_idx = classes.index("Buy") if "Buy" in classes else None
sell_idx = classes.index("Sell") if "Sell" in classes else None
buy_p = proba[:, buy_idx] if buy_idx is not None else np.zeros(len(df))
sell_p = proba[:, sell_idx] if sell_idx is not None else np.zeros(len(df))

signal = []
confidence = []
for b, s in zip(buy_p, sell_p):
    if b >= 0.6 and b >= s:
        signal.append("Buy"); confidence.append(b)
    elif s >= 0.6 and s > b:
        signal.append("Sell"); confidence.append(s)
    else:
        signal.append("Hold"); confidence.append(b)  # unchanged Hold-row convention: report Buy-probability
return PredictionResult(confidence=pd.Series(confidence), signal=pd.Series(signal))
```

`confidence` keeps meaning "confidence in the emitted signal" — Buy-probability when the signal is Buy, Sell-probability when the signal is Sell. For Hold rows, `confidence` continues to report the Buy-class probability, matching every existing cached Hold record today (no behavior change for the already-shipped Hold convention — downstream consumers that read `confidence` on Hold rows, if any exist, keep seeing what they've always seen).

**`linear.py`** (regression, not a classifier — predicts a continuous return, no `classes_`) gets a separate, symmetric extension:

```python
# before:
signal = pd.Series(["Buy" if r >= self.buy_threshold else "Hold" for r in pred_return])

# after:
def _label(r: float) -> str:
    if r >= self.buy_threshold:
        return "Buy"
    if r <= self.sell_threshold:
        return "Sell"
    return "Hold"
signal = pd.Series([_label(r) for r in pred_return])
```

`self.sell_threshold` is a new constructor parameter, defaulting to `-self.buy_threshold` (symmetric around zero, matching the training label generator's own `buy_threshold=0.02` / `sell_threshold=-0.02` convention in `src/features/label_generator.py`).

**`isolation_forest_strategy.py`**: untouched — no `classes_`, no continuous return, purely an anomaly-plus-RSI-oversold Buy heuristic. Stays Buy/Hold-only.

### 2. Data layer — `dashboard/data_loader.py`

`get_combined_ratings(market)` already assembles, per ticker, every strategy's `(signal, confidence, weight)` row (`by_ticker` dict, built from `signals.json` + `leaderboard.json`). Today it only accumulates a buy-side `weighted_buy` / `overall_rating`. Add a parallel sell-side accumulation using the exact same rows and weights:

```python
weighted_sell = 0.0
...
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
    ...
summary_rows.append({
    ...,  # existing fields unchanged
    "sell_rating": 100.0 * weighted_sell / total_weight,
    "net_rating": 100.0 * (weighted_sell - weighted_buy) / total_weight,
})
```

`overall_rating` (existing, unchanged) stays purely buy-side, for the existing Combined Signal page. `sell_rating` and `net_rating` are additive — no existing caller's output shape changes for fields it already reads.

### 3. New page — `dashboard/pages/6_Should_I_Sell.py`

- Ticker selector scoped to the current market (same source as Data Overview's ticker list).
- Optional "Buy price" number input and "Buy date" date input. If both are given, price takes precedence for P&L math (date is used only to resolve a price when price is blank). If neither is given, the page still shows the recommendation, just no P&L section.
- Buy-date → price resolution: load that ticker's historical OHLCV (same `load_training_data(parquet_dir, tickers=[ticker])` call Data Overview already uses), take the closing price of the last trading day at or before the given date. If the given date is before the ticker's earliest available data, or in the future, show an explicit error message and skip P&L (don't silently guess).
- "Current price": the ticker's `entry_price` from its signal rows (today's close, already cached — every per-strategy row for a ticker carries the same value).
- P&L: `(current_price - buy_price) / buy_price * 100`, shown as both a % and an absolute value in the market's currency (`format_price()` from `dashboard/market_state.py`).
- Recommendation label, from `net_rating` (mirrors Combined Signal's existing 4/8/12 bucket convention, `dashboard/pages/5_Combined_Signal.py`'s `_label()`):
  - `net_rating >= 12` → **"Strong Sell"**
  - `4 <= net_rating < 12` → **"Consider Selling"**
  - `-4 < net_rating < 4` → **"Hold"**
  - `net_rating <= -4` → **"Keep Holding (Bullish)"**
- Per-strategy vote breakdown table (same shape as Combined Signal's existing drill-down: strategy, weight, signal, confidence, contribution), reusing `detail_by_ticker[ticker]` from `get_combined_ratings()` directly — no new per-strategy lookup needed.
- Market-aware throughout (ticker list, currency formatting, `get_selected_market()`), consistent with every other page after the market-switching plan.

### 4. Data regeneration

Changing `predict()` changes strategy output during **backtests** too (`walk_forward_backtest_strategy` calls the same `predict()`), not just live signals — so the 10 affected strategies' cached backtest results, and the leaderboard built from them, are now stale the moment the code changes, for both markets. Required re-runs, both markets:

```
python scripts/train_models.py --market <market>          # unaffected by this change but re-run for consistency
python scripts/precompute_dashboard.py --market <market>   # regenerates backtests, leaderboard, and live signals
```

**Known risk (carried forward from the prior plan):** `precompute_dashboard.py --market china`'s live-signals step (`step_signals`) previously crashed repeatedly with a Rust/Polars allocator panic at full 500-ticker scale, root cause never fully pinned down. The prior plan's workaround — computing signals one ticker at a time instead of the framework's all-tickers-then-concat/predict approach — is the known-working path. Go in expecting to need it again for China rather than attempting the naive full-scale run first and discovering the crash again; for US (which has never shown this issue, and 492 tickers vs China's 500), a normal full run is expected to work.

## Data Flow

1. Strategy `predict()` now genuinely returns Sell for 10 classifier strategies + `linear_regression`, alongside existing Buy/Hold.
2. `step_backtests()`/`step_leaderboard()`/`step_signals()` (`scripts/precompute_dashboard.py`, unchanged code) pick this up automatically on the next real run — the fix is entirely inside the strategies, the pipeline code doesn't need to change.
3. Regenerated `signals.json` now has real Sell records; regenerated `backtest_<name>.json`/`leaderboard.json` reflect the strategies' true 3-class behavior.
4. `get_combined_ratings(market)` (extended) computes both `overall_rating` (buy-side, existing) and `sell_rating`/`net_rating` (new) from the same per-ticker rows.
5. The new page calls `get_combined_ratings(market)`, looks up the one selected ticker's summary + detail rows, combines with the optional buy price/date for P&L, and renders the recommendation.

## Error Handling

- Ticker has no signal data at all (shouldn't happen given 90%+ coverage floors from prior plans, but defensively): show "no data for this ticker" rather than crashing.
- Buy date resolves to no available price (before earliest data / in the future): explicit error message, P&L section skipped, recommendation still shown.
- Buy price entered as zero or negative: reject with a validation message (division by zero in P&L math otherwise).
- No behavior change to existing error paths in `get_combined_ratings()` (already handles missing leaderboard/signals cache).

## Testing

- Unit tests per fixed strategy file: construct a small synthetic 3-class training set, fit, and confirm `predict()` can return `"Sell"` for an input engineered to score high Sell-probability (or, for `linear_regression`, a predicted-return input below `sell_threshold`) — mirroring this repo's existing `tests/unit/test_zoo_classical.py` style fit/predict tests. Confirm the unchanged Hold-row `confidence` convention with a case that should land on Hold.
- `dashboard/data_loader.py`: unit test `get_combined_ratings()`'s new `sell_rating`/`net_rating` fields against real committed cache data once regenerated (same no-mocking convention as the market-switching plan's tests) — construct a known small-scale expectation or assert structural properties (fields present, `0 <= sell_rating <= 100`, `-100 <= net_rating <= 100`) since exact values depend on the live regenerated data.
- New page: source-text regression tests matching the established per-page pattern (`market`-aware, no hardcoded market list, etc.), plus an `AppTest`-based check that the page renders without exception for a ticker with a known Sell-leaning `net_rating` in each market, given the regenerated data.
- Full suite run after regeneration, comparing against the pre-existing 15-failure baseline for regressions.
- Manual verification: run the dashboard, pick a real ticker in each market, confirm the recommendation and P&L math are sane.
