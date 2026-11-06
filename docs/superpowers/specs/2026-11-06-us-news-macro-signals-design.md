# US News + Fed/Macro Signals — Design

Date: 2026-11-06
Scope: US market only. China is out of scope for this project.

## Intent

The US pipeline today is purely price-driven. The sentiment features exist in the
schema but are hard-coded constants (`add_neutral_sentiment` in
`scripts/build_features.py`), and the only market context is SPY 5-day return
and VIX level. The user wants predictions that combine **stock news**, **Fed /
macro conditions**, and **price action**, and wants the pipeline to be **much
cheaper to run** than the current ~37 strategies × ~500 tickers walk-forward
(US `regime_consensus` alone took 12.5h on 2026-10-29).

## Goals

1. Real per-ticker news sentiment flows into US features daily.
2. Market-wide Fed/macro features (rates, curve, credit, FOMC tone) flow into US
   features daily.
3. Live US signals are adjusted by news and macro conditions from day one, even
   before enough news history exists to train on it.
4. US full-precompute wall time drops substantially (target: under 2h for the
   weekend retrain + backtest, vs ~16h today).
5. Once ~6–12 months of stored news exist, a single pooled model trained on
   price + news + macro features replaces the per-strategy models.

## Non-goals

- China market (no changes to China pipeline, features, or caches).
- Intraday / real-time trading. Cadence stays daily.
- Paid data feeds.
- Social-media sentiment (Reddit, X, Stocktwits).

## Anti-goals (failure even if it "works")

- **Look-ahead leakage.** News published after the US close on day T must not be
  used for a day-T signal. FOMC statements count only from their release time.
  Macro values must be ones known on that date.
- **Breaking the existing daily flow.** `incremental_train.py` + `--signals-only`
  on weekdays and full retrain on weekends must keep working (see project
  memory: daily = incremental only).
- **Faking sentiment again.** If news is missing for a ticker/day, the feature is
  null/neutral *and flagged*, not silently a constant across all history.

## Constraints and facts established

- **Finnhub free tier**: ~1 year of company-news history, ~60 calls/min,
  personal use, no sentiment scores (score locally). Code already exists:
  `collect_finnhub` in `src/ingestion/news_collector.py`. `FINNHUB_KEY` is in
  `config/settings.py` but not yet set in `.env`.
- **FRED API**: free key, daily series. Needs a new `FRED_KEY` setting.
- **FinBERT** (`ProsusAI/finbert`, CPU) scoring with VADER fallback already
  exists in `src/ingestion/sentiment_processor.py`.
- **Rolling sentiment features** already exist in
  `src/features/sentiment_features.py` (`sent_pos_avg_{3,5,10}d`,
  `sent_pos_mom_3d`, `news_vol_spike`); feature names are already in the model
  feature list.
- Long jobs must be launched via `scripts/run_job.py` (extension host OOM kills
  tool-launched jobs).
- Walk-forward steps default to 21 days.
- Backtests that use news are limited to ~1 year of history; price-only
  backtests keep their full history.

## Design

### 1. News ingestion (per ticker, daily)

- Source: Finnhub company news for each US ticker in `_STOCK_TICKERS`. Keep
  Yahoo RSS as a secondary source via existing `collect_all_news` dedupe by URL.
- One-time backfill: 1 year, chunked by month per ticker (~6k calls, ~100 min
  at 60/min; run via `run_job.py`).
- Daily incremental: previous day's articles only (~500 calls, ~9 min).
- Score each article **once** at ingest (FinBERT on headline + summary; VADER
  fallback) and store the score with the article. Re-runs never re-score.
- Storage: parquet under `markets/us/data/news/`, which matches the rest of the
  pipeline and avoids a running-DB dependency. The existing `news_articles`
  Timescale table is left as-is. Storage is kept small in three tiers
  (user requirement: minimise news storage):
  1. **Daily aggregates (permanent, the only long-term store)**:
     `sentiment_daily.parquet`, one row per (ticker, trading day): `avg_pos`,
     `avg_neg`, `avg_neu`, `article_count`. Features and Phase B training read
     only this. Size: ~500 tickers × ~252 days ≈ 126k rows/year, well under
     5 MB/year.
  2. **Article scores (rolling 60 days)**: `articles_scores_YYYY-MM.parquet`
     (monthly files, not daily, to avoid many small files): `ticker`
     (categorical), `published_at` (UTC), `url_hash` (int64 xxhash of the URL,
     for dedupe), `sentiment_pos/neg/neu` (float32), `scorer` (categorical).
     No headline, body, or full URL. Kept so aggregates can be rebuilt for
     late-arriving articles; months older than 60 days are deleted.
  3. **Headlines (rolling 14 days)**: `headlines_recent.parquet` holding
     `url_hash, ticker, published_at, headline` only, so the dashboard can
     show the news behind an overlay decision. Older rows are dropped each run.
  - Article body/summary text is never written to disk; it is scored in
    memory and discarded.
  - All parquet written with zstd compression and dictionary encoding.
  - Trade-off accepted: articles older than 60 days cannot be re-scored with a
    different model. Acceptable because Finnhub only offers ~1 year anyway;
    a re-score would require a fresh backfill.
  - Estimated total: roughly 10–20 MB steady state, vs ~1–2 GB/year if raw
    text were kept (~5k articles/day × ~1 KB). Builder should report actual
    sizes after backfill.
- Timestamp rule: an article belongs to trading day T if published before
  16:00 America/New_York on T; otherwise it rolls to T+1.

### 2. Fed / macro features (market-wide, daily, computed once)

- FRED daily series: `DFF` (fed funds), `DGS2`, `DGS10`, `T10Y2Y` (curve),
  `BAMLH0A0HYM2` (HY credit spread). Derived features: levels, 5d and 20d
  changes, and 20d z-scores.
- Monthly/revised series (e.g. CPI) are excluded in v1 to avoid revision
  leakage *(my call)*.
- Shift every macro value by one trading day (use only what was published
  before the open). *(my call: simplest leakage-safe rule)*
- **FOMC tone**: fetch statements from federalreserve.gov, score
  hawkish↔dovish with the FOMC-RoBERTa model from the "Trillion Dollar Words"
  paper (builder: confirm the Hugging Face model id; fall back to a
  sentence-level LLM or dictionary score if unavailable). Features:
  `fomc_tone` (carried forward until next meeting), `fomc_tone_change`, and
  `days_since_fomc`. Effective from the trading day after release.
- Computed once per day into `markets/us/data/macro/macro_daily.parquet` and
  joined onto every ticker by date, so per-ticker cost is a join.

### 3. Feature build

- *(Revised during planning.)* Nulls cannot be used: `walk_forward.py:53`,
  `incremental_train.py:53` and `step_signals` all drop rows with null
  features, which would delete all pre-news history. So:
  - The legacy constant columns (`sent_pos_avg_*`, `sent_pos_mom_3d`,
    `news_vol_spike`) and `FEATURE_COLS` stay **unchanged** in Phase A.
    Existing models and China see exactly what they see today.
  - US feature parquet gains new columns prefixed `news_` and `macro_` /
    `fomc_` (the `OVERLAY_COLS` list). Days without news get `news_count_* = 0`
    and `news_net_* = 0.0`, with `news_has_3d` as the explicit flag, so
    "no news" is visible rather than faked.
  - Phase B adds `OVERLAY_COLS` to the pooled model's feature list.
- Join macro features by date.

### 4. Phase A — overlay on existing signals (ships first)

- Prune the US strategy set to the top 4 by leaderboard composite score
  (2026-10-29 leaderboard): `extra_trees`, `lda_strategy`, `random_forest`,
  `conformal_lgbm` *(my call; drop `regime_consensus`, which is the slowest)*.
  Other strategies stay in the code but are not run by default for US.
- Overlay applied to final BUY/SELL signals and confidence:
  - **News veto/boost**: strongly negative recent news (e.g. 3d sentiment well
    below the ticker's norm, with a news-volume spike) suppresses BUY; strongly
    positive news modestly raises confidence.
  - **Macro risk-off**: hawkish FOMC shift, sharp 2y yield rise, or widening HY
    spread scales down BUY confidence market-wide.
  - Thresholds tuned on the ~1 year with news; builder picks the exact rules
    and must report backtest with vs without overlay on that window.
- Dashboard: show the overlay's effect and reason per signal (e.g. "BUY
  suppressed: negative news").

### 5. Phase B — pooled model (after ~6–12 months of stored news)

- One LightGBM trained pooled across all US tickers on price + sentiment +
  macro features, 21-day walk-forward. Weekly full retrain, daily incremental,
  matching current cadence.
- Promoted only if it beats the Phase A overlay ensemble on the same window.
- Not built in this project's first plan; the plan should leave the feature
  pipeline ready for it.

### Compute impact

| Change | Effect |
|---|---|
| 37+ → 4 US strategies | most of the saving |
| Score each article once, store it | daily scoring only for new articles |
| Macro computed once per day | negligible per ticker |
| Phase B single pooled model | replaces 4 models |

## Left to the builder

- Exact overlay thresholds and rule shapes, chosen on backtest evidence.
- FOMC statement scraping details and model id confirmation.
- Rate-limit/backoff handling for Finnhub.
- Where the overlay lives in code (likely `src/signals/signal_engine.py`).
- Tests for timestamp/leakage rules (required).

## Needed from the user before running

- A Finnhub API key and a FRED API key in `.env`.
