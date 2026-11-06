# US News + Fed/Macro Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real per-ticker news sentiment and market-wide Fed/macro data to the US pipeline, use them as an overlay on live signals, cut the US strategy set to 4, and store news in a small tiered format.

**Architecture:** Daily ingest writes compact news scores and macro series to parquet under `markets/us/data/{news,macro}/`. `build_features_for_ticker` joins them onto US features as new `news_*` / `macro_*` / `fomc_*` columns (`OVERLAY_COLS`), leaving `FEATURE_COLS` unchanged. An `OverlayStrategy` wrapper strips those columns before the inner model sees them, then adjusts Buy/confidence and records a reason.

**Tech Stack:** Python 3.12, polars, pandas, requests, finnhub-python, transformers (FinBERT, FOMC-RoBERTa), pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-us-news-macro-signals-design.md`

## Global Constraints

- US only. No change to China features, caches, or behavior; `add_neutral_sentiment` and `FEATURE_COLS` stay as they are.
- No look-ahead: an article published at or after 16:00 America/New_York on day T belongs to the next trading day; macro values are shifted 1 trading day; FOMC tone takes effect the trading day after release.
- News storage tiers: `sentiment_daily.parquet` permanent; `articles_scores_YYYY-MM.parquet` rolling 60 days; `headlines_recent.parquet` rolling 14 days; article body text is never written; all parquet written with `compression="zstd"`.
- No new dependencies: use `requests` for FRED, `hashlib.blake2b(digest_size=8)` for URL hashes.
- New settings: `FRED_KEY: str = ""` (plus the existing `FINNHUB_KEY`). Missing keys → log a warning and skip ingest, never crash the daily flow.
- Long runs (backfill, precompute, evaluation) go through `python scripts/run_job.py`; check with `--status`.
- Walk-forward `step_days=21`.
- Daily flow stays: `refresh_data.py` + `incremental_train.py` + `precompute_dashboard.py --signals-only` on weekdays; full retrain weekends.
- Work on a branch (`feat/us-news-macro`), not `main`. Commit only the files each task names, because the working tree has many unrelated uncommitted data files.

## Review Focus

1. Finnhub returns 429 or an empty list for a ticker: ingest backs off and retries, then skips that ticker; the run still completes. → test in Task 2.
2. Weekend/holiday articles (Saturday 10:00 ET): they land on the next trading day, not dropped and not on Friday. → test in Task 1.
3. Same article fetched twice (daily re-run, overlapping backfill chunks): stored once, counts unchanged. → test in Task 1.
4. Feature frame with no news file at all (first run, or a ticker never covered): `OVERLAY_COLS` present with neutral values, row count unchanged. → test in Task 4.
5. FRED series with gaps ("." values, holidays): forward-filled, never null after the first valid value. → test in Task 3.

---

### Task 1: Tiered news store

**Files:**
- Create: `src/ingestion/news_store.py`
- Test: `tests/unit/test_news_store.py`

**Interfaces:**
- Produces:
  - `url_hash(url: str) -> int`: signed int64 from `blake2b(url.encode(), digest_size=8)`.
  - `assign_trading_day(published_at: pl.Series, trading_days: pl.Series) -> pl.Series`: date per article, using the 16:00 ET cutoff, then the first trading day >= that date (`join_asof`, `strategy="forward"`).
  - `write_scores(news_dir: Path, scores: pl.DataFrame) -> int`: scores columns `ticker, published_at, url_hash, sentiment_pos, sentiment_neg, sentiment_neu, scorer`. Merges into monthly files, dedupes on `(ticker, url_hash)`, returns the count of new rows.
  - `write_headlines(news_dir: Path, rows: pl.DataFrame, today: date, keep_days: int = 14) -> None`: columns `url_hash, ticker, published_at, headline`.
  - `rebuild_daily(news_dir: Path, trading_days: pl.Series, since: date) -> None`: recomputes `sentiment_daily.parquet` rows with `time >= since` from the score files and keeps older rows. Columns: `ticker, time (date), avg_pos, avg_neg, avg_neu, article_count`.
  - `prune(news_dir: Path, today: date, keep_days: int = 60) -> None`: deletes monthly score files whose whole month is older than `today - keep_days`.
  - `load_daily(news_dir: Path) -> pl.DataFrame`: returns an empty frame with the schema above if the file is missing.

- [ ] **Step 1: Write failing tests**
  - `test_url_hash_stable_and_int64`: same URL gives the same value across calls; value fits in int64.
  - `test_assign_trading_day_cutoff_and_weekend`: trading days Fri 2026-10-30 and Mon 2026-11-02. Fri 15:59 ET → 10-30; Fri 16:00 ET → 11-02; Sat 10:00 ET → 11-02.
  - `test_write_scores_dedupes`: writing the same 3 rows twice returns 3 then 0; the file holds 3 rows.
  - `test_rebuild_daily_aggregates_and_keeps_old`: a pre-existing daily row before `since` survives; 2 articles for AAPL on one day give `article_count == 2` and the mean `avg_pos`.
  - `test_prune_keeps_recent_months` and `test_headlines_rolls_off_after_14_days`.
  - `test_files_use_zstd`: `pyarrow.parquet.ParquetFile(...).metadata.row_group(0).column(0).compression == "ZSTD"`.
- [ ] **Step 2: Run** `pytest tests/unit/test_news_store.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `src/ingestion/news_store.py` with the signatures above. Read and write with polars `write_parquet(..., compression="zstd")`; store `ticker` and `scorer` as `pl.Categorical` and the sentiments as `pl.Float32`.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** `git add src/ingestion/news_store.py tests/unit/test_news_store.py && git commit -m "feat: tiered compact news store"`

### Task 2: News ingest (Finnhub → score → store), backfill, daily hook

**Files:**
- Modify: `src/ingestion/sentiment_processor.py` (add a batch scorer)
- Create: `src/ingestion/news_ingest.py`, `scripts/backfill_news.py`
- Modify: `scripts/refresh_data.py` (US-only hook before the feature rebuild)
- Modify: `config/settings.py` (add `FRED_KEY`), `.env.example`
- Test: `tests/unit/test_news_ingest.py`, extend `tests/unit/test_sentiment_processor.py`

**Interfaces:**
- Consumes: Task 1 functions; `collect_finnhub(ticker, from_date, to_date) -> list[NewsArticle]`.
- Produces:
  - `score_texts(texts: list[str], batch_size: int = 32) -> list[SentimentResult]`: one FinBERT pipeline call per batch; falls back to VADER for each text on error or when FinBERT is unavailable.
  - `ingest_news(tickers: list[str], start: date, end: date, news_dir: Path, trading_days: pl.Series, fetch=collect_finnhub, sleep=time.sleep) -> dict`: chunks the range into calendar months; paces at most 55 calls/min; on an exception containing 429 it sleeps 60s and retries once, then skips. Scores headline + " " + summary in memory, then calls `write_scores`, `write_headlines`, `rebuild_daily(since=start)`, and `prune`. Returns `{"tickers_ok", "tickers_failed", "articles_new"}`.
- [ ] **Step 1: Write failing tests**
  - `test_score_texts_batches`: with a mocked pipeline, 70 texts → 3 pipeline calls; output length 70.
  - `test_ingest_skips_ticker_after_429`: `fetch` raises "429" twice for MSFT and returns 2 articles for AAPL. Result: `tickers_failed == ["MSFT"]`, `articles_new == 2`, and an injected fake `sleep` was called with 60.
  - `test_ingest_never_writes_body_text`: no stored parquet contains a column named `body`, `summary`, or `url`.
  - `test_ingest_missing_key_returns_empty`: with `FINNHUB_KEY=""` it logs a warning and returns zero counts.
- [ ] **Step 2: Run** `pytest tests/unit/test_news_ingest.py tests/unit/test_sentiment_processor.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `score_texts` and `ingest_news`. Build `trading_days` from the SPY raw parquet `time` column.
- [ ] **Step 4: Implement** `scripts/backfill_news.py --days 365` (US tickers from `_TICKERS_BY_MARKET["us"]`, news_dir `get_market("us").data_root / "news"`). In `refresh_data.main`, when `market == "us"`, call `ingest_news` for `[last stored daily date - 3 days, today]` before `[2/2] Rebuilding features`, wrapped in try/except with a warning.
- [ ] **Step 5: Run** the tests. Expected: PASS. Also run `pytest tests/unit/test_refresh_data.py -v`: PASS.
- [ ] **Step 6: Commit** the files listed above with `-m "feat: Finnhub news ingest with batch FinBERT scoring"`.

### Task 3: Fed/macro daily series and FOMC tone

**Files:**
- Create: `src/ingestion/macro_ingest.py`, `scripts/refresh_macro.py`
- Modify: `scripts/refresh_data.py` (US-only call next to the news hook)
- Test: `tests/unit/test_macro_ingest.py`

**Interfaces:**
- Produces:
  - `FRED_SERIES = ("DFF", "DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2")`.
  - `fetch_fred(series_id: str, start: date, api_key: str, get=requests.get) -> pl.DataFrame`: columns `date, value` (Float64). "." values become null.
  - `score_fomc_statements(statements: list[tuple[datetime, str]]) -> pl.DataFrame`: columns `release_ts, tone`. Splits each statement into sentences and runs the FOMC-RoBERTa classifier (Hugging Face id `gtfintechlab/FOMC-RoBERTa`; confirm the id and the label→hawkish/dovish/neutral mapping on its model card before coding). `tone = (n_hawkish - n_dovish) / n_sentences`.
  - `fetch_fomc_statements(since: date, get=requests.get) -> list[tuple[datetime, str]]`: statement pages linked from `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm` (and the historical pages for older years). Release time is 14:00 America/New_York on the meeting's final day.
  - `build_macro_daily(fred: dict[str, pl.DataFrame], fomc: pl.DataFrame, trading_days: pl.Series) -> pl.DataFrame`: one row per trading day with columns `macro_dff, macro_dgs2, macro_dgs10, macro_t10y2y, macro_hy_spread, macro_dgs2_chg_5d, macro_dgs10_chg_5d, macro_hy_spread_chg_5d, macro_hy_spread_z20, fomc_tone, fomc_tone_change, days_since_fomc`. Forward-fill each series, then shift every column by 1 trading day. FOMC tone takes effect on the first trading day after the release date.
  - Outputs: `markets/us/data/macro/macro_daily.parquet` and `fomc_scores.parquet` (cached; only new statements are scored).
- [ ] **Step 1: Write failing tests**
  - `test_fetch_fred_parses_dots_as_null` (mocked `get`).
  - `test_macro_forward_fill_and_shift`: a gap day is filled; the value from day T appears on day T+1, not day T.
  - `test_fomc_effective_next_trading_day`: a release Wed 14:00 ET → `days_since_fomc == 0` on Thu and the tone shows from Thu.
  - `test_fomc_tone_formula`: mocked classifier labels [hawkish, hawkish, dovish, neutral] → tone 0.25.
- [ ] **Step 2: Run** `pytest tests/unit/test_macro_ingest.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the module. `scripts/refresh_macro.py` fetches from 2010-01-01 on its first run and from the last stored date - 10 days after that, scores only new FOMC statements, and writes both files. Hook it into `refresh_data.main` for US with try/except + warning. Skip with a warning when `FRED_KEY` is empty.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `-m "feat: FRED and FOMC tone macro features"`.

### Task 4: Join news/macro columns into US features

**Files:**
- Create: `src/features/news_macro_features.py`
- Modify: `scripts/build_features.py` (`build_features_for_ticker`, `iter_live_features`, `main`), `scripts/refresh_data.py` (pass the frames through)
- Test: `tests/unit/test_news_macro_features.py`, extend `tests/unit/test_build_features.py`

**Interfaces:**
- Consumes: `load_daily(news_dir)`, `macro_daily.parquet`.
- Produces:
  - `NEWS_COLS = ["news_count_1d", "news_count_3d", "news_has_3d", "news_net_3d", "news_net_z60", "news_vol_z20"]`
  - `MACRO_COLS =` the 12 columns from Task 3.
  - `OVERLAY_COLS = NEWS_COLS + MACRO_COLS`
  - `add_news_features(df: pl.DataFrame, news_daily: pl.DataFrame | None) -> pl.DataFrame`: left-join on `(ticker, date)`. Missing → counts 0, `news_has_3d` False, net 0.0. `news_net_3d` is the count-weighted mean of `avg_pos - avg_neg` over 3 rows. `news_net_z60` is the z-score vs. a 60-row rolling window (min 20, else 0.0). `news_vol_z20` is the z-score of `news_count_1d` over 20 rows (min 10, else 0.0).
  - `add_macro_features(df: pl.DataFrame, macro_daily: pl.DataFrame | None) -> pl.DataFrame`: left-join on date. Missing file → all `MACRO_COLS` = 0.0 (`days_since_fomc` = 999).
  - `build_features_for_ticker(..., news_daily: pl.DataFrame | None = None, macro_daily: pl.DataFrame | None = None)`: new keyword args, applied after `add_neutral_sentiment`. China callers pass nothing, so China output is unchanged.
- [ ] **Step 1: Write failing tests**
  - `test_no_news_gives_neutral_columns_same_rows`: with `news_daily=None`, all `NEWS_COLS` are present, the row count is unchanged, and the frame has no nulls.
  - `test_news_net_3d_weighted`: day counts 1 and 3 with net 1.0 and 0.0 → 0.25.
  - `test_china_output_unchanged`: the column set from `build_features_for_ticker` without the new args equals today's set (no `news_`/`macro_` columns).
  - `test_us_features_contain_overlay_cols` (the `test_build_features.py` extension).
  - `test_feature_cols_unchanged`: `dashboard.ui_config.FEATURE_COLS` has the same 22 names as before.
- [ ] **Step 2: Run** `pytest tests/unit/test_news_macro_features.py tests/unit/test_build_features.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement**, loading the news/macro frames once per run in `main` / `iter_live_features` / `refresh_data.main` when `market == "us"`.
- [ ] **Step 4: Run** the tests plus `pytest tests/unit -q`. Expected: all PASS.
- [ ] **Step 5: Commit** with `-m "feat: join news and macro columns into US features"`.

### Task 5: Overlay rules and OverlayStrategy wrapper

**Files:**
- Create: `src/signals/overlay.py`
- Test: `tests/unit/test_overlay.py`

**Interfaces:**
- Consumes: `OVERLAY_COLS`; `Strategy`, `PredictionResult` from `src/strategies/base.py`.
- Produces:
  - `@dataclass(frozen=True) OverlayConfig`: `veto_z=-1.5`, `boost_z=1.5`, `min_news=3`, `boost_mult=1.10`, `dgs2_jump=0.25`, `hy_z=2.0`, `fomc_window_days=5`, `risk_off_mult=0.85`, `buy_threshold=0.6`. These are starting values; Task 7 tunes them.
  - `apply_overlay(signal: pd.Series, confidence: pd.Series, ctx: pd.DataFrame, cfg: OverlayConfig) -> tuple[pd.Series, pd.Series, pd.Series]`: returns `(signal, confidence, reason)`. Rules, in order:
    1. Risk-off flags: `macro_dgs2_chg_5d > dgs2_jump`; `macro_hy_spread_z20 > hy_z`; `fomc_tone_change > 0 and days_since_fomc <= fomc_window_days`. Confidence × `risk_off_mult ** n_flags`.
    2. News veto: `news_count_3d >= min_news and news_net_z60 < veto_z` → Buy becomes Hold.
    3. News boost: `news_count_3d >= min_news and news_net_z60 > boost_z` → confidence × `boost_mult`, capped at 1.0.
    4. A Buy whose adjusted confidence falls below `buy_threshold` becomes Hold.

    Reason strings are joined with "; ": `"BUY suppressed: negative news"`, `"confidence boosted: positive news"`, `"risk-off: 2y yield jump"`, `"risk-off: credit spread"`, `"risk-off: hawkish FOMC"`. Empty string when no rule fires.
  - `class OverlayStrategy(Strategy)`: `__init__(inner: Strategy, cfg: OverlayConfig = OverlayConfig())`. `data_source = inner.data_source`. `fit` drops `OVERLAY_COLS` and calls `inner.fit`. `predict` drops them for the inner model, then applies the overlay using them. A missing `ctx` column counts as neutral. Exposes `last_reason: pd.Series`.
- [ ] **Step 1: Write failing tests**
  - `test_veto_turns_buy_to_hold` and `test_veto_needs_min_news` (count 2 → no veto).
  - `test_risk_off_two_flags`: 0.8 × 0.85² ≈ 0.578 → Hold, reason contains both risk-off strings.
  - `test_boost_capped_at_one`.
  - `test_wrapper_inner_never_sees_overlay_cols`: the fake inner strategy asserts no `news_`/`macro_`/`fomc_` columns in `fit` or `predict`.
  - `test_missing_ctx_columns_is_noop`.
- [ ] **Step 2: Run** `pytest tests/unit/test_overlay.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with vectorized pandas, no row loops.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `-m "feat: news/macro overlay wrapper for strategies"`.

### Task 6: US strategy pruning, overlaid live signals, dashboard reason

**Files:**
- Modify: `config/markets.py` (`MarketConfig.default_strategies: tuple[str, ...] | None = None`; US = `("extra_trees", "lda_strategy", "random_forest", "conformal_lgbm")`, China = `None`)
- Modify: `scripts/precompute_dashboard.py` (`main`, `step_signals`)
- Modify: `dashboard/data_loader.py` and the dashboard signals table that renders `trade_status` (add an "Overlay" column)
- Test: extend `tests/unit/test_precompute_dashboard.py`, `tests/unit/test_markets.py`, `tests/unit/test_dashboard_data_loader.py`

**Interfaces:**
- Consumes: `OverlayStrategy`, `OverlayConfig`, `OVERLAY_COLS`.
- Produces:
  - `main` resolves names as `strategies or market_cfg.default_strategies or list_strategies()` and passes them to `step_backtests(names=...)` and `step_signals(names=...)`.
  - `step_signals(names: list[str] | None = None, exclude=..., market=..., ...)`: for `market == "us"`, wraps each strategy in `OverlayStrategy` and selects `FEATURE_COLS + OVERLAY_COLS`. Each signal dict gains `"overlay_reason": str` (empty when nothing fired) and `"raw_confidence": float`.
  - `get_live_signals` / `get_combined_ratings` pass `overlay_reason` through (missing key → "").
- [ ] **Step 1: Write failing tests**
  - `test_us_default_strategies_used`: `main("us", no_signals=True)` with `step_backtests` patched receives the 4 names.
  - `test_china_runs_all_strategies`.
  - `test_us_signals_include_overlay_reason`: synthetic live features with a negative-news row → that ticker's signal is Hold with reason `"BUY suppressed: negative news"`.
  - `test_loader_defaults_missing_overlay_reason`.
- [ ] **Step 2: Run** the three test files. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/unit -q`. Expected: all PASS.
- [ ] **Step 5: Commit** with `-m "feat: prune US strategies, overlay live signals, show reason"`.

### Task 7: Backfill, overlay evaluation, timing check (ops + script)

**Files:**
- Create: `scripts/evaluate_overlay.py`
- Test: `tests/unit/test_evaluate_overlay.py`

**Interfaces:**
- Consumes: `walk_forward_backtest_strategy`, `OverlayStrategy`, `OverlayConfig`, US default strategies.
- Produces: `evaluate(df: pl.DataFrame, names: list[str], cfg: OverlayConfig) -> list[dict]`, rows `{strategy, variant ("plain"|"overlay"), n_trades, sharpe, precision_buy}`. CLI `--since YYYY-MM-DD` filters `df` to the news window (with a 400-day train lookback before it) and writes `markets/us/data/cache/overlay_eval.json`.
- [ ] **Step 1: Write failing test** `test_evaluate_returns_plain_and_overlay_rows`: on a small synthetic df with 1 strategy → 2 rows with matching strategy names.
- [ ] **Step 2: Run** it. Expected: FAIL. **Step 3: Implement.** **Step 4: Run** it. Expected: PASS. **Commit** with `-m "feat: overlay vs plain evaluation script"`.
- [ ] **Step 5: Ops.** Requires `FINNHUB_KEY` and `FRED_KEY` in `.env`; stop and ask the user if they are missing.
  - `python scripts/run_job.py backfill_news -- scripts/backfill_news.py --days 365`
  - `python scripts/run_job.py macro_us -- scripts/refresh_macro.py`
  - `python scripts/run_job.py build_us -- scripts/build_features.py --market us`

  Check `--status` until each is `done exit=0`. Report the `markets/us/data/news/` total size (expected 10–20 MB) and the article count.
- [ ] **Step 6: Evaluate.** `python scripts/run_job.py overlay_eval -- scripts/evaluate_overlay.py --since <first backfilled date>`. If the overlay lowers Sharpe for most strategies, adjust the `OverlayConfig` defaults once, re-run, and record both runs. Don't loop beyond that; report back to the user instead.
- [ ] **Step 7: Timing.** `python scripts/run_job.py precompute_us -- scripts/precompute_dashboard.py --market us`. Report the wall time against the target of under 2h (the 2026-10-28 run was ~16h).
- [ ] **Step 8:** Commit `src/signals/overlay.py` if the defaults changed. Ask the user before committing any data or cache files.
