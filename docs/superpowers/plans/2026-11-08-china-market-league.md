# China Market + Method League Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run China A-shares live with Chinese news and macro, replace "all 31 methods" with a live/bench league (2-week bench re-test, dedup, retirement), add new candidate methods, and update the dashboard.

**Architecture:** Server B (Guangzhou) fetches China news and China macro with akshare, scores news with a Chinese FinBERT, and syncs to A's write-only inbox under `china/`. Server A runs China daily/weekly pipelines and a 2-weekly bench pass for both markets. A pure-function league module decides live/bench/duplicate/retired from bench results and persists `cache/league.json`; the live set replaces `MarketConfig.default_strategies` (which becomes the seed).

**Tech Stack:** Python 3.12, polars, DuckDB, LightGBM, transformers (CPU), akshare, systemd, Streamlit.

**Spec:** `docs/superpowers/specs/2026-11-08-china-market-league-design.md`

## Global Constraints

- No paid data, no server upgrades, no firewall changes without asking the user.
- Secrets only in `~/app/.env` (chmod 600); never logged, committed or backed up.
- Server A: one heavy job at a time; `MODEL_N_JOBS=2`, `SIGNAL_TRAIN_DAYS=400`. US daily (Mon–Fri 18:30 ET) and US weekly (Sat 03:00 ET) times unchanged.
- Server B: `MemoryMax=3G`, `Nice=19`, `IOSchedulingClass=idle`, `HF_ENDPOINT=https://hf-mirror.com`.
- Walk-forward: step 21 days, train window 400 days, test window 21 days.
- News scoring cap: 30 scored articles per ticker per market-local day; unscored rows kept with null sentiment, `scorer="unscored"`.
- League: live ≤ 6 per market; ineligible for live if < 20 trades in the bench window; promotion/demotion needs 2 consecutive bench passes; similar = rank-corr ≥ 0.85 OR buy-pick Jaccard ≥ 0.70; retirement after 3 consecutive similar passes spanning ≥ 28 days; nothing deleted.
- Costs: 10 bps per side US, 15 bps per side China.
- Bench window: last 3 years (1095 days). Bench hard stop Mon 04:30 ET.
- All dashboard times US Eastern; China pages also show Beijing time.
- Long jobs run as systemd units on the servers, never on the dev PC.

## Review Focus

1. A China trading day with zero akshare articles for a ticker (or akshare raising) → that ticker's news columns are neutral, the run continues, other tickers unaffected. Pinned in Task 2.
2. Chinese holidays / weekends → China daily run completes, signals unchanged, no failure state. Pinned in Task 4 (pipeline treats "no new bars" as success).
3. A method that errors during the bench pass → recorded as `error` with its previous result kept, never promoted, pass continues. Pinned in Task 7.
4. First bench pass with no history → no promotions (needs 2 passes), live set = seed; league.json created. Pinned in Task 6.
5. Bench pass stopped by the time limit → untested methods keep old date and are first next pass; league update still runs on what finished. Pinned in Task 7.

## Rulings (deviations from spec)

- Ruling: China macro is fetched on **B**, not A — akshare endpoints are mainland sites and A is offshore — cost if wrong: one extra file in the B→A sync, nothing else.
- Ruling: the "every 2nd Sunday" bench is a weekly Sun 13:00 timer whose script exits 0 unless the last completed pass is ≥ 13 days old — systemd has no native biweekly calendar.

---

### Task 1: Market-local time for news storage

**Files:**
- Modify: `config/markets.py` (add fields), `src/ingestion/news_store.py:42-66,146`, `src/ingestion/news_ingest.py:147-160`
- Test: `tests/unit/test_news_store.py`, `tests/unit/test_markets_config.py`

**Interfaces:**
- Produces: `MarketConfig.timezone: str` (`"America/New_York"` us, `"Asia/Shanghai"` china), `MarketConfig.close_time: datetime.time` (16:00 us, 15:00 china), `MarketConfig.cost_bps: float` (10.0 us, 15.0 china).
- Produces: `assign_trading_day(published_at, trading_days, tz: str = "America/New_York", cutoff: time = time(16, 0)) -> pl.Series`; `rebuild_daily(news_dir, trading_days, since, tz=..., cutoff=...) -> None`.
- Produces: `news_ingest._score_and_write(news_dir, fresh, today, tz: ZoneInfo = _NY) -> int` (per-day cap uses `tz`).

- [ ] **Step 1: Write failing tests**

```python
def test_china_cutoff_rolls_after_1500_shanghai():
    days = pl.Series([date(2026, 11, 9), date(2026, 11, 10)])
    ts = pl.Series([datetime(2026, 11, 9, 6, 59, tzinfo=timezone.utc),   # 14:59 CST
                    datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc)])   # 15:00 CST
    out = assign_trading_day(ts, days, tz="Asia/Shanghai", cutoff=time(15, 0))
    assert out.to_list() == [date(2026, 11, 9), date(2026, 11, 10)]

def test_us_default_unchanged():
    # existing US tests keep passing with no tz argument
    ...

def test_market_config_china_fields():
    c = get_market("china")
    assert (c.timezone, c.close_time, c.cost_bps) == ("Asia/Shanghai", time(15, 0), 15.0)
```

- [ ] **Step 2:** `pytest tests/unit/test_news_store.py tests/unit/test_markets_config.py -v` → new tests FAIL.
- [ ] **Step 3:** Add the fields and parameters; defaults keep US behavior byte-identical.
- [ ] **Step 4:** Full `pytest tests/unit -q` → only the 21 baseline failures.
- [ ] **Step 5:** Commit `feat: market-local timezone and close for news day assignment`.

---

### Task 2: China news ingest (akshare + Chinese FinBERT)

**Files:**
- Create: `src/ingestion/cn_news_ingest.py`, `scripts/refresh_cn_news.py`
- Modify: `src/ingestion/sentiment_processor.py` (add Chinese scorer), `requirements-server.txt` (add `akshare` pinned to the version installed in Step 3)
- Test: `tests/unit/test_cn_news_ingest.py`

**Interfaces:**
- Consumes: Task 1 `rebuild_daily(..., tz="Asia/Shanghai", cutoff=time(15,0))`, `_score_and_write(..., tz=ZoneInfo("Asia/Shanghai"))`.
- Produces: `fetch_cn_stock_news(code: str, get=ak.stock_news_em) -> list[Article]` (code = 6-digit, e.g. `"000009"` from `"000009.SZ"`; Article fields as in `news_collector.Article`; `published_at` parsed as Asia/Shanghai then UTC).
- Produces: `ingest_cn_news(tickers: list[str], news_dir: Path, trading_days: pl.Series, fetch=fetch_cn_stock_news, sleep=time.sleep, today: date | None = None) -> dict` with keys `tickers_ok`, `tickers_failed`, `articles_new` (same contract as US `ingest_news`).
- Produces: `score_texts_cn(texts: list[str], batch_size: int = 16) -> list[SentimentResult]`, model `yiyanghkust/finbert-tone-chinese`, fallback `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` (2-class: map to pos/neg, neu=0.0); `scorer` = model short name.
- Produces: `scripts/refresh_cn_news.py main() -> dict` — reads China tickers from `refresh_data._TICKERS_BY_MARKET["china"]`, trading days from `510300.SS.parquet`, writes `markets/china/data/news/`; also refreshes `markets/china/data/index_compositions/names.parquet` (`ticker, name`) from akshare `stock_info_a_code_name` at most once a week, synced to A with the news.

- [ ] **Step 1: Write failing tests** (fake `fetch`, fake scorer via monkeypatch of `score_texts_cn`)

```python
def test_dedup_by_url_hash_across_runs(tmp_path): ...  # second run with same articles -> articles_new == 0
def test_fetch_error_marks_failed_and_continues(tmp_path):
    def fetch(code):
        if code == "000009": raise RuntimeError("boom")
        return [art("000021.SZ", "https://x/1")]
    res = ingest_cn_news(["000009.SZ", "000021.SZ"], tmp_path, DAYS, fetch=fetch, sleep=lambda s: None)
    assert res["tickers_failed"] == ["000009.SZ"] and res["tickers_ok"] == ["000021.SZ"]
def test_zero_articles_is_ok(tmp_path): ...  # ticker in tickers_ok, no daily row for it
def test_cap_30_per_shanghai_day(tmp_path): ...  # 35 articles same CST day -> 30 scored, 5 scorer=="unscored", article_count==35
def test_daily_file_uses_1500_cutoff(tmp_path): ...  # 15:30 CST article lands next trading day
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Pace akshare at 1 request/second; one retry after 30 s; per-ticker checkpoint `rebuild_daily` every 25 tickers (same as US). Install akshare in the dev venv and pin its exact version.
- [ ] **Step 4:** Tests pass; one live smoke on the PC for 3 tickers: `python scripts/refresh_cn_news.py --tickers 000009.SZ,000021.SZ,000027.SZ` prints `ok=3`.
- [ ] **Step 5:** Commit `feat: China stock news ingest with Chinese FinBERT`.

---

### Task 3: China macro + market-aware overlay columns

**Files:**
- Create: `src/ingestion/cn_macro_ingest.py`, `scripts/refresh_cn_macro.py`
- Modify: `src/features/news_macro_features.py`, `scripts/build_features.py:270`, `scripts/refresh_data.py:124-129`, `src/strategies/statistical/pooled_lgbm.py`, `scripts/precompute_dashboard.py:262-265`, `src/strategies/strategies.yaml`
- Test: `tests/unit/test_cn_macro.py`, `tests/unit/test_news_macro_features.py`, `tests/unit/test_pooled_lgbm.py`

**Interfaces:**
- Produces: `CN_MACRO_COLS = ["cn_lpr_1y", "cn_lpr_1y_chg_20d", "cn_rrr", "cn_rrr_cut_20d", "cn_cpi_yoy", "cn_ppi_yoy", "cn_pmi", "cn_pmi_caixin", "cn_m2_yoy", "cn_margin_bal_z20", "cn_margin_bal_chg_5d", "cn_usdcny_chg_5d"]`.
- Produces: `overlay_cols(market: str) -> list[str]` (us: `NEWS_COLS + MACRO_COLS` = current `OVERLAY_COLS`; china: `NEWS_COLS + CN_MACRO_COLS`). `OVERLAY_COLS` stays as the US alias.
- Produces: `load_news_macro(market: str) -> tuple[pl.DataFrame, pl.DataFrame]` (US delegates to `load_us_news_macro`; china reads `markets/china/data/news/sentiment_daily.parquet` and `markets/china/data/macro/macro_daily.parquet`, empty frame when missing).
- Produces: `add_macro_features(df, macro_daily, cols: list[str] = MACRO_COLS)` — neutral 0.0 fill for CN cols.
- Produces: `build_cn_macro_daily(raw: dict[str, pl.DataFrame], days: list[date]) -> pl.DataFrame` with column `date` + `CN_MACRO_COLS`, each series forward-filled **from its release date** (monthly series use the publication date akshare reports, or month-end + 15 days when only the period is given).
- Produces: `PooledLGBMStrategy(market: str = "us", ...)` → `extra_feature_cols = tuple(overlay_cols(market))`; new yaml entry `pooled_lgbm_news_cn` with `markets: [china]`, `params: {market: china}`.
- Precompute: China signals pass `overlay_cols("china")` along for strategies that declare `extra_feature_cols`; the rule-based `OverlayStrategy` wrapper stays US-only.

- [ ] **Step 1: Failing tests**

```python
def test_monthly_series_not_visible_before_release():
    # CPI for October released 2026-11-09 -> rows dated 2026-11-08 carry September's value
def test_overlay_cols_china():
    assert overlay_cols("china") == NEWS_COLS + CN_MACRO_COLS
def test_china_features_have_neutral_cn_cols_when_macro_missing(): ...
def test_pooled_lgbm_china_declares_cn_cols():
    assert PooledLGBMStrategy(market="china").extra_feature_cols == tuple(overlay_cols("china"))
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (akshare macro calls live only in `cn_macro_ingest.fetch_cn_macro_raw() -> dict[str, pl.DataFrame]`, untested network wrapper). **Step 4:** Unit suite at baseline; `python scripts/build_features.py --market china --tickers 000009.SZ` writes a parquet containing every `CN_MACRO_COLS` column.
- [ ] **Step 5:** Commit `feat: China macro features and market-aware overlay columns`.

---

### Task 4: China pipelines, B collector, deploy units

**Files:**
- Modify: `deploy/run_pipeline.sh`, `deploy/backup.sh`, `dashboard/system_status.py` (SERVICES/TIMERS/_STEP_NAMES)
- Create: `deploy/cn_collect_and_sync.sh`, `deploy/systemd/fsp-cn-news.service`, `fsp-cn-news.timer` (B), `fsp-cn-daily.service`, `fsp-cn-daily.timer`, `fsp-cn-weekly.service`, `fsp-cn-weekly.timer` (A)
- Test: `tests/unit/test_system_status.py` (step names), `bash -n` on scripts

**Interfaces:**
- `run_pipeline.sh` modes: `cn-daily` = `apply_cn_inbox` → `refresh_data.py --market china` → `incremental_train.py --market china` → `precompute_dashboard.py --market china --signals-only`; `cn-weekly` = `train_models.py --market china` → `precompute_dashboard.py --market china`.
- `apply_cn_inbox`: inbox `/home/fspsync/incoming/china/` with `READY`, `news/`, `macro/`; stamp file `$PWD/.cn_applied`; rsyncs into `markets/china/data/news/` and `markets/china/data/macro/`.
- `cn_collect_and_sync.sh` (B, user `fsp`): `refresh_cn_news.py` then `refresh_cn_macro.py`, then `rsync -a --delete` news and macro to `fspsync@A:china/news/`, `china/macro/`, then upload `china/READY` (UTC stamp). Uses the existing `fspsync` key; rrsync root stays `/home/fspsync/incoming`.
- Timers: `fsp-cn-news.timer` `Mon..Fri 16:30 Asia/Shanghai`; `fsp-cn-daily.timer` `Mon..Fri 05:30 America/New_York`; `fsp-cn-weekly.timer` `Sun 03:00 America/New_York`, `TimeoutStartSec=12h`. All `Persistent=true`. Service env same as `fsp-daily.service`.
- `refresh_data.py --market china`: zero new bars is success (exit 0).
- `backup.sh`: archive `markets/us/data` and `markets/china/data`, excluding `raw/` for China (re-downloadable); keep 7.

- [ ] **Step 1:** Failing test: `summarize_run` maps `refresh_data.py --market china` → "Refresh data (prices, news, macro, features)" and a `=== ... applying china upload ... ===` marker → "Apply China news from Server B".
- [ ] **Step 2–4:** Implement; `bash -n` every script; unit suite at baseline.
- [ ] **Step 5:** Commit `feat: China daily/weekly pipelines and B collector`.

---

### Task 5: Prediction log + cost-aware score from walk-forward

**Files:**
- Modify: `src/backtesting/strategy_runner.py` (`walk_forward_backtest_strategy`, `_run_stateless`, `_run_stateful`)
- Create: `src/league/__init__.py`, `src/league/scoring.py`
- Test: `tests/unit/test_league_scoring.py`, `tests/unit/test_strategy_runner.py`

**Interfaces:**
- Produces: `walk_forward_backtest_strategy(..., start: datetime | None = None, pred_log: list | None = None)`. `start` drops fold windows whose test_start < `start` (training data before `start` is still used). `pred_log`, when given, receives one `(ticker: str, time: date, buy_score: float)` per test row: `confidence` if signal == "Buy" and confidence ≥ threshold, else 0.0.
- Produces: `score_predictions(preds: pl.DataFrame, outcomes: pl.DataFrame, cost_bps: float) -> dict` — `preds` columns `ticker, time, buy_score`; `outcomes` columns `ticker, time, forward_return_5d`. Trades = rows with `buy_score > 0`. Net return per trade = `forward_return_5d - 2*cost_bps/1e4`. Returns `{"net_sharpe": float, "n_trades": int, "precision": float, "mean_net_return": float}`; `net_sharpe = mean/std * sqrt(252/5)` over trades, 0.0 when n_trades < 2 or std == 0; precision = share of trades with net return > 0.

- [ ] **Step 1: Failing tests**

```python
def test_cost_reduces_return():
    r = score_predictions(preds([1.0]*3), outcomes([0.01, 0.02, 0.0]), cost_bps=10)
    assert r["mean_net_return"] == pytest.approx(0.01 - 0.002)
def test_no_trades_scores_zero(): assert score_predictions(preds([0.0]*5), ..., 10)["net_sharpe"] == 0.0
def test_start_limits_folds(): ...  # fold test_start >= start for all folds
def test_pred_log_one_row_per_test_row(): ...
```

- [ ] **Steps 2–4:** Run, implement, unit suite at baseline (existing backtest outputs unchanged when new args omitted).
- [ ] **Step 5:** Commit `feat: walk-forward prediction log and cost-aware league score`.

---

### Task 6: League state machine

**Files:**
- Create: `src/league/league.py`, `config/league.yaml`
- Test: `tests/unit/test_league.py`

**Interfaces:**
- `config/league.yaml`: `live_max: 6`, `min_trades: 20`, `promote_after_passes: 2`, `similar_rank_corr: 0.85`, `similar_jaccard: 0.70`, `retire_after_passes: 3`, `retire_min_days: 28`, `bench_window_days: 1095`, `bench_min_gap_days: 13`.
- `similarity(a: pl.DataFrame, b: pl.DataFrame) -> tuple[float, float]` — inputs are pred logs (`ticker,time,buy_score`). Rank-corr = mean over days of Spearman correlation of `buy_score` across tickers (days where either side is constant are skipped; no usable days → 0.0). Jaccard = |buys_a ∩ buys_b| / |buys_a ∪ buys_b| on (ticker,time) with `buy_score > 0` (empty union → 0.0).
- `LeagueState` = dict persisted at `<cache_dir>/league.json`:
  `{"market", "passes": [{"date", "completed": bool}], "methods": {name: {"status": "live"|"bench"|"retired", "history": [{"date", "net_sharpe", "n_trades", "precision", "error": str|None}], "beats_weakest_streak": int, "duplicate_of": str|None, "similar_since": date|None, "similar_passes": int, "retired_on": date|None, "retired_because": str|None}}}`
- `load_state(cache_dir, market, seed: list[str]) -> dict` (new file → seed methods `live`, others added as `bench` on first sight).
- `update_league(state: dict, results: dict[str, dict], preds: dict[str, pl.DataFrame], today: date, cfg: dict) -> dict` — pure. Order:
  1. Append each result to history (errors recorded, keep previous numbers for ranking).
  2. Similarity groups among non-retired methods tested this pass; within a group the best `net_sharpe` is the winner, others get `duplicate_of=winner`; `similar_passes += 1` if same winner as last pass else reset to 1 with `similar_since=today`; non-similar → cleared.
  3. Retire duplicates with `similar_passes >= 3` and `today - similar_since >= 28 days` (`retired_because = "similar to <winner> since <date>"`). A live method is never retired while it is the group winner.
  4. Eligible = not retired, not duplicate, no error this pass, `n_trades >= 20`. For each eligible bench method that beats the weakest eligible-or-not live method on `net_sharpe`, `beats_weakest_streak += 1`, else reset to 0. Swap when streak ≥ 2 (then reset). If live has < 6 and an eligible bench method has streak ≥ 2, promote without demotion. Live duplicates are demoted to bench immediately.
  5. Record the pass.
- `live_set(cache_dir: Path, market: str) -> list[str] | None` — live names from league.json, None when the file doesn't exist.
- `save_state(cache_dir, state) -> None` — atomic write (tmp + rename).

- [ ] **Step 1: Failing tests** (synthetic results, no data)

```python
def test_first_pass_no_promotion(): ...                    # Review Focus 4
def test_promotion_needs_two_consecutive_wins(): ...
def test_streak_resets_on_loss(): ...
def test_min_trades_blocks_live(): ...
def test_duplicate_keeps_best_and_marks_other(): ...
def test_retire_after_three_passes_and_28_days():
    # passes on day 0, 14, 28 similar to same winner -> retired on day 28; day 0,14 only -> not retired
def test_error_method_keeps_status_and_never_promotes(): ...  # Review Focus 3
def test_similarity_identical_preds_is_1_1(): ...
def test_similarity_disjoint_buys_jaccard_0(): ...
```

- [ ] **Steps 2–4:** Run, implement, pass.
- [ ] **Step 5:** Commit `feat: league state machine with promotion, dedup and retirement`.

---

### Task 7: Bench pass script + live set wiring

**Files:**
- Create: `scripts/bench_pass.py`, `deploy/systemd/fsp-bench.service`, `fsp-bench.timer`
- Modify: `scripts/precompute_dashboard.py:371-374`, `deploy/run_pipeline.sh` (mode `bench`), `dashboard/system_status.py`
- Test: `tests/unit/test_bench_pass.py`

**Interfaces:**
- `bench_pass.main(market: str, deadline: datetime | None = None, strategies: list[str] | None = None, force: bool = False) -> dict`:
  - Exit early (return `{"skipped": "recent"}`) unless `force` or last completed pass ≥ `bench_min_gap_days` ago.
  - Methods = `list_strategies(market)` minus retired; order = never tested or stopped last pass first, then oldest test date.
  - Loads `load_training_data(feature_dir, start=today - (1095 + 400) days)` once; per method runs `walk_forward_backtest_strategy(..., start=today-1095d, pred_log=...)`, scores with `score_predictions(..., cost_bps=get_market(market).cost_bps)`, writes `cache/bench/bench_<name>.json` (`tested_on`, `window_days: 1095`, score dict, standard metrics) and `cache/bench/preds_<name>.parquet`.
  - A method exception → result `{"error": str(exc)}`, continue. Past `deadline` → stop starting new methods; pass `completed=False`.
  - Then `update_league` + `save_state`; retired methods' latest bench json moved to `cache/retired/`.
- `--seed-from-pass` (one-time, used when no live set exists yet for the market): after the pass, set live = top `live_max - 1` (= 5) eligible methods by `net_sharpe`, skipping duplicates.
- `precompute_dashboard.main`: `default_names = live_set(cache_dir, market) or get_market(market).default_strategies or list_strategies(market)`.
- `run_pipeline.sh bench`: `bench_pass.py --market us --deadline <next Mon 04:30 ET>` then `--market china` with same deadline.
- `fsp-bench.timer`: `Sun 13:00 America/New_York`, `Persistent=true`; service `TimeoutStartSec=16h`, same env as weekly.

- [ ] **Step 1: Failing tests** (tiny synthetic feature frame, 2 cheap rule strategies + 1 that raises)

```python
def test_skips_when_recent(tmp_path): ...
def test_error_strategy_recorded_and_pass_continues(tmp_path): ...     # Review Focus 3
def test_deadline_stops_and_untested_go_first_next_time(tmp_path): ...  # Review Focus 5
def test_precompute_uses_league_live_set(tmp_path, monkeypatch): ...
```

- [ ] **Steps 2–4:** Run, implement, pass; unit suite at baseline.
- [ ] **Step 5:** Commit `feat: two-weekly bench pass driving the live set`.

---

### Task 8: New candidate methods

**Files:**
- Create: `src/strategies/statistical/news_surprise.py`, `regime_gated.py`, `stacked_live.py`; `src/strategies/statistical/limit_filter.py`
- Modify: `src/strategies/strategies.yaml`, `src/strategies/statistical/cross_sectional_ranker.py` (optional `use_news_macro`, `market` params)
- Test: `tests/unit/test_new_methods.py`

**Interfaces (all subclass the existing `Strategy`; fit/predict on pandas frames like `PooledLGBMStrategy`; `n_jobs=N_JOBS`):**
- `cross_sectional_ranker_news` (yaml entry, markets us+china): ranker with `use_news_macro=True, market=<m>` → `extra_feature_cols = overlay_cols(market)`.
- `NewsSurpriseStrategy(market)`: features `news_net_3d - rolling 60d mean` × `news_vol_z20`, plus price features; LightGBM classifier; `extra_feature_cols = NEWS_COLS`.
- `RegimeGatedStrategy(market)`: two LightGBM models split on volatility regime (US `vix` feature column; China 20d realized vol of the benchmark, already present as the synthetic vol index column) at the trailing 252d median; predict routes rows by regime.
- `StackedLiveStrategy(market, base: list[str] | None = None)`: base = current live set (via `live_set`, fallback seed) minus itself; out-of-fold base probabilities (3 time-ordered folds inside fit) → logistic regression stacker.
- `limit_filter.drop_limit_days(df: pd.DataFrame, market: str) -> pd.Series[bool]`: China only — True where `|close/prev_close - 1| >= 0.095` (ST stocks 0.045 when ticker flagged ST is unavailable → use 0.095 everywhere); applied to Buy entries of China strategies in `walk_forward_backtest_strategy` and live signals (signal forced to Hold). US → all False.
- yaml: each new entry lists `markets`; China gets `pooled_lgbm_news_cn` (Task 3), `cross_sectional_ranker_news_cn`, `news_surprise_cn`, `regime_gated_cn`, `stacked_live_cn`.

- [ ] **Step 1:** Failing tests: each strategy fits/predicts on a 3-ticker synthetic frame, returns confidence in [0,1] and signals in {Buy,Hold,Sell}; `StackedLiveStrategy` never includes itself; `drop_limit_days` marks a +10% day True and +5% False for china, all False for us.
- [ ] **Steps 2–4:** Run, implement, pass.
- [ ] **Step 5:** Commit `feat: new bench candidate methods and China limit-day filter`.

---

### Task 9: Dashboard — live/bench/retired, staleness, China

**Files:**
- Modify: `dashboard/data_loader.py`, `dashboard/pages/2_Model_Leaderboard.py`, `dashboard/pages/3_Backtest_Results.py`, `dashboard/pages/4_Live_Signals.py`, `dashboard/pages/5_Combined_Signal.py`, `dashboard/pages/6_System_Status.py`
- Test: `tests/unit/test_data_loader_league.py`

**Interfaces:**
- `get_league_table(market: str, today: date | None = None) -> dict[str, list[dict]]` with keys `live`, `bench`, `retired`; each row: `method, net_sharpe, precision, n_trades, tested_on, trend ("↑"|"↓"|"→"|""), duplicate_of, stale: bool, retired_because`. Stale = live tested > 8 days ago or bench > 15 days ago. No league.json → all registered methods under `bench` from existing backtest jsons, `tested_on` = their `generated_at`.
- `beijing_time(utc_iso: str) -> str` in `dashboard/system_status.py` (format `"%m-%d %H:%M CST"`).
- Leaderboard page: Live table first (sorted by net_sharpe), Bench in an expander, Retired in an expander; stale rows get a "stale" badge column.
- Backtest Results: method picker shows only live+bench; "last tested" caption on every result.
- Live/Combined Signals for china: ticker shown as `code name` when a name map exists (`markets/china/data/index_compositions/names.parquet`, written by Task 2's script from akshare `stock_info_a_code_name`), times ET + Beijing.
- System Status: China jobs, `fsp-bench`, B's China news status (B uploads `status/cn_news.txt` with the same key=value format as backfill).

- [ ] **Step 1:** Failing tests: staleness thresholds (8/15 days), fallback without league.json, trend arrows from the last two history entries.
- [ ] **Steps 2–4:** Run, implement, pass; `streamlit run dashboard/app.py` locally, switch market to China, every page renders without exceptions.
- [ ] **Step 5:** Commit `feat: dashboard league view, staleness badges and China pages`.

---

### Task 10: Roll out to servers

No code; each step's check is its output. Ask the user before anything that costs money.

- [ ] **Step 1:** Push: `git push origin main`, `git push server main`; on A `cd ~/app && git pull`; B gets a `git archive` snapshot (code only) into `/home/fsp/app`.
- [ ] **Step 2:** Install akshare (pinned) in both venvs; on B pre-download the Chinese FinBERT via the mirror. Check: `python -c "import akshare; print(akshare.__version__)"` on both.
- [ ] **Step 3:** Install units: A `fsp-cn-daily`, `fsp-cn-weekly`, `fsp-bench`; B `fsp-cn-news`. `systemctl daemon-reload`, enable timers. Check `systemctl list-timers 'fsp-*'` on each.
- [ ] **Step 4:** First China news run on B via `systemctl start fsp-cn-news`; check A inbox has `china/READY` and `china/news/sentiment_daily.parquet`.
- [ ] **Step 5:** China data catch-up on A: `systemd-run` a one-off `run_pipeline.sh cn-daily` (OHLCV from 2026-10-28). Check signals.json for china has today's date.
- [ ] **Step 6:** First bench pass, forced: `systemd-run ... bench_pass.py --market us --force` then china, outside US job hours. Check `league.json` exists for both markets, live set = seed (US) and top-by-score is visible for China; promote China seed = top 5 eligible by `net_sharpe` from this pass (one-time `--seed-from-pass` flag in `bench_pass.py`).
- [ ] **Step 7:** Dashboard check at https://43-166-232-106.sslip.io: China selector works, league tables populated, System Status lists all new jobs.
