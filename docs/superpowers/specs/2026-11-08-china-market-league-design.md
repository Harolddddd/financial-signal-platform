# China market + news, method league, dashboard update — design

Date: 2026-11-08. Status: draft for approval.

## Intent

1. Run the China A-share market (CSI 500, already in `config/markets.py`, OHLCV to 2026-10-28) live on the servers the same way the US runs: daily signals, weekly retrain, plus Chinese news and China macro.
2. Stop treating all 31 methods equally. Keep a small **live tier** per market retrained weekly. Keep every other method (and new candidates) on a **bench**, re-tested every **2 weeks**, with automatic promotion/demotion.
3. Research and add a handful of new methods that plausibly beat the current ones. They enter on the bench and earn promotion.
4. Update the dashboard: live tier on top, bench ranking table, "last tested" date on every result, US/China switch.

Success: China signals on the dashboard each China trading day; every method shown has a result no older than 2 weeks (bench) or 1 week (live); at least one new method evaluated on both markets.

## Non-goals

- No paid data (Tushare etc.). Free akshare + yfinance only.
- No server upgrades. Ask before any cost.
- No intraday trading, no order execution.
- No change to the US daily/weekly schedule times.

## Anti-goals (failure even if it "works")

- The Saturday US weekly run or the weekday US daily run gets slower or OOMs because China/bench work overlaps it.
- Dashboard results without a visible test date (stale results that look fresh).
- Promotion driven by in-sample or look-ahead numbers.
- Secrets leaving `~/app/.env`.

## Constraints

- Server A: 2 cores, 8 GB RAM, no paid swap growth; US weekly peaks 6.4 GB. Only one heavy job at a time on A.
- Server B (Guangzhou): ~3.7 GB RAM, 26 GB free disk, hosts a small website. Light jobs only; `MemoryMax=3G`, `Nice=19`, idle IO. huggingface via `HF_ENDPOINT=https://hf-mirror.com`.
- Threads: `MODEL_N_JOBS=2`. Long jobs run as systemd units, never from the PC.
- Walk-forward step stays 21 days.
- Northbound (HK Connect) daily net flow is no longer published (since Aug 2024), so it is not a feature.

## Decisions (user's)

- Compute split A + B: B collects and scores Chinese news and syncs to A; A trains China.
- Free akshare news source.
- Live + bench league, bench re-test every 2 weeks (not monthly).

## Design

### China news (Server B)

- Source: akshare `stock_news_em(symbol)` per CSI 500 constituent (East Money; returns about the latest 100 articles). Plus market-wide headlines (Cailian 财联社 telegraph) used as a market-level sentiment column.
- History builds forward from the first run; one initial pull grabs whatever each stock's ~100 articles cover.
- Scoring: Chinese financial sentiment model (my call: `yiyanghkust/finbert-tone-chinese`; fallback `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` if the mirror lacks it). Same cap as US: 30 scored articles per ticker per Beijing day, deterministic sample, unscored rows kept with null sentiment so counts stay exact.
- Storage: same tiered layout as US news (`markets/china/data/news/`), same daily per-ticker aggregate file that the feature build reads.
- Schedule on B: daily 16:30 Asia/Shanghai (after 15:00 close), systemd timer. Sync to A via the existing write-only `fspsync` rrsync inbox, into a `china/` subfolder with its own READY stamp.

### China macro (Server A, during the China daily run)

akshare: LPR, RRR changes, CPI/PPI, official + Caixin PMI, M2/TSF, margin financing balance (SSE+SZSE), USD/CNY. Forward-filled by release date (no look-ahead), same overlay-column scheme as US macro.

### China daily pipeline (Server A)

- Timer: Mon–Fri 05:30 America/New_York (= 18:30 Beijing; after China close and after B's news sync). Persistent. Finishes well before the US 18:30 ET run.
- Steps: apply China news inbox → refresh China OHLCV (yfinance `.SS/.SZ`) + macro + features → incremental train → signals for China live tier.
- Same `SIGNAL_TRAIN_DAYS=400` cap.
- Chinese holidays: run is harmless on non-trading days (no new bars → signals unchanged).

### Weekly live retrain

- US: unchanged, Sat 03:00 ET.
- China: Sun 03:00 ET. Full retrain + full-history walk-forward backtests for the China live tier only.

### League (both markets)

- Live tier: up to 6 methods per market. Starts as: US = current 4 defaults + `pooled_lgbm_news`; China = top 5 by the first bench pass.
- Bench: every registered method for that market not in the live tier, including new candidates.
- Bench re-test: every 2nd Sunday, 13:00 ET (after the China weekly run). Cheaper than live backtests: walk-forward over the **last 3 years only** (≈52 folds instead of ≈620), same 21-day step and 400-day train window. Live-tier methods are scored on the same 3-year window in this pass so the comparison is apples to apples.
- Ranking score: out-of-sample Sharpe after costs (10 bps per side US, 15 bps per side China incl. stamp duty), tie-break buy-precision. Methods with < 20 trades in the window are ineligible for live.
- Promotion: a bench method replaces the weakest live method only if it beats it on the score in **2 consecutive** bench passes (stops flip-flopping). Demotion is the same swap. Changes are written to `markets/<m>/data/cache/league.json` with history; the live set is read from there instead of `default_strategies` (config value becomes the seed only).
- Near-duplicates (user's rule): two methods count as "highly similar" when, over the 3-year bench window, the correlation of their daily per-ticker buy-probability ranks is ≥ 0.85 **or** the overlap (Jaccard) of their buy picks is ≥ 0.70. Within each similar group only the best-scoring method may be live; the others are marked `duplicate_of: <winner>` on the bench, still shown, still re-tested (they may take over if the winner degrades). Thresholds live in config so they can be tuned.
- Retirement (user's rule): if a duplicate stays similar to the same winner for over a month (3 consecutive bench passes, i.e. ≥ 28 days), the lower-scoring one is retired: no more training or testing, removed from the dashboard. Retirement is a flag in `league.json` (`retired_on`, `retired_because`), its last results are archived to `cache/retired/`; the code and `strategies.yaml` entry stay so it can be restored by clearing the flag. Nothing is deleted.
- Bench pass runtime: ~10 min per ML method per market on 52 folds (scaled from the measured ~1.9 h per 620 folds), rule-based methods are seconds, so ≈ 5 h per market, both markets sequential. Hard stop Mon 04:30 ET (`RuntimeMaxSec`), before the China daily at 05:30. If stopped, untested methods keep their previous result and date and are tested first next pass.

### New methods to research and add (bench candidates)

My shortlist; each must run in 2 cores/8 GB and use only existing data:

1. LightGBM LambdaRank cross-sectional ranker on news+macro features (buys the top decile per day rather than a fixed probability threshold).
2. News momentum/surprise: change in 5-day sentiment vs 60-day baseline, interacted with abnormal article count.
3. Stacked meta-ensemble over the live tier's out-of-fold probabilities (logistic stacker).
4. Regime-gated model: trains separate weights for high/low volatility regimes (VIX for US, realized CSI vol for China).
5. China-only: policy/liquidity overlay (LPR/RRR change days, margin balance trend) and limit-up/limit-down aware labels (drop untradeable limit days from entries).

Only methods that win promotion go live; the rest stay benched with results visible.

### Dashboard

- Market switch (US / China) in the sidebar, applying to all pages.
- Backtest Results and Leaderboard: "Live" section first (sorted by score), "Bench" collapsed table with score, Sharpe, precision, trades, **last tested** date, and a trend arrow vs the previous pass.
- Every result shows its test date; anything older than 15 days (bench) or 8 days (live) gets a "stale" badge.
- Live Signals / Combined Signal: China tickers show name + code; times in ET, with Beijing time alongside on China pages.
- System Status: add China jobs, B's China news job, and the bench pass, with last result and next run.

## Schedule summary (ET)

| When | Server | Job |
|---|---|---|
| Mon–Fri 03:30 (16:30 Beijing) | B | China news fetch + score + sync |
| Mon–Fri 05:30 | A | China daily |
| Mon–Fri 18:30 | A | US daily (unchanged) |
| Daily 04:30 | A | Backup (now includes China data) |
| Sat 03:00 | A | US weekly (unchanged) |
| Sun 03:00 | A | China weekly |
| Every 2nd Sun 13:00 | A | Bench pass, both markets |

## Left to the builder

File layout, test structure, exact akshare call signatures, whether China news and US news share one module with a market parameter (preferred) or two.
