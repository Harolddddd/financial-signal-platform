# Dashboard Market Switching — Design

## Problem

The Streamlit dashboard (`dashboard/`) is hardwired to the US market: `dashboard/ui_config.py`'s `PARQUET_DIR`/`REGISTRY_DIR` and `dashboard/data_loader.py`'s `CACHE_DIR` are all module-level constants resolved from `get_market("us")`. The China full-universe-training plan (2026-07-31) produced a complete, real China dataset — 500 CSI 500 tickers, 3 trained models, 32 backtested strategies, a leaderboard, and live signals, all under `markets/china/data/cache/` and `markets/china/data/registry/` — but nothing in the dashboard reads it. The app can currently only ever display US data, regardless of what's on disk.

## Goals

- A market selector on the home page (`dashboard/app.py`) that governs all 5 pages (`app.py` + `pages/1..5`) for the rest of the session.
- Every page — Data Overview, Model Leaderboard, Backtest Results, Live Signals, Combined Signal — reads from the selected market's cache/feature/registry directories.
- Price labels and formatting reflect the selected market's currency (USD vs CNY) instead of a hardcoded `$`/"USD".
- No duplication of page files per market.

## Non-Goals

- No changes to how the underlying data is produced (`scripts/train_models.py`, `scripts/precompute_dashboard.py`, etc.) — this is a read-side change only.
- No support for markets beyond what's already in `config.markets.MARKETS` (currently `us`, `china`) — the selector is driven by that dict, so adding a third market later needs no dashboard code change.
- No fix for the known `build_live_features()`/`step_signals()` memory issue on the China data-generation side (tracked separately in the plan's Deviations section) — out of scope here.
- No live-compute fallback testing against China specifically beyond what naturally falls out of parameterizing the existing fallback paths (the China cache already exists, so the fallback path is expected to be rarely exercised for China in practice, same as it rarely is for US today).

## Architecture

**New module: `dashboard/market_state.py`.** Two functions:
- `render_market_selector() -> str` — called once, from `app.py` only. Renders a `st.selectbox` built from `config.markets.MARKETS` (options are each market's `.label`, e.g. "China A-Share (CSI 500)"), writes the chosen key (`"us"`/`"china"`) to `st.session_state["market"]`, and returns it. Defaults to `"us"` on first load.
- `get_selected_market() -> str` — called at the top of every page in `pages/`. Returns `st.session_state.get("market", "us")` — so a page reached without visiting `app.py` first (a direct sidebar click, since Streamlit's multipage sidebar lists every page regardless of navigation history) silently defaults to US, matching today's behavior exactly.

Both functions are thin — no new state beyond the one session_state key `"market"`.

**`dashboard/ui_config.py`:** `PARQUET_DIR`/`REGISTRY_DIR` module constants are replaced by `get_paths(market: str) -> tuple[Path, Path]`, mirroring the `_market_paths()` helper already used in `scripts/train_models.py`/`scripts/precompute_dashboard.py`. `FEATURE_COLS`/`OHLCV_COLS`/`CONFIDENCE_THRESHOLD`/`GRADE_COLORS` are market-independent and stay as-is.

**`dashboard/data_loader.py`:** `CACHE_DIR` module constant is replaced by `get_cache_dir(market: str) -> Path`. Every public function (`get_data_summary`, `get_leaderboard`, `get_backtest_result`, `get_live_signals`, `get_combined_ratings`) gains a `market: str` parameter, used internally wherever `CACHE_DIR` was previously referenced. `get_combined_ratings()` currently takes zero arguments — it gains `market: str` too.

**Each page (`app.py` + `pages/1..5`):**
- `app.py` calls `render_market_selector()` near the top and shows the resulting label.
- Each of the 5 pages calls `get_selected_market()` at the top, threads the result into its `ui_config.get_paths(market)` / `data_loader.get_*(..., market=market)` calls, and includes the returned market in its `@st.cache_data`-wrapped function's arguments (Streamlit's cache keys on args, so US and China results never collide in cache — no manual cache-key work needed). Each page shows a small `st.caption(f"Market: {label}")` near its header since the selector control itself only lives on `app.py`.
- `pages/4_Live_Signals.py` currently imports its own separate `CACHE_DIR` constant (duplicating `data_loader.py`'s) — this is folded into the same `get_cache_dir(market)` call, removing the duplication.

**Currency-aware formatting:** `config.markets.get_market(market).currency` (`"USD"`/`"CNY"`) drives:
- `pages/1_Data_Overview.py`'s price-chart Y-axis title ("Price (USD)" → "Price (CNY)" for China).
- The `$`-prefixed price formatting in `pages/4_Live_Signals.py` and `pages/5_Combined_Signal.py` (currently `f"${x:.2f}"` in three places) — replaced with a small shared helper (e.g. `dashboard/market_state.py`'s `format_price(value: float, market: str) -> str`) that prefixes `$` for USD and `¥` for CNY.

## Data Flow

1. User opens the app → lands on `app.py` → `render_market_selector()` shows "United States (S&P 500)" / "China A-Share (CSI 500)", defaults to US, writes choice to `session_state`.
2. User navigates to any page → that page's top-of-script code calls `get_selected_market()` → gets `"us"` or `"china"` back.
3. Page resolves `get_paths(market)` / `get_cache_dir(market)` and passes `market` into its `data_loader` calls.
4. `data_loader.py` functions read from the market-correct cache JSON (`markets/<market>/data/cache/*.json`), falling back to live compute against the market-correct feature parquet dir only if the cache file is missing — same fallback behavior as today, just market-parameterized.
5. Switching the selector on `app.py` and revisiting a page re-triggers all of the above with the new market; Streamlit's per-argument caching means both markets' results can be cached simultaneously without cross-contamination.

## Error Handling

No new failure modes beyond what each page already handles (missing cache → live-compute fallback → `st.warning`/`st.stop()` if that's also empty). The only new edge case is a market with no cache data at all (not currently possible — both `us` and `china` have full caches) — in that case the existing "No strategies found" / "No cached signals/leaderboard found" messages already in each page fire naturally, since `get_cache_dir(market)` still points at a real (just-empty) directory rather than raising.

## Testing

- Unit tests for `dashboard/market_state.py` (`get_selected_market()` defaults to `"us"` with no session_state set; returns the stored value once set) and for `dashboard/ui_config.get_paths()` / `dashboard/data_loader.get_cache_dir()` (both markets resolve to the correct `markets/<market>/data/...` paths, following the same pattern as the existing `tests/unit/test_data_paths.py`).
- Manual verification: run the dashboard locally (`streamlit run dashboard/app.py`), confirm each of the 5 pages renders correctly for both US (unchanged from today) and China (new — 500-ticker leaderboard, backtests, live signals, combined ratings, all in CNY-formatted prices), and that switching markets and clicking between pages doesn't require a full app restart.
