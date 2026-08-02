# Dashboard Market Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a market selector to the Streamlit dashboard's home page that governs all 5 pages for the rest of the session, so the real China CSI 500 data already on disk (leaderboard, backtests, live signals, registry) becomes visible instead of the dashboard only ever showing US data.

**Architecture:** A new `dashboard/market_state.py` module holds the selector widget (`render_market_selector()`, called only from `app.py`) and a session-state reader (`get_selected_market()`, called from every page). `dashboard/ui_config.py` and `dashboard/data_loader.py` gain market-parameterized path/cache-dir lookups alongside (not replacing) their existing US-default constants, since 8 unrelated scripts under `scripts/` import those constants directly. Each of the 5 pages threads the selected market through its existing calls instead of using the hardcoded constants.

**Tech Stack:** Python, Streamlit, Polars, pytest.

## Global Constraints

- `dashboard/ui_config.py`'s `PARQUET_DIR` and `REGISTRY_DIR` constants (currently resolving `get_market("us")`) **must not change** — `scripts/train_models.py`, `scripts/precompute_dashboard.py`, `scripts/precompute_full.py`, `scripts/precompute_new_strategies.py`, `scripts/run_signals_isolated.py`, `scripts/signal_one_strategy.py`, `scripts/train_new_models.py`, and `scripts/train_lstm_only.py` all import them directly and are out of scope for this plan.
- `dashboard/data_loader.py`'s `CACHE_DIR` constant has no consumers outside this file and one direct test — it is removed and replaced by `get_cache_dir(market)`, not kept alongside.
- No changes to `scripts/*.py`, `config/markets.py`, or any data-generation code — this plan is dashboard-read-side only.
- `config.markets.MARKETS` (currently `{"us": ..., "china": ...}`) is the sole source of truth for which markets the selector offers — do not hardcode a market list anywhere in `dashboard/`.
- Real committed data already exists for both markets and is used directly in tests (no mocking): `markets/us/data/cache/data_summary.json` has `n_tickers: 492`; `markets/china/data/cache/data_summary.json` has `n_tickers: 500`; `markets/china/data/cache/leaderboard.json` has 32 grades.
- Every `@st.cache_data`-wrapped function in the pages must take `market` as an **explicit argument**, even where the code only needs a closure-captured `parquet_dir` — Streamlit's cache keys on the decorated function's arguments only, not on closed-over module-level variables, so omitting `market` from the signature would silently serve stale cached results from whichever market was viewed first after a market switch.

---

### Task 1: `dashboard/market_state.py` — selector, session reader, currency formatter

**Files:**
- Create: `dashboard/market_state.py`
- Test: `tests/unit/test_dashboard_market_switching.py` (new file)

**Interfaces:**
- Consumes: `config.markets.MARKETS: dict[str, MarketConfig]`, `config.markets.get_market(key) -> MarketConfig` (existing).
- Produces: `render_market_selector() -> str` (call once, from `app.py` only), `get_selected_market() -> str` (call from every page), `format_price(value: float, market: str) -> str` — all consumed by later tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_dashboard_market_switching.py`:

```python
def test_get_selected_market_defaults_to_us(monkeypatch):
    from dashboard import market_state
    monkeypatch.setattr(market_state.st, "session_state", {})
    assert market_state.get_selected_market() == "us"


def test_get_selected_market_returns_stored_value(monkeypatch):
    from dashboard import market_state
    monkeypatch.setattr(market_state.st, "session_state", {"market": "china"})
    assert market_state.get_selected_market() == "china"


def test_format_price_usd():
    from dashboard.market_state import format_price
    assert format_price(123.456, "us") == "$123.46"


def test_format_price_cny():
    from dashboard.market_state import format_price
    assert format_price(123.456, "china") == "¥123.46"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.market_state'`.

- [ ] **Step 3: Create `dashboard/market_state.py`**

```python
# dashboard/market_state.py
from __future__ import annotations

import streamlit as st

from config.markets import MARKETS, get_market

_SESSION_KEY = "market"
_DEFAULT_MARKET = "us"

_CURRENCY_SYMBOLS = {"USD": "$", "CNY": "¥"}


def render_market_selector() -> str:
    """Render the market dropdown. Call this once, from app.py only —
    every other page reads the resulting choice via get_selected_market()."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = _DEFAULT_MARKET

    keys = list(MARKETS.keys())
    st.selectbox(
        "Market",
        keys,
        format_func=lambda key: MARKETS[key].label,
        key=_SESSION_KEY,
    )
    return st.session_state[_SESSION_KEY]


def get_selected_market() -> str:
    """Read the market chosen on app.py. Defaults to "us" if no selection
    has been made yet in this session (e.g. a page reached directly)."""
    return st.session_state.get(_SESSION_KEY, _DEFAULT_MARKET)


def format_price(value: float, market: str) -> str:
    symbol = _CURRENCY_SYMBOLS.get(get_market(market).currency, "$")
    return f"{symbol}{value:.2f}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/market_state.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: add dashboard/market_state.py (selector, session reader, currency formatter)"
```

---

### Task 2: `dashboard/ui_config.py` — add `get_paths(market)`

**Files:**
- Modify: `dashboard/ui_config.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `config.markets.get_market(market) -> MarketConfig` (already imported in this file).
- Produces: `get_paths(market: str) -> tuple[Path, Path]` — returns `(feature_dir, registry_dir)`. Consumed by Tasks 5-8 (the pages). `PARQUET_DIR`/`REGISTRY_DIR`/`FEATURE_COLS`/`OHLCV_COLS`/`CONFIDENCE_THRESHOLD`/`GRADE_COLORS` are all unchanged — existing callers elsewhere in the repo are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_ui_config_get_paths_for_us():
    from dashboard.ui_config import get_paths
    from config.markets import get_market
    feature_dir, registry_dir = get_paths("us")
    assert feature_dir == get_market("us").data_root / "features"
    assert registry_dir == get_market("us").data_root / "registry"


def test_ui_config_get_paths_for_china():
    from dashboard.ui_config import get_paths
    from config.markets import get_market
    feature_dir, registry_dir = get_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert registry_dir == get_market("china").data_root / "registry"


def test_ui_config_paths_still_exported_for_other_scripts():
    # scripts/train_models.py and 7 others import PARQUET_DIR/REGISTRY_DIR
    # directly — this must keep working unchanged.
    from dashboard.ui_config import PARQUET_DIR, REGISTRY_DIR
    from config.markets import get_market
    assert PARQUET_DIR == get_market("us").data_root / "features"
    assert REGISTRY_DIR == get_market("us").data_root / "registry"
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_ui_config_get_paths_for_us tests/unit/test_dashboard_market_switching.py::test_ui_config_get_paths_for_china -v`
Expected: FAIL with `ImportError: cannot import name 'get_paths'`. (`test_ui_config_paths_still_exported_for_other_scripts` will already PASS — it's a guard against a future regression, not new behavior.)

- [ ] **Step 3: Add `get_paths()` to `dashboard/ui_config.py`**

Change the top of the file from:

```python
# dashboard/ui_config.py
from config.markets import get_market

PARQUET_DIR  = get_market("us").data_root / "features"
REGISTRY_DIR = get_market("us").data_root / "registry"
```

to:

```python
# dashboard/ui_config.py
from pathlib import Path

from config.markets import get_market

PARQUET_DIR  = get_market("us").data_root / "features"
REGISTRY_DIR = get_market("us").data_root / "registry"


def get_paths(market: str) -> tuple[Path, Path]:
    market_cfg = get_market(market)
    return market_cfg.data_root / "features", market_cfg.data_root / "registry"
```

(Leave everything below `REGISTRY_DIR` — `OHLCV_COLS`, `FEATURE_COLS`, `CONFIDENCE_THRESHOLD`, `GRADE_COLORS` — untouched.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS (7 so far).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ui_config.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: add get_paths(market) to dashboard/ui_config.py"
```

---

### Task 3: `dashboard/data_loader.py` — market-parameterize every public function

**Files:**
- Modify: `dashboard/data_loader.py`
- Modify: `tests/unit/test_data_paths.py:16-19` (existing test imports the `CACHE_DIR` constant this task removes)
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `config.markets.get_market(market) -> MarketConfig` (already imported in this file).
- Produces: `get_cache_dir(market: str) -> Path` (replaces the `CACHE_DIR` constant — no other file imports `CACHE_DIR` directly, confirmed by repo-wide grep). `get_data_summary(parquet_dir, market="us")`, `get_leaderboard(parquet_dir, ohlcv_cols, feature_cols, market="us")`, `get_backtest_result(strategy_name, parquet_dir, ohlcv_cols, feature_cols, market="us")`, `get_live_signals(parquet_dir, ohlcv_cols, feature_cols, confidence_threshold=0.75, market="us")`, `get_combined_ratings(market="us")` — each gains a trailing `market` keyword argument defaulting to `"us"` (existing zero-market-arg call shape still no longer applies once Tasks 5-8 update every caller, but the default keeps the function importable/callable exactly as before for safety). Consumed by Tasks 5-8.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_data_loader_get_cache_dir_for_us():
    from dashboard.data_loader import get_cache_dir
    from config.markets import get_market
    assert get_cache_dir("us") == get_market("us").data_root / "cache"


def test_data_loader_get_cache_dir_for_china():
    from dashboard.data_loader import get_cache_dir
    from config.markets import get_market
    assert get_cache_dir("china") == get_market("china").data_root / "cache"


def test_get_data_summary_routes_to_china_cache():
    from dashboard.data_loader import get_data_summary
    from dashboard.ui_config import get_paths
    parquet_dir, _ = get_paths("china")
    summary = get_data_summary(parquet_dir, market="china")
    assert summary["n_tickers"] == 500


def test_get_data_summary_routes_to_us_cache_by_default():
    from dashboard.data_loader import get_data_summary
    from dashboard.ui_config import PARQUET_DIR
    summary = get_data_summary(PARQUET_DIR)
    assert summary["n_tickers"] == 492


def test_get_leaderboard_routes_to_china_cache():
    from dashboard.data_loader import get_leaderboard
    from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS
    parquet_dir, _ = get_paths("china")
    leaderboard = get_leaderboard(parquet_dir, OHLCV_COLS, FEATURE_COLS, market="china")
    assert len(leaderboard) == 32


def test_get_combined_ratings_routes_to_china_cache():
    from dashboard.data_loader import get_combined_ratings
    summary_rows, detail_by_ticker = get_combined_ratings(market="china")
    assert len(summary_rows) > 0
    assert isinstance(detail_by_ticker, dict)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -k "get_cache_dir or routes_to" -v`
Expected: FAIL — `get_cache_dir` doesn't exist yet, and `get_data_summary("china"...)`/`get_leaderboard(..., market=...)`/`get_combined_ratings(market=...)` raise `TypeError: unexpected keyword argument 'market'`.

- [ ] **Step 3: Rewrite `dashboard/data_loader.py`**

Change:

```python
CACHE_DIR = get_market("us").data_root / "cache"
```

to:

```python
def get_cache_dir(market: str) -> Path:
    return get_market(market).data_root / "cache"
```

Then update each public function's signature and its `CACHE_DIR` reference:

```python
def get_data_summary(parquet_dir: Path, market: str = "us") -> dict:
    cached = _load_cache(get_cache_dir(market) / "data_summary.json")
    ...  # body unchanged below this line
```

```python
def get_leaderboard(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    market: str = "us",
) -> list[ModelGrade]:
    cached = _load_cache(get_cache_dir(market) / "leaderboard.json")
    ...  # body unchanged below this line
```

```python
def get_backtest_result(
    strategy_name: str,
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    market: str = "us",
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    cached = _load_cache(get_cache_dir(market) / f"backtest_{_safe(strategy_name)}.json")
    ...  # body unchanged below this line
```

```python
def get_live_signals(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
    market: str = "us",
) -> list[LiveSignal]:
    cached = _load_cache(get_cache_dir(market) / "signals.json")
    ...  # body unchanged below this line
```

```python
def get_combined_ratings(market: str = "us") -> tuple[list[dict], dict[str, list[dict]]]:
    """..."""  # docstring unchanged
    leaderboard = _load_cache(get_cache_dir(market) / "leaderboard.json")
    signals = _load_cache(get_cache_dir(market) / "signals.json")
    ...  # body unchanged below this line
```

Every other line in the file (the four dataclass-conversion helpers, `_load_cache`, `_safe`, and the rest of each function's body below the `cached = ...` line) stays exactly as it is today — only the `CACHE_DIR` references and each function's parameter list change.

- [ ] **Step 4: Update the existing test that imports the removed `CACHE_DIR` constant**

In `tests/unit/test_data_paths.py`, change:

```python
def test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data():
    from dashboard.data_loader import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
    assert CACHE_DIR.exists()
```

to:

```python
def test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data():
    from dashboard.data_loader import get_cache_dir
    assert get_cache_dir("us") == _US_ROOT / "cache"
    assert get_cache_dir("us").exists()
```

- [ ] **Step 5: Run the full set of touched tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py tests/unit/test_data_paths.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/data_loader.py tests/unit/test_data_paths.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: market-parameterize dashboard/data_loader.py (get_cache_dir + market kwarg on every public function)"
```

---

### Task 4: `dashboard/app.py` — render the selector

**Files:**
- Modify: `dashboard/app.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.render_market_selector()` (Task 1), `config.markets.get_market()`.
- Produces: nothing new consumed by later tasks — `app.py` is a leaf.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
from pathlib import Path


def test_app_py_renders_market_selector():
    # app.py runs Streamlit calls at module level, so it can't be imported
    # directly in a test — verify the source text instead, same approach
    # already used for dashboard/pages/4_Live_Signals.py in test_data_paths.py.
    source = Path("dashboard/app.py").read_text()
    assert "render_market_selector" in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_app_py_renders_market_selector -v`
Expected: FAIL — `render_market_selector` not present in `dashboard/app.py`.

- [ ] **Step 3: Update `dashboard/app.py`**

Replace the full file content with:

```python
import streamlit as st

from config.markets import get_market
from dashboard.market_state import render_market_selector

st.set_page_config(
    page_title="Financial Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Signal Platform")

market = render_market_selector()
st.caption(f"Active market: {get_market(market).label}")

st.markdown("""
Navigate using the sidebar:

- **Data Overview** — data ingestion status, ticker universe, date ranges
- **Strategy Leaderboard** — all strategies ranked by composite grade (walk-forward backtest)
- **Backtest Results** — fold-by-fold performance metrics per strategy
- **Live Signals** — today's Buy/Hold/Sell recommendations with confidence scores
""")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: render market selector on dashboard home page"
```

---

### Task 5: `dashboard/pages/1_Data_Overview.py` — market + currency aware

**Files:**
- Modify: `dashboard/pages/1_Data_Overview.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.get_selected_market()` (Task 1), `dashboard.ui_config.get_paths(market)` (Task 2), `dashboard.data_loader.get_data_summary(parquet_dir, market=...)` (Task 3).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_data_overview_page_is_market_aware():
    source = Path("dashboard/pages/1_Data_Overview.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_data_overview_page_is_market_aware -v`
Expected: FAIL — page still imports `PARQUET_DIR` directly and has no `get_selected_market`/`get_paths(` usage.

- [ ] **Step 3: Update `dashboard/pages/1_Data_Overview.py`**

Replace the full file content with:

```python
import streamlit as st
import plotly.graph_objects as go
from config.markets import get_market
from dashboard.market_state import get_selected_market
from dashboard.ui_config import get_paths, FEATURE_COLS
from dashboard.data_loader import get_data_summary
from src.features.duckdb_client import load_training_data

st.set_page_config(page_title="Data Overview", layout="wide")
st.header("Data Overview")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)


@st.cache_data(ttl=3600)
def _summary(market: str):
    return get_data_summary(parquet_dir, market=market)


@st.cache_data(ttl=3600)
def _load_ticker_df(ticker: str, market: str):
    return load_training_data(parquet_dir, tickers=[ticker])


summary = _summary(market)

col1, col2, col3 = st.columns(3)
col1.metric("Tickers", summary["n_tickers"])
col2.metric("Total Rows", f"{summary['n_rows']:,}")
col3.metric("Date Range", f"{summary['date_range_start'][:10]} → {summary['date_range_end'][:10]}")

st.divider()

ticker = st.selectbox("Select ticker to preview", summary["tickers"])
if ticker:
    df = _load_ticker_df(ticker, market)
    if not df.is_empty():
        fig = go.Figure()
        times = df["time"].to_list()
        closes = df["close"].to_list()
        fig.add_trace(go.Scatter(x=times, y=closes, mode="lines", name="Close"))
        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=times, y=df["sma_20"].to_list(),
                                     mode="lines", name="SMA 20", line=dict(dash="dot")))
        currency = get_market(market).currency
        fig.update_layout(title=f"{ticker} Price + SMA 20",
                          xaxis_title="Date", yaxis_title=f"Price ({currency})")
        st.plotly_chart(fig, use_container_width=True)

        if "sent_pos_avg_5d" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=times, y=df["sent_pos_avg_5d"].to_list(),
                                  name="5d Avg Positive Sentiment"))
            fig2.update_layout(title=f"{ticker} News Sentiment (5d Rolling)",
                                xaxis_title="Date", yaxis_title="Positive Sentiment Score")
            st.plotly_chart(fig2, use_container_width=True)
```

Note `_summary` and `_load_ticker_df` both take `market` as an explicit cached-function argument even though the body only uses the closure-captured `parquet_dir` — this is required per the Global Constraints note so Streamlit's cache doesn't serve a stale market's data after a switch.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/1_Data_Overview.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: make Data Overview page market- and currency-aware"
```

---

### Task 6: `dashboard/pages/2_Model_Leaderboard.py` + `dashboard/pages/3_Backtest_Results.py`

**Files:**
- Modify: `dashboard/pages/2_Model_Leaderboard.py`
- Modify: `dashboard/pages/3_Backtest_Results.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.get_selected_market()` (Task 1), `dashboard.ui_config.get_paths(market)` (Task 2), `dashboard.data_loader.get_leaderboard(..., market=...)` / `get_backtest_result(..., market=...)` (Task 3). Neither page shows a price value, so no `format_price` usage here.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_leaderboard_page_is_market_aware():
    source = Path("dashboard/pages/2_Model_Leaderboard.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source


def test_backtest_results_page_is_market_aware():
    source = Path("dashboard/pages/3_Backtest_Results.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -k "leaderboard_page or backtest_results_page" -v`
Expected: both FAIL.

- [ ] **Step 3: Update `dashboard/pages/2_Model_Leaderboard.py`**

Replace the full file content with:

```python
# dashboard/pages/2_Model_Leaderboard.py
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from config.markets import get_market
from dashboard.market_state import get_selected_market
from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_leaderboard

st.set_page_config(page_title="Strategy Leaderboard", layout="wide")
st.header("Strategy Leaderboard")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)


@st.cache_data(ttl=1800)
def _leaderboard(market: str):
    return get_leaderboard(parquet_dir, OHLCV_COLS, FEATURE_COLS, market=market)


with st.spinner("Computing grades..."):
    leaderboard = _leaderboard(market)

if not leaderboard:
    st.warning("No strategies found. Check src/strategies/strategies.yaml.")
    st.stop()

rows = [{
    "Rank": i + 1,
    "Strategy": g.model_name,
    "Grade": g.grade.value,
    "Score": f"{g.composite_score:.3f}",
    "Precision Buy": f"{g.metrics.precision_buy:.3f}",
    "Sharpe": f"{g.metrics.sharpe_ratio:.2f}",
    "Max Drawdown": f"{g.metrics.max_drawdown_pct:.1%}",
    "Win Rate": f"{g.metrics.win_rate:.1%}",
    "Trades": g.metrics.n_trades,
} for i, g in enumerate(leaderboard)]

df = pl.DataFrame(rows)
st.dataframe(df.to_pandas(), use_container_width=True, hide_index=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=[g.model_name for g in leaderboard],
        y=[g.metrics.precision_buy for g in leaderboard],
        marker_color=[GRADE_COLORS[g.grade.value] for g in leaderboard],
    ))
    fig.update_layout(title="Precision (Buy class)", xaxis_title="Strategy",
                      yaxis_title="Precision", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Bar(
        x=[g.model_name for g in leaderboard],
        y=[g.metrics.sharpe_ratio for g in leaderboard],
        marker_color=[GRADE_COLORS[g.grade.value] for g in leaderboard],
    ))
    fig2.update_layout(title="Sharpe Ratio", xaxis_title="Strategy",
                       yaxis_title="Sharpe")
    st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 4: Update `dashboard/pages/3_Backtest_Results.py`**

Replace the full file content with:

```python
# dashboard/pages/3_Backtest_Results.py
import streamlit as st
import plotly.graph_objects as go
from config.markets import get_market
from dashboard.market_state import get_selected_market
from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_backtest_result
from src.strategies.registry import list_strategies

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Backtest Results")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)

strategy_names = list_strategies()
if not strategy_names:
    st.warning("No strategies in registry. Check src/strategies/strategies.yaml.")
    st.stop()

selected = st.selectbox("Select strategy", strategy_names)


@st.cache_data(ttl=1800)
def _backtest(strategy_name: str, market: str):
    return get_backtest_result(strategy_name, parquet_dir, OHLCV_COLS, FEATURE_COLS, market=market)


with st.spinner(f"Running walk-forward backtest for {selected}..."):
    wf_result, grade = _backtest(selected, market)

color = GRADE_COLORS[grade.grade.value]
st.markdown(f"### Grade: <span style='color:{color};font-size:2em'>{grade.grade.value}</span> "
            f"(score: {grade.composite_score:.3f})", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
last = wf_result.folds[-1].metrics
c1.metric("Precision Buy", f"{last.precision_buy:.3f}")
c2.metric("Sharpe Ratio", f"{last.sharpe_ratio:.2f}")
c3.metric("Max Drawdown", f"{last.max_drawdown_pct:.1%}")
c4.metric("Win Rate", f"{last.win_rate:.1%}")

st.divider()

fold_labels = [f"Fold {f.fold}" for f in wf_result.folds]
precisions  = [f.metrics.precision_buy for f in wf_result.folds]
sharpes     = [f.metrics.sharpe_ratio for f in wf_result.folds]
n_trades    = [f.n_trades for f in wf_result.folds]

fig = go.Figure()
fig.add_trace(go.Scatter(x=fold_labels, y=precisions, mode="lines+markers", name="Precision Buy"))
fig.add_trace(go.Scatter(x=fold_labels, y=sharpes, mode="lines+markers",
                         name="Sharpe", yaxis="y2"))
fig.update_layout(
    title="Walk-Forward Performance by Fold",
    yaxis=dict(title="Precision", range=[0, 1]),
    yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade Count per Fold")
fig2 = go.Figure(go.Bar(x=fold_labels, y=n_trades))
fig2.update_layout(xaxis_title="Fold", yaxis_title="# Trades")
st.plotly_chart(fig2, use_container_width=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/pages/2_Model_Leaderboard.py dashboard/pages/3_Backtest_Results.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: make Leaderboard and Backtest Results pages market-aware"
```

---

### Task 7: `dashboard/pages/4_Live_Signals.py` — market + currency aware, drop duplicate CACHE_DIR

**Files:**
- Modify: `dashboard/pages/4_Live_Signals.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.get_selected_market()` / `format_price()` (Task 1), `dashboard.ui_config.get_paths(market)` (Task 2), `dashboard.data_loader.get_cache_dir(market)` / `get_live_signals(..., market=...)` (Task 3).

This page currently defines its own `CACHE_DIR = get_market("us").data_root / "cache"` (duplicating `data_loader.py`'s, now-removed, constant) instead of using a shared helper — that duplication is removed here.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_live_signals_page_is_market_aware():
    source = Path("dashboard/pages/4_Live_Signals.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert 'CACHE_DIR = get_market("us")' not in source
    assert "format_price" in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_live_signals_page_is_market_aware -v`
Expected: FAIL.

- [ ] **Step 3: Update `dashboard/pages/4_Live_Signals.py`**

Replace the full file content with:

```python
import json

import pandas as pd
import streamlit as st

from config.markets import get_market
from dashboard.market_state import get_selected_market, format_price
from dashboard.ui_config import CONFIDENCE_THRESHOLD, get_paths, OHLCV_COLS, FEATURE_COLS
from dashboard.data_loader import get_cache_dir, get_live_signals

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

# Load raw cache so we can surface the strategy field.
_raw = None
_cache_path = get_cache_dir(market) / "signals.json"
if _cache_path.exists():
    _raw = json.loads(_cache_path.read_text())

if _raw:
    raw_signals = [
        s for s in _raw["signals"]
        if s["signal"] == "Buy" and s["confidence"] >= threshold
    ]
    generated_at = _raw.get("generated_at", "unknown")
else:
    with st.spinner("Generating signals..."):
        live = get_live_signals(parquet_dir, OHLCV_COLS, FEATURE_COLS, threshold, market=market)
    raw_signals = [
        {
            "ticker": s.ticker,
            "date": s.date,
            "signal": s.signal.value,
            "confidence": s.confidence,
            "entry_price": s.entry_price,
            "position_size": s.position_size,
            "strategy": "—",
        }
        for s in live
    ]
    generated_at = "live"

st.caption(f"Cache generated: {generated_at}")

if not raw_signals:
    st.info("No Buy signals above the current confidence threshold.")
    st.stop()

st.success(f"Found **{len(raw_signals)}** Buy signal(s) across all strategies")

# Build a tidy DataFrame grouped by ticker.
df = pd.DataFrame(raw_signals)[["ticker", "strategy", "confidence", "entry_price", "date"]]
df = df.sort_values(["ticker", "confidence"], ascending=[True, False])
df["confidence"] = df["confidence"].map(lambda x: f"{x:.1%}")
df["entry_price"] = df["entry_price"].map(lambda x: format_price(x, market))
df.columns = ["Ticker", "Strategy", "Confidence", "Entry Price", "Date"]

# Summary: tickers with the most strategy agreement
ticker_counts = df.groupby("Ticker").size().sort_values(ascending=False)
top_tickers = ticker_counts.head(10)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Top tickers by strategy agreement")
    st.bar_chart(top_tickers)

with col2:
    st.subheader("All signals")
    st.dataframe(df, use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Run the full touched-test set to verify everything still passes**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py tests/unit/test_data_paths.py -v`
Expected: all PASS — including the pre-existing `test_live_signals_page_no_longer_hardcodes_old_data_path` in `test_data_paths.py`, which asserts `'Path("data/cache")' not in source` (still true) and `"get_market(" in source` (still true — used in the `st.caption` line).

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/4_Live_Signals.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: make Live Signals page market- and currency-aware"
```

---

### Task 8: `dashboard/pages/5_Combined_Signal.py` — market + currency aware

**Files:**
- Modify: `dashboard/pages/5_Combined_Signal.py`
- Test: `tests/unit/test_dashboard_market_switching.py`

**Interfaces:**
- Consumes: `dashboard.market_state.get_selected_market()` / `format_price()` (Task 1), `dashboard.data_loader.get_combined_ratings(market=...)` (Task 3). This page has no `parquet_dir` dependency (it's cache-only, per its existing docstring), so `ui_config.get_paths()` isn't needed here.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_dashboard_market_switching.py`:

```python
def test_combined_signal_page_is_market_aware():
    source = Path("dashboard/pages/5_Combined_Signal.py").read_text()
    assert "get_selected_market" in source
    assert "get_combined_ratings(market=market)" in source
    assert "format_price" in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py::test_combined_signal_page_is_market_aware -v`
Expected: FAIL.

- [ ] **Step 3: Update `dashboard/pages/5_Combined_Signal.py`**

Replace the full file content with:

```python
import pandas as pd
import streamlit as st

from config.markets import get_market
from dashboard.market_state import get_selected_market, format_price
from dashboard.data_loader import get_combined_ratings

st.set_page_config(page_title="Combined Signal", layout="wide")
st.header("Combined Signal — Overall Rating by Stock")
st.caption(
    "Overall Rating = weighted average of every strategy's live Buy confidence "
    "for that ticker, weighted by the strategy's leaderboard composite score "
    "(stronger track-record strategies count for more). A strategy with no "
    "signal for a ticker isn't counted. Hold signals count as zero but still "
    "dilute the average, since \"no edge\" is itself an opinion."
)

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")


def _label(rating: float) -> str:
    if rating >= 12:
        return "Strong Buy"
    if rating >= 8:
        return "Buy"
    if rating >= 4:
        return "Neutral"
    return "Avoid"


summary_rows, detail_by_ticker = get_combined_ratings(market=market)

if not summary_rows:
    st.warning(
        "No cached signals/leaderboard found. Run "
        f"`python scripts/precompute_dashboard.py --market {market}` first."
    )
    st.stop()

df = pd.DataFrame(summary_rows)
df["Rating"] = df["overall_rating"].round(1)
df["Signal"] = df["Rating"].map(_label)
df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df["Entry Price"] = df["entry_price"].map(lambda x: format_price(x, market))
df = df.rename(columns={"ticker": "Ticker", "n_buy": "# Buy", "n_strategies": "# Strategies"})
df = df[["Ticker", "Signal", "Rating", "Entry Price", "Date", "# Buy", "# Strategies"]]

col1, col2, col3 = st.columns(3)
col1.metric("Tickers rated", len(df))
col2.metric("Strong Buy / Buy", int((df["Signal"].isin(["Strong Buy", "Buy"])).sum()))
col3.metric("Median rating", f"{df['Rating'].median():.1f}")

st.subheader(f"All {len(df)} tickers, ranked by overall rating")
st.caption("Click a row to see that stock's per-strategy breakdown below.")

event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.rows if event and event.selection else []

st.divider()

if selected_rows:
    ticker = df.iloc[selected_rows[0]]["Ticker"]
    st.subheader(f"Per-strategy contribution — {ticker}")

    detail = detail_by_ticker.get(ticker, [])
    ddf = pd.DataFrame(detail)
    ddf["confidence"] = ddf["confidence"].round(3)
    ddf["weight"] = ddf["weight"].round(3)
    ddf["contribution"] = ddf["contribution"].round(3)
    ddf["date"] = pd.to_datetime(ddf["date"]).dt.strftime("%Y-%m-%d")
    ddf["entry_price"] = ddf["entry_price"].map(lambda x: format_price(x, market))
    ddf = ddf[["strategy", "weight", "signal", "confidence", "contribution", "entry_price", "date"]]
    ddf.columns = ["Strategy", "Weight (composite score)", "Signal", "Confidence",
                   "Contribution", "Entry Price", "Date"]

    chart_col, table_col = st.columns([2, 3])
    with chart_col:
        st.bar_chart(ddf.set_index("Strategy")["Contribution"])
    with table_col:
        st.dataframe(ddf, use_container_width=True, hide_index=True)
else:
    st.info("Click a row in the table above to see that stock's per-strategy breakdown.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_dashboard_market_switching.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/5_Combined_Signal.py tests/unit/test_dashboard_market_switching.py
git commit -m "feat: make Combined Signal page market- and currency-aware"
```

---

### Task 9: Full regression + manual verification

**Files:**
- No source changes — this task verifies the finished work.

- [ ] **Step 1: Run the full unit test suite**

Run: `python -m pytest tests/unit -q`
Expected: pass count at or above the pre-plan baseline (366 passed, 15 pre-existing failures unrelated to this plan, 2 skipped — per the China full-universe-training plan's final state), plus the ~17 new tests added across Tasks 1-8 in `tests/unit/test_dashboard_market_switching.py`.

- [ ] **Step 2: Manual verification — launch the dashboard and check both markets**

Run: `streamlit run dashboard/app.py`

Checklist (all must hold before this task is done):
1. Home page shows the market dropdown, defaulted to "United States (S&P 500)", with an "Active market: United States (S&P 500)" caption.
2. Switch the dropdown to "China A-Share (CSI 500)". The caption updates immediately.
3. Navigate to **Data Overview**: shows "Market: China A-Share (CSI 500)", ~500 tickers, and the ticker-preview chart's Y-axis reads "Price (CNY)" for a China ticker (e.g. `600519.SS`).
4. Navigate to **Strategy Leaderboard**: shows 32 China strategies ranked, distinct from the US leaderboard (switch back to US and confirm the numbers differ).
5. Navigate to **Backtest Results**: pick any strategy, confirm fold-by-fold China data renders (not a US-shaped error).
6. Navigate to **Live Signals**: confirm China Buy signals render with `¥`-prefixed entry prices.
7. Navigate to **Combined Signal**: confirm China tickers are rated with `¥`-prefixed entry prices, and the per-strategy drill-down works on a China ticker.
8. Switch back to US on the home page, revisit each of the 5 pages again, and confirm all data reverts to the original US values (no China data leaking into US view, or vice versa — this is the cache-correctness property from the Global Constraints note).

- [ ] **Step 3: Commit** (only if Step 2 surfaced a fix — otherwise this task has no commit of its own)

If manual verification passes cleanly with no code changes needed, this task requires no commit — Tasks 1-8's commits are the complete deliverable. If a fix was needed, commit it with a message describing what manual verification caught.
