import streamlit as st
import plotly.graph_objects as go
from config.markets import get_market
from dashboard.market_state import get_selected_market
from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
import json

from dashboard.data_loader import get_backtest_result, get_cache_dir, get_league_table

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Backtest Results")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)

league = get_league_table(market)
section_of = {r["method"]: sec for sec in ("live", "bench") for r in league[sec]}
tested_on = {r["method"]: r["tested_on"] for sec in ("live", "bench") for r in league[sec]}
strategy_names = [r["method"] for r in league["live"]] + [r["method"] for r in league["bench"]]
if not strategy_names:
    st.warning("No strategies in registry. Check src/strategies/strategies.yaml.")
    st.stop()

selected = st.selectbox(
    "Select strategy", strategy_names, format_func=lambda n: f"{n}  ({section_of.get(n, '')})"
)
cache_dir = get_cache_dir(market)
st.caption(f"Last tested: {tested_on.get(selected) or 'never'}")

if not (cache_dir / f"backtest_{selected}.json").exists():
    # No full-history backtest (bench-only method): show its 3-year bench result
    # instead of computing a walk-forward inside the dashboard.
    bench_path = cache_dir / "bench" / f"bench_{selected}.json"
    if not bench_path.exists():
        st.info("Not tested yet. It will be scored in the next bench re-test.")
        st.stop()
    b = json.loads(bench_path.read_text(encoding="utf-8"))
    sc = b["score"]
    st.markdown(f"### Bench result (last {b['window_days'] // 365} years, after costs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Sharpe", f"{sc['net_sharpe']:.2f}", help="League score: yearly Sharpe of 5-day trades on every Buy signal, after round-trip costs. >0.5 decent, >1 strong.")
    c2.metric("Precision", f"{sc['precision']:.1%}", help="Share of those trades that made money after costs.")
    c3.metric("Trades", sc["n_trades"], help="Buy signals scored in the test window.")
    c4.metric("Mean net return / trade", f"{sc['mean_net_return']:.2%}")
    if b.get("ran_on") or b.get("runtime_s"):
        st.caption(f"Tested on server {b.get('ran_on') or 'A'} in {(b.get('runtime_s') or 0) / 60:.1f} min, {b.get('tested_on', '')}.")
    st.caption("Full fold-by-fold history is computed only for live methods (weekly).")
    st.stop()


@st.cache_data(ttl=1800)
def _backtest(strategy_name: str, market: str):
    return get_backtest_result(strategy_name, parquet_dir, OHLCV_COLS, FEATURE_COLS, market=market)


with st.spinner(f"Running walk-forward backtest for {selected}..."):
    wf_result, grade = _backtest(selected, market)

color = GRADE_COLORS[grade.grade.value]
st.markdown(f"### Grade: <span style='color:{color};font-size:2em'>{grade.grade.value}</span> "
            f"(score: {grade.composite_score:.3f} of 1 = 40% Buy precision + 30% Sharpe + 30% drawdown safety)", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
last = wf_result.folds[-1].metrics
c1.metric("Precision Buy", f"{last.precision_buy:.3f}", help="Of the days the model said Buy, share where the price went up (last fold).")
c2.metric("Sharpe Ratio", f"{last.sharpe_ratio:.2f}", help="Risk-adjusted return of following the signals, yearly scale, before costs (last fold).")
c3.metric("Max Drawdown", f"{last.max_drawdown_pct:.1%}", help="Worst peak-to-trough loss (last fold).")
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
st.plotly_chart(fig, width="stretch")

st.subheader("Trade Count per Fold")
fig2 = go.Figure(go.Bar(x=fold_labels, y=n_trades))
fig2.update_layout(xaxis_title="Fold", yaxis_title="# Trades")
st.plotly_chart(fig2, width="stretch")
