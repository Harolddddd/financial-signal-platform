import streamlit as st
import plotly.graph_objects as go
import polars as pl
from config.markets import get_market
from dashboard.market_state import get_selected_market
from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import STALE_DAYS, get_leaderboard, get_league_pass, get_league_table

st.set_page_config(page_title="Strategy Leaderboard", layout="wide")
st.header("Strategy Leaderboard")

market = get_selected_market()
st.caption(f"Market: {get_market(market).label}")
parquet_dir, _ = get_paths(market)

HELP = {
    "Net Sharpe": "Main league score. Every Buy signal is a 5-day trade; its return minus a round trip of "
                  "trading costs, averaged and divided by its spread, scaled to a year. "
                  "Above 0.5 is decent, above 1 is strong, below 0 loses money after costs.",
    "Precision": "Share of those trades that made money after costs.",
    "Trades": "Number of Buy signals scored in the 3-year test window. Few trades = less reliable score.",
    "Ran on": "Server that computed this test: A = production, B = compute server (A re-scores B's results).",
    "Runtime (min)": "Wall-clock time of the last bench test for this method.",
    "Trend": "Net Sharpe versus the previous test: up, down or flat (within 0.05).",
}
_CFG = {k: st.column_config.Column(help=v) for k, v in HELP.items()}

with st.expander("What the numbers mean"):
    st.markdown("\n".join([
        "- **Net Sharpe** (league score): " + HELP["Net Sharpe"],
        "- **Precision**: " + HELP["Precision"],
        "- **Trades**: " + HELP["Trades"],
        "- **Score** (full-history table, 0-1): 40% precision of Buy calls + 30% Sharpe (squashed to 0-1) "
        "+ 30% drawdown safety (0% drawdown = full marks, 50%+ = none). Grades: A >= 0.65, B >= 0.50, "
        "C >= 0.35, else D.",
        "- **Ran on**: " + HELP["Ran on"],
    ]))


def _league_rows(rows: list[dict], retired: bool = False) -> list[dict]:
    out = []
    for r in rows:
        row = {
            "Method": r["method"],
            "Net Sharpe": "-" if r["net_sharpe"] is None else f"{r['net_sharpe']:.2f}",
            "Trend": r["trend"],
            "Precision": "-" if r["precision"] is None else f"{r['precision']:.1%}",
            "Trades": r["n_trades"] if r["n_trades"] is not None else "-",
            "Last tested": r["tested_on"] or "never",
            "Ran on": r.get("ran_on") or "-",
            "Runtime (min)": "-" if r.get("runtime_min") is None else f"{r['runtime_min']:.1f}",
        }
        if retired:
            row["Why retired"] = r["retired_because"] or ""
        else:
            row["Status"] = ("stale" if r["stale"] else "ok") + (" · error" if r.get("error") else "")
            row["Duplicate of"] = r["duplicate_of"] or ""
        out.append(row)
    return out


league = get_league_table(market)
lp = get_league_pass(market)
if lp is None:
    st.caption("No bench pass has run yet.")
elif lp["completed"]:
    st.success(f"Last bench pass {lp['date']}: finished, all methods tested.")
else:
    st.warning(f"Last bench pass {lp['date']} hit its deadline: {len(lp['untested'])} methods not tested "
               f"({', '.join(lp['untested'])}). They are tested first in the next pass.")
st.subheader("Live methods")
st.caption(
    "Retrained weekly; daily signals come from these. Score = Sharpe after trading costs "
    f"on the last 3 years (bench re-test every 2 weeks). 'stale' = not re-tested for "
    f"{STALE_DAYS['live']}+ days."
)
if league["live"]:
    st.dataframe(_league_rows(league["live"]), use_container_width=True, hide_index=True, column_config=_CFG)
else:
    st.info("No bench re-test has run yet for this market; the live set is the configured seed.")
with st.expander(f"Bench ({len(league['bench'])} methods) - promoted after beating the weakest live method twice in a row"):
    st.dataframe(_league_rows(league["bench"]), use_container_width=True, hide_index=True, column_config=_CFG)
if league["retired"]:
    with st.expander(f"Retired ({len(league['retired'])}) - near-identical to a better method for over a month"):
        st.dataframe(_league_rows(league["retired"], retired=True), use_container_width=True, hide_index=True, column_config=_CFG)

st.divider()
st.subheader("Full-history backtests")
st.caption("Walk-forward over the whole history; refreshed weekly for live methods only.")


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
st.dataframe(df.to_pandas(), use_container_width=True, hide_index=True, column_config={
    "Score": st.column_config.Column(help="0-1 blend: 40% Buy precision, 30% Sharpe, 30% drawdown safety."),
    "Precision Buy": st.column_config.Column(help="Of the days the model said Buy, share where the price went up."),
    "Sharpe": st.column_config.Column(help="Risk-adjusted return of following the signals, yearly scale, before costs."),
    "Max Drawdown": st.column_config.Column(help="Worst peak-to-trough loss over the history."),
})

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
