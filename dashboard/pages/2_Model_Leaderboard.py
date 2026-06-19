# dashboard/pages/2_Model_Leaderboard.py
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from dashboard.config import PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_leaderboard

st.set_page_config(page_title="Strategy Leaderboard", layout="wide")
st.header("Strategy Leaderboard")


@st.cache_data(ttl=1800)
def _leaderboard():
    return get_leaderboard(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS)


with st.spinner("Computing grades..."):
    leaderboard = _leaderboard()

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
