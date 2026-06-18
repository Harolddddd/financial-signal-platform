import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, REGISTRY_DIR, FEATURE_COLS, GRADE_COLORS
from dashboard.data_loader import get_backtest_result
from src.models.registry import list_models

st.set_page_config(page_title="Backtest Results", layout="wide")
st.header("Backtest Results")

records = list_models(REGISTRY_DIR)
if not records:
    st.warning("No models in registry.")
    st.stop()

model_names = [r.model_name for r in records]
selected = st.selectbox("Select model", model_names)


@st.cache_data(ttl=1800)
def _backtest(model_name: str):
    return get_backtest_result(model_name, REGISTRY_DIR, PARQUET_DIR, FEATURE_COLS)


with st.spinner(f"Running walk-forward backtest for {selected}..."):
    wf_result, grade = _backtest(selected)

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
