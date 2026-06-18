import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, REGISTRY_DIR, FEATURE_COLS, CONFIDENCE_THRESHOLD
from dashboard.data_loader import get_live_signals

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

with st.spinner("Generating signals..."):
    signals = get_live_signals(REGISTRY_DIR, PARQUET_DIR, FEATURE_COLS, threshold)

if not signals:
    st.info("No Buy signals above the current confidence threshold.")
    st.stop()

st.success(f"Found **{len(signals)}** Buy signal(s)")

for sig in signals:
    with st.expander(f"**{sig.ticker}** — Confidence {sig.confidence:.1%} | Entry ${sig.entry_price:.2f}"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{sig.confidence:.1%}")
        c2.metric("Position Size", f"{sig.position_size:.1%}")
        c3.metric("Entry Price", f"${sig.entry_price:.2f}")

        if sig.feature_explanation:
            top_n = dict(list(sig.feature_explanation.items())[:10])
            features = list(top_n.keys())
            values   = list(top_n.values())
            colors   = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
            fig = go.Figure(go.Bar(
                x=values[::-1], y=features[::-1],
                orientation="h",
                marker_color=colors[::-1],
            ))
            fig.update_layout(
                title=f"Top 10 Features Driving {sig.ticker} Buy Signal",
                xaxis_title="SHAP Value (positive = bullish)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Feature explanation not available.")
