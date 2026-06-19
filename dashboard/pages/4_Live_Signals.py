# dashboard/pages/4_Live_Signals.py
import streamlit as st
from dashboard.config import PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, CONFIDENCE_THRESHOLD
from dashboard.data_loader import get_live_signals

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

with st.spinner("Generating signals..."):
    signals = get_live_signals(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, threshold)

if not signals:
    st.info("No Buy signals above the current confidence threshold.")
    st.stop()

st.success(f"Found **{len(signals)}** Buy signal(s)")

for sig in signals:
    with st.expander(f"**{sig.ticker}** — Confidence {sig.confidence:.1%} | Entry ${sig.entry_price:.2f}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", sig.signal.value)
        c2.metric("Confidence", f"{sig.confidence:.1%}")
        c3.metric("Position Size", f"{sig.position_size:.1%}")
        c4.metric("Entry Price", f"${sig.entry_price:.2f}")
