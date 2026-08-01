# dashboard/pages/4_Live_Signals.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config.markets import get_market
from dashboard.config import CONFIDENCE_THRESHOLD, PARQUET_DIR, OHLCV_COLS, FEATURE_COLS
from dashboard.data_loader import get_live_signals

CACHE_DIR = get_market("us").data_root / "cache"

st.set_page_config(page_title="Live Signals", layout="wide")
st.header("Live Buy Signals")

threshold = st.slider(
    "Confidence threshold", min_value=0.5, max_value=1.0,
    value=CONFIDENCE_THRESHOLD, step=0.05
)

# Load raw cache so we can surface the strategy field.
_raw = None
_cache_path = CACHE_DIR / "signals.json"
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
        live = get_live_signals(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, threshold)
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
df["entry_price"] = df["entry_price"].map(lambda x: f"${x:.2f}")
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
