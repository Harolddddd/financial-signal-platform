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
            "trade_status": "open",
            "trade_open_date": s.date,
            "trade_close_date": None,
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
cols = ["ticker", "strategy", "confidence", "entry_price", "date", "trade_open_date"]
df = pd.DataFrame(raw_signals)
if "trade_open_date" not in df.columns:
    df["trade_open_date"] = df["date"]
df = df[cols]
df = df.sort_values(["ticker", "confidence"], ascending=[True, False])
df["confidence"] = df["confidence"].map(lambda x: f"{x:.1%}")
df["entry_price"] = df["entry_price"].map(lambda x: format_price(x, market))
df.columns = ["Ticker", "Strategy", "Confidence", "Entry Price", "Date", "Trade Opened"]

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

st.divider()
st.subheader("Recently closed trades")
st.caption("Positions that were open (Buy) and have since flipped to Hold/Sell.")

if _raw:
    closed = [s for s in _raw["signals"] if s.get("trade_status") == "closed"]
    if closed:
        cdf = pd.DataFrame(closed)[
            ["ticker", "strategy", "signal", "trade_open_date", "trade_close_date", "entry_price"]
        ]
        cdf = cdf.sort_values("trade_close_date", ascending=False).head(200)
        cdf["entry_price"] = cdf["entry_price"].map(lambda x: format_price(x, market))
        cdf.columns = ["Ticker", "Strategy", "Current Signal", "Trade Opened", "Trade Closed", "Entry Price"]
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades found in the current cache.")
else:
    st.info("Closed-trade history requires the precomputed cache — run precompute_dashboard.py.")
