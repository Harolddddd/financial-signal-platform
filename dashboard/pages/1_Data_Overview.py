import streamlit as st
import plotly.graph_objects as go
from dashboard.config import PARQUET_DIR, FEATURE_COLS
from dashboard.data_loader import get_data_summary
from src.features.duckdb_client import load_training_data

st.set_page_config(page_title="Data Overview", layout="wide")
st.header("Data Overview")


@st.cache_data(ttl=3600)
def _summary():
    return get_data_summary(PARQUET_DIR)


@st.cache_data(ttl=3600)
def _load_ticker_df(ticker: str):
    return load_training_data(PARQUET_DIR, tickers=[ticker])


summary = _summary()

col1, col2, col3 = st.columns(3)
col1.metric("Tickers", summary["n_tickers"])
col2.metric("Total Rows", f"{summary['n_rows']:,}")
col3.metric("Date Range", f"{summary['date_range_start'][:10]} → {summary['date_range_end'][:10]}")

st.divider()

ticker = st.selectbox("Select ticker to preview", summary["tickers"])
if ticker:
    df = _load_ticker_df(ticker)
    if not df.is_empty():
        fig = go.Figure()
        times = df["time"].to_list()
        closes = df["close"].to_list()
        fig.add_trace(go.Scatter(x=times, y=closes, mode="lines", name="Close"))
        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=times, y=df["sma_20"].to_list(),
                                     mode="lines", name="SMA 20", line=dict(dash="dot")))
        fig.update_layout(title=f"{ticker} Price + SMA 20",
                          xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        if "sent_pos_avg_5d" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=times, y=df["sent_pos_avg_5d"].to_list(),
                                  name="5d Avg Positive Sentiment"))
            fig2.update_layout(title=f"{ticker} News Sentiment (5d Rolling)",
                                xaxis_title="Date", yaxis_title="Positive Sentiment Score")
            st.plotly_chart(fig2, use_container_width=True)
