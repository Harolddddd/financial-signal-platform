# dashboard/pages/5_Combined_Signal.py
import pandas as pd
import streamlit as st

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


def _label(rating: float) -> str:
    if rating >= 12:
        return "Strong Buy"
    if rating >= 8:
        return "Buy"
    if rating >= 4:
        return "Neutral"
    return "Avoid"


summary_rows, detail_by_ticker = get_combined_ratings()

if not summary_rows:
    st.warning(
        "No cached signals/leaderboard found. Run "
        "`python scripts/precompute_dashboard.py` first."
    )
    st.stop()

df = pd.DataFrame(summary_rows)
df["Rating"] = df["overall_rating"].round(1)
df["Signal"] = df["Rating"].map(_label)
df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df["Entry Price"] = df["entry_price"].map(lambda x: f"${x:.2f}")
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
    ddf["entry_price"] = ddf["entry_price"].map(lambda x: f"${x:.2f}")
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
