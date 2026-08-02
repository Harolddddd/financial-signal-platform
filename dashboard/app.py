import streamlit as st

from config.markets import get_market
from dashboard.market_state import render_market_selector

st.set_page_config(
    page_title="Financial Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Signal Platform")

market = render_market_selector()
st.caption(f"Active market: {get_market(market).label}")

st.markdown("""
Navigate using the sidebar:

- **Data Overview** — data ingestion status, ticker universe, date ranges
- **Strategy Leaderboard** — all strategies ranked by composite grade (walk-forward backtest)
- **Backtest Results** — fold-by-fold performance metrics per strategy
- **Live Signals** — today's Buy/Hold/Sell recommendations with confidence scores
""")
