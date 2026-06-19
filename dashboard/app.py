import streamlit as st

st.set_page_config(
    page_title="Financial Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Signal Platform")
st.markdown("""
Navigate using the sidebar:

- **Data Overview** — data ingestion status, ticker universe, date ranges
- **Strategy Leaderboard** — all strategies ranked by composite grade (walk-forward backtest)
- **Backtest Results** — fold-by-fold performance metrics per strategy
- **Live Signals** — today's Buy/Hold/Sell recommendations with confidence scores
""")
