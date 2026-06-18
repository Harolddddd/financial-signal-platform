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
- **Model Leaderboard** — all trained models ranked by composite grade
- **Backtest Results** — equity curve, trade log, and financial metrics per model
- **Live Signals** — today's Buy/Hold/Sell recommendations with SHAP explanations
""")
