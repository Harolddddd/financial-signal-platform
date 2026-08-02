# dashboard/market_state.py
from __future__ import annotations

import streamlit as st

from config.markets import MARKETS, get_market

_SESSION_KEY = "market"
_DEFAULT_MARKET = "us"

_CURRENCY_SYMBOLS = {"USD": "$", "CNY": "¥"}


def render_market_selector() -> str:
    """Render the market dropdown. Call this once, from app.py only —
    every other page reads the resulting choice via get_selected_market()."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = _DEFAULT_MARKET

    keys = list(MARKETS.keys())
    st.selectbox(
        "Market",
        keys,
        format_func=lambda key: MARKETS[key].label,
        key=_SESSION_KEY,
        # Without this, Streamlit drops a keyed widget's session_state value
        # the moment the widget stops being rendered — which is every page
        # other than this one, since the selector only lives on app.py. See
        # streamlit's own session-state docs: "By default, widgets are NOT
        # stateful across pages ... set persist_state='session'" to survive
        # page switches.
        persist_state="session",
    )
    return st.session_state[_SESSION_KEY]


def get_selected_market() -> str:
    """Read the market chosen on app.py. Defaults to "us" if no selection
    has been made yet in this session (e.g. a page reached directly)."""
    return st.session_state.get(_SESSION_KEY, _DEFAULT_MARKET)


def format_price(value: float, market: str) -> str:
    symbol = _CURRENCY_SYMBOLS.get(get_market(market).currency, "$")
    return f"{symbol}{value:.2f}"
