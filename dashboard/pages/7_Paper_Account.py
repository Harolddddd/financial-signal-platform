import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.markets import get_market
from dashboard.data_loader import get_cache_dir, get_ticker_names
from dashboard.market_state import format_price, get_selected_market

st.set_page_config(page_title="Paper Account", layout="wide")
st.header("Paper Account")

market = get_selected_market()
cfg = get_market(market)
st.caption(
    f"Market: {cfg.label}. Simulated money, no real orders. Each trading day after the close it "
    "buys the top-rated stocks on the Combined Signal page (rating 8+, up to 5 positions, equal "
    "size) at the next open, and sells after 5 trading days or when the rating drops below 4. "
    f"Costs: {cfg.cost_bps:.0f} bps per side"
    + (" (incl. stamp duty); China trades in lots of 100 shares." if market == "china"
       else "; fractional shares allowed.")
)

path = get_cache_dir(market) / "paper_account.json"
if not path.exists():
    st.info("The account starts with the next daily run.")
    st.stop()
a = json.loads(path.read_text(encoding="utf-8"))
names = get_ticker_names(market)
money = lambda v: format_price(v, market)

eq = a["equity"]
total = eq[-1]["total"] if eq else a["cash"]
ret = total / a["start_cash"] - 1
bench_ret = None
bvals = [e.get("benchmark") for e in eq if e.get("benchmark")]
if len(bvals) >= 1:
    bench_ret = bvals[-1] / bvals[0] - 1

c1, c2, c3, c4 = st.columns(4)
c1.metric("Account value", money(total), f"{ret:+.2%}")
c2.metric("Cash", money(a["cash"]))
c3.metric("Positions", len(a["positions"]))
c4.metric(f"Benchmark ({cfg.benchmark_ticker})", f"{bench_ret:+.2%}" if bench_ret is not None else "-")
st.caption(f"Started with {money(a['start_cash'])}; last updated for {a.get('last_day') or '-'}.")

if len(eq) >= 2:
    df = pd.DataFrame(eq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["total"], name="Account", mode="lines+markers"))
    if df["benchmark"].notna().any():
        b0 = df["benchmark"].dropna().iloc[0]
        fig.add_trace(go.Scatter(x=df["date"], y=df["benchmark"] / b0 * a["start_cash"],
                                 name=f"{cfg.benchmark_ticker} (same start)", mode="lines"))
    fig.update_layout(title="Account value", yaxis_title=a["currency"], height=360)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Holdings")
if a["positions"]:
    rows = []
    for t, p in a["positions"].items():
        value = p["shares"] * p["last_price"]
        rows.append({
            "Ticker": t, "Name": names.get(t, ""),
            "Shares": round(p["shares"], 4), "Bought": p["entry_date"],
            "Entry": money(p["entry_price"]), "Last": money(p["last_price"]),
            "Value": money(value), "P&L": f"{value / p['cost_basis'] - 1:+.2%}",
            "Days held": p["held_days"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.caption("No holdings (all cash).")

st.subheader("Orders for the next open")
if a["pending"]:
    st.dataframe(pd.DataFrame([{
        "Ticker": o["ticker"], "Name": names.get(o["ticker"], ""), "Side": o["side"],
        "Signal date": o["signal_date"], "Why": o.get("reason", ""),
    } for o in a["pending"]]), hide_index=True, use_container_width=True)
else:
    st.caption("No orders queued (no stock rated 8+ today, or all slots full).")

st.subheader("Trade history")
done = [t for t in a["trades"] if t["side"] in ("buy", "sell")]
if done:
    closed = [t for t in done if t["side"] == "sell"]
    wins = sum(1 for t in closed if t["pnl"] > 0)
    if closed:
        st.caption(f"{len(closed)} closed trades, {wins} winners ({wins / len(closed):.0%}), "
                   f"realized P&L {money(sum(t['pnl'] for t in closed))}.")
    st.dataframe(pd.DataFrame([{
        "Date": t["date"], "Ticker": t["ticker"], "Name": names.get(t["ticker"], ""),
        "Side": t["side"], "Shares": round(t["shares"], 4), "Price": money(t["price"]),
        "Value": money(t["value"]), "P&L": money(t["pnl"]) if t["side"] == "sell" else "",
        "Why": t.get("reason", ""),
    } for t in reversed(done)]), hide_index=True, use_container_width=True)
else:
    st.caption("No trades yet.")
