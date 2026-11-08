"""Paper-trading account driven by the Combined Signal ratings.

Once per trading day, after the close (`run_day`):
1. fill yesterday's orders at today's open (sells first, then buys), with costs;
2. mark positions to today's close and record equity;
3. queue orders for tomorrow's open from today's ratings: sell after `hold_days`
   trading days or when the rating falls below `sell_rating`; buy the highest
   rated (>= `buy_rating`) stocks not held, up to `max_positions`.
Buys are equal slots of equity / max_positions. With `lot_size` (China: 100
shares) at least one whole lot is bought if cash allows; otherwise fractional.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PaperConfig:
    cost_bps: float
    lot_size: int | None
    max_positions: int = 5
    hold_days: int = 5
    buy_rating: float = 8.0   # "Buy" or better on the Combined Signal page
    sell_rating: float = 4.0  # below this is "Avoid"


def new_account(market: str, currency: str, start_cash: float) -> dict:
    return {
        "market": market, "currency": currency, "start_cash": start_cash, "cash": start_cash,
        "positions": {}, "pending": [], "trades": [], "equity": [], "last_day": None,
    }


def _fee(cfg: PaperConfig) -> float:
    return cfg.cost_bps / 1e4


def _fill_sells(a: dict, today: str, bars: dict, cfg: PaperConfig) -> None:
    keep = []
    for o in a["pending"]:
        b = bars.get(o["ticker"])
        if o["side"] != "sell" or b is None or o["ticker"] not in a["positions"]:
            keep.append(o)
            continue
        pos = a["positions"].pop(o["ticker"])
        proceeds = pos["shares"] * b["open"] * (1 - _fee(cfg))
        a["cash"] += proceeds
        a["trades"].append({
            "date": today, "ticker": o["ticker"], "side": "sell", "shares": pos["shares"],
            "price": b["open"], "value": proceeds, "pnl": proceeds - pos["cost_basis"],
            "reason": o.get("reason", ""),
        })
    a["pending"] = keep


def _fill_buys(a: dict, today: str, bars: dict, cfg: PaperConfig) -> None:
    equity = a["cash"] + sum(
        p["shares"] * bars.get(t, {}).get("open", p["last_price"]) for t, p in a["positions"].items()
    )
    slot = equity / cfg.max_positions
    keep = []
    for o in a["pending"]:
        b = bars.get(o["ticker"])
        if o["side"] != "buy" or b is None:
            keep.append(o)
            continue
        unit = b["open"] * (1 + _fee(cfg))
        if cfg.lot_size:
            lot_cost = cfg.lot_size * unit
            budget = min(max(slot, lot_cost), a["cash"])
            shares = math.floor(budget / lot_cost) * cfg.lot_size
        else:
            budget = min(slot, a["cash"])
            shares = budget / unit if budget >= 1.0 else 0.0
        if shares <= 0:
            a["trades"].append({"date": today, "ticker": o["ticker"], "side": "skip", "shares": 0,
                                "price": b["open"], "value": 0.0, "pnl": 0.0, "reason": "not enough cash"})
            continue
        cost = shares * unit
        a["cash"] -= cost
        a["positions"][o["ticker"]] = {
            "shares": shares, "entry_date": today, "entry_price": b["open"],
            "cost_basis": cost, "held_days": 0, "last_price": b["open"],
        }
        a["trades"].append({"date": today, "ticker": o["ticker"], "side": "buy", "shares": shares,
                            "price": b["open"], "value": cost, "pnl": 0.0, "reason": o.get("reason", "")})
    a["pending"] = keep


def _mark(a: dict, today: str, bars: dict) -> None:
    for t, p in a["positions"].items():
        b = bars.get(t)
        if b is not None:
            p["last_price"] = b["close"]
            p["held_days"] += 1
    holdings = sum(p["shares"] * p["last_price"] for p in a["positions"].values())
    a["equity"].append({"date": today, "cash": a["cash"], "holdings": holdings, "total": a["cash"] + holdings})


def _queue(a: dict, today: str, bars: dict, ratings: list[dict], cfg: PaperConfig) -> None:
    rating = {r["ticker"]: r for r in ratings}
    pending_sell = {o["ticker"] for o in a["pending"] if o["side"] == "sell"}
    pending_buy = {o["ticker"] for o in a["pending"] if o["side"] == "buy"}
    for t, p in a["positions"].items():
        if t in pending_sell:
            continue
        r = rating.get(t, {}).get("overall_rating")
        reason = None
        if p["held_days"] >= cfg.hold_days:
            reason = f"held {p['held_days']} days"
        elif r is not None and r < cfg.sell_rating:
            reason = f"rating {r:.1f} (Avoid)"
        if reason:
            a["pending"].append({"ticker": t, "side": "sell", "signal_date": today, "reason": reason})
            pending_sell.add(t)

    slots = cfg.max_positions - (len(a["positions"]) - len(pending_sell)) - len(pending_buy)
    est_cash = a["cash"] + sum(
        a["positions"][t]["shares"] * a["positions"][t]["last_price"] * (1 - _fee(cfg)) for t in pending_sell
    )
    cands = sorted((r for r in ratings if r["overall_rating"] >= cfg.buy_rating),
                   key=lambda r: r["overall_rating"], reverse=True)
    for r in cands:
        if slots <= 0:
            break
        t = r["ticker"]
        if t in a["positions"] or t in pending_buy:
            continue
        if cfg.lot_size:
            price = bars.get(t, {}).get("close") or r.get("entry_price")
            if not price:
                continue
            lot_cost = cfg.lot_size * price * (1 + _fee(cfg))
            if lot_cost > est_cash:
                continue
            est_cash -= lot_cost
        a["pending"].append({"ticker": t, "side": "buy", "signal_date": today,
                             "reason": f"rating {r['overall_rating']:.1f}"})
        pending_buy.add(t)
        slots -= 1


def run_day(acct: dict, today: date, bars: dict, ratings: list[dict], cfg: PaperConfig) -> dict:
    """bars: ticker -> {"open", "close"} for `today`; ratings: today's Combined Signal rows."""
    day = today.isoformat()
    if acct["last_day"] is not None and day <= acct["last_day"]:
        return acct
    a = copy.deepcopy(acct)
    _fill_sells(a, day, bars, cfg)
    _fill_buys(a, day, bars, cfg)
    _mark(a, day, bars)
    _queue(a, day, bars, ratings, cfg)
    a["last_day"] = day
    return a
