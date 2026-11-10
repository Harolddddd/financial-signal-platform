"""Paper-trading account driven by the Combined Signal ratings and a sell model.

Once per trading day, after the close (`run_day`):
1. fill yesterday's orders at today's open (sells first, then buys), with costs;
2. mark positions to today's close and record equity;
3. queue orders for tomorrow's open.

Sell (first rule that fires):
- sell model: the stock is in today's highest-risk group (`high_risk`);
- stop-loss: close <= entry - stop_atr * ATR at entry (no ATR: -fallback_stop);
- trailing stop: once up >= trail_after_atr * ATR, close <= peak - trail_atr * ATR;
- max hold: held >= max_hold_days;
- signal gone: held >= min_hold_days and no longer rated Buy.
Buy: rated >= buy_rating, not high risk, not in a post-stop cooldown, best first,
up to max_positions. Size = equity / max_positions * clamp(rating / 12, 0.5, 1.5),
capped at max_position_frac of equity. With lot_size (China: 100) at least one
whole lot is bought if cash allows; otherwise fractional shares.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date

_RATING_FOR_FULL_SLOT = 12.0  # "Strong Buy" on the Combined Signal page


@dataclass(frozen=True)
class PaperConfig:
    cost_bps: float
    lot_size: int | None
    max_positions: int = 5
    buy_rating: float = 8.0
    min_size_mult: float = 0.5
    max_size_mult: float = 1.5
    max_position_frac: float = 0.30
    # China trades in lots of 100: one lot may exceed its slot (else small accounts
    # could buy almost nothing), but never this share of equity.
    max_lot_frac: float = 0.40
    stop_atr: float = 2.0
    fallback_stop: float = 0.08
    trail_after_atr: float = 1.0
    trail_atr: float = 2.0
    min_hold_days: int = 5
    max_hold_days: int = 20
    cooldown_days: int = 5


def new_account(market: str, currency: str, start_cash: float) -> dict:
    return {
        "market": market, "currency": currency, "start_cash": start_cash, "cash": start_cash,
        "positions": {}, "pending": [], "trades": [], "equity": [], "cooldown": {}, "last_day": None,
    }


def _fee(cfg: PaperConfig) -> float:
    return cfg.cost_bps / 1e4


def _trade(a, day, t, side, shares, price, value, pnl, reason):
    a["trades"].append({"date": day, "ticker": t, "side": side, "shares": shares, "price": price,
                        "value": value, "pnl": pnl, "reason": reason})


def _fill_sells(a: dict, day: str, bars: dict, cfg: PaperConfig) -> None:
    keep = []
    for o in a["pending"]:
        b = bars.get(o["ticker"])
        if o["side"] != "sell" or b is None or o["ticker"] not in a["positions"]:
            keep.append(o)
            continue
        pos = a["positions"].pop(o["ticker"])
        proceeds = pos["shares"] * b["open"] * (1 - _fee(cfg))
        a["cash"] += proceeds
        _trade(a, day, o["ticker"], "sell", pos["shares"], b["open"], proceeds,
               proceeds - pos["cost_basis"], o.get("reason", ""))
        if o.get("reason", "").startswith("stop-loss"):
            a["cooldown"][o["ticker"]] = len(a["equity"]) + cfg.cooldown_days
    a["pending"] = keep


def _fill_buys(a: dict, day: str, bars: dict, cfg: PaperConfig) -> None:
    equity = a["cash"] + sum(
        p["shares"] * bars.get(t, {}).get("open", p["last_price"]) for t, p in a["positions"].items()
    )
    keep = []
    for o in a["pending"]:
        b = bars.get(o["ticker"])
        if o["side"] != "buy" or b is None:
            keep.append(o)
            continue
        mult = min(max(o.get("rating", _RATING_FOR_FULL_SLOT) / _RATING_FOR_FULL_SLOT,
                       cfg.min_size_mult), cfg.max_size_mult)
        target = min(equity / cfg.max_positions * mult, equity * cfg.max_position_frac)
        unit = b["open"] * (1 + _fee(cfg))
        why = "not enough cash"
        if cfg.lot_size:
            lot_cost = cfg.lot_size * unit
            budget = min(max(target, lot_cost), a["cash"])
            shares = math.floor(budget / lot_cost + 1e-9) * cfg.lot_size
            if shares == cfg.lot_size and lot_cost > target and lot_cost > equity * cfg.max_lot_frac:
                shares, why = 0, f"one lot over {cfg.max_lot_frac:.0%} of equity"
        else:
            budget = min(target, a["cash"])
            shares = budget / unit if budget >= 1.0 else 0.0
        if shares <= 0:
            _trade(a, day, o["ticker"], "skip", 0, b["open"], 0.0, 0.0, why)
            continue
        cost = shares * unit
        a["cash"] -= cost
        a["positions"][o["ticker"]] = {
            "shares": shares, "entry_date": day, "entry_price": b["open"], "cost_basis": cost,
            "held_days": 0, "last_price": b["open"], "peak": b["open"], "entry_atr": b.get("atr"),
        }
        _trade(a, day, o["ticker"], "buy", shares, b["open"], cost, 0.0, o.get("reason", ""))
    a["pending"] = keep


def _mark(a: dict, day: str, bars: dict) -> None:
    for t, p in a["positions"].items():
        b = bars.get(t)
        if b is not None:
            p["last_price"] = b["close"]
            p["peak"] = max(p.get("peak", b["close"]), b["close"])
            p["atr"] = b.get("atr")
            p["held_days"] += 1
    holdings = sum(p["shares"] * p["last_price"] for p in a["positions"].values())
    a["equity"].append({"date": day, "cash": a["cash"], "holdings": holdings, "total": a["cash"] + holdings})


def _exit_reason(t: str, p: dict, rating: float | None, high_risk: set, cfg: PaperConfig) -> str | None:
    if t in high_risk:
        return "sell model: high downside risk"
    entry, close = p["entry_price"], p["last_price"]
    atr0 = p.get("entry_atr")
    stop = entry - cfg.stop_atr * atr0 if atr0 else entry * (1 - cfg.fallback_stop)
    if close <= stop:
        return f"stop-loss ({close / entry - 1:+.1%})"
    atr = p.get("atr") or atr0
    if atr and p["peak"] >= entry + cfg.trail_after_atr * atr and close <= p["peak"] - cfg.trail_atr * atr:
        return f"trailing stop ({close / p['peak'] - 1:+.1%} from peak)"
    if p["held_days"] >= cfg.max_hold_days:
        return f"max hold {cfg.max_hold_days} days"
    if p["held_days"] >= cfg.min_hold_days and (rating is None or rating < cfg.buy_rating):
        return f"held {p['held_days']} days, no longer rated Buy"
    return None


def _queue(a: dict, day: str, bars: dict, ratings: list[dict], high_risk: set, cfg: PaperConfig) -> None:
    rating = {r["ticker"]: r for r in ratings}
    pending_sell = {o["ticker"] for o in a["pending"] if o["side"] == "sell"}
    pending_buy = {o["ticker"] for o in a["pending"] if o["side"] == "buy"}
    for t, p in a["positions"].items():
        if t in pending_sell:
            continue
        reason = _exit_reason(t, p, rating.get(t, {}).get("overall_rating"), high_risk, cfg)
        if reason:
            a["pending"].append({"ticker": t, "side": "sell", "signal_date": day, "reason": reason})
            pending_sell.add(t)

    slots = cfg.max_positions - (len(a["positions"]) - len(pending_sell)) - len(pending_buy)
    est_cash = a["cash"] + sum(
        a["positions"][t]["shares"] * a["positions"][t]["last_price"] * (1 - _fee(cfg)) for t in pending_sell
    )
    n_days = len(a["equity"])
    equity = a["cash"] + sum(p["shares"] * p["last_price"] for p in a["positions"].values())
    cands = sorted((r for r in ratings if r["overall_rating"] >= cfg.buy_rating),
                   key=lambda r: r["overall_rating"], reverse=True)
    for r in cands:
        if slots <= 0:
            break
        t = r["ticker"]
        if t in a["positions"] or t in pending_buy or t in high_risk:
            continue
        if n_days < a.get("cooldown", {}).get(t, 0):
            continue
        if cfg.lot_size:
            price = bars.get(t, {}).get("close") or r.get("entry_price")
            if not price:
                continue
            lot_cost = cfg.lot_size * price * (1 + _fee(cfg))
            if lot_cost > est_cash or lot_cost > equity * cfg.max_lot_frac:
                continue
            est_cash -= lot_cost
        a["pending"].append({"ticker": t, "side": "buy", "signal_date": day, "rating": r["overall_rating"],
                             "reason": f"rating {r['overall_rating']:.1f}"})
        pending_buy.add(t)
        slots -= 1


def run_day(acct: dict, today: date, bars: dict, ratings: list[dict], cfg: PaperConfig,
            high_risk: set | frozenset = frozenset()) -> dict:
    """bars: ticker -> {"open", "close", "atr"} for `today`; ratings: today's Combined Signal
    rows; high_risk: tickers the sell model puts in today's riskiest group."""
    day = today.isoformat()
    if acct["last_day"] is not None and day <= acct["last_day"]:
        return acct
    a = copy.deepcopy(acct)
    a.setdefault("cooldown", {})
    _fill_sells(a, day, bars, cfg)
    _fill_buys(a, day, bars, cfg)
    _mark(a, day, bars)
    _queue(a, day, bars, ratings, set(high_risk), cfg)
    a["last_day"] = day
    return a
