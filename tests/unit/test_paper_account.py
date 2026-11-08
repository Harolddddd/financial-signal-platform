from datetime import date, timedelta

import pytest

from src.paper import account as pa

US = pa.PaperConfig(cost_bps=10.0, lot_size=None)
CN = pa.PaperConfig(cost_bps=15.0, lot_size=100)
D = [date(2026, 11, 9) + timedelta(days=i) for i in range(10)]


def bar(o, c=None):
    return {"open": o, "close": c if c is not None else o}


def rate(*pairs):
    return [{"ticker": t, "overall_rating": r} for t, r in pairs]


def test_new_account():
    a = pa.new_account("us", "USD", 1000.0)
    assert a["cash"] == 1000.0 and a["positions"] == {} and a["pending"] == []


def test_signal_creates_pending_then_fills_next_open_with_costs():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {"AAA": bar(50, 50)}, rate(("AAA", 12.0)), US)
    assert a["positions"] == {} and [p["ticker"] for p in a["pending"]] == ["AAA"]
    a = pa.run_day(a, D[1], {"AAA": bar(40, 44)}, rate(("AAA", 12.0)), US)
    pos = a["positions"]["AAA"]
    # one of 5 slots: equity 1000 / 5 = 200 incl. 10 bps fee
    assert pos["shares"] == pytest.approx(200 / (40 * 1.001), rel=1e-6)
    assert a["cash"] == pytest.approx(800.0, abs=1e-6)
    assert a["equity"][-1]["total"] == pytest.approx(800 + pos["shares"] * 44)


def test_top5_only_and_rating_threshold():
    a = pa.new_account("us", "USD", 1000.0)
    ratings = rate(*[(f"T{i}", 20.0 - i) for i in range(8)], ("LOW", 7.9))
    a = pa.run_day(a, D[0], {}, ratings, US)
    assert [p["ticker"] for p in a["pending"]] == ["T0", "T1", "T2", "T3", "T4"]


def test_sell_after_5_trading_days():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    for i in range(1, 6):
        a = pa.run_day(a, D[i], {"AAA": bar(10)}, rate(("AAA", 12.0)), US)
    assert a["positions"]["AAA"]["held_days"] == 5
    assert [(p["ticker"], p["side"]) for p in a["pending"]] == [("AAA", "sell")]
    a = pa.run_day(a, D[6], {"AAA": bar(11)}, rate(("AAA", 12.0)), US)
    assert "AAA" not in a["positions"]
    t = a["trades"][-1]
    assert t["side"] == "sell" and t["pnl"] > 0


def test_sell_when_rating_drops_to_avoid():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    a = pa.run_day(a, D[1], {"AAA": bar(10)}, rate(("AAA", 3.0)), US)
    assert [(p["ticker"], p["side"]) for p in a["pending"]] == [("AAA", "sell")]


def test_china_lots_of_100_and_unaffordable_skipped():
    a = pa.new_account("china", "CNY", 1000.0)
    a = pa.run_day(a, D[0], {"CHEAP": bar(8), "PRICEY": bar(30)},
                   rate(("PRICEY", 15.0), ("CHEAP", 12.0)), CN)
    assert [p["ticker"] for p in a["pending"]] == ["CHEAP"]   # one lot of PRICEY = 3000 > cash
    a = pa.run_day(a, D[1], {"CHEAP": bar(8.5)}, rate(("CHEAP", 12.0)), CN)
    assert a["positions"]["CHEAP"]["shares"] == 100           # a full lot, above the 200 slot
    assert a["cash"] == pytest.approx(1000 - 850 * 1.0015)


def test_no_bar_keeps_order_pending():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    a = pa.run_day(a, D[1], {}, rate(("AAA", 12.0)), US)      # halted / no bar
    assert a["positions"] == {} and [p["ticker"] for p in a["pending"]] == ["AAA"]


def test_same_day_rerun_is_idempotent():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    b = pa.run_day(a, D[0], {}, rate(("BBB", 12.0)), US)
    assert [p["ticker"] for p in b["pending"]] == ["AAA"] and len(b["equity"]) == 1


def test_no_signals_holds_cash():
    a = pa.run_day(pa.new_account("us", "USD", 1000.0), D[0], {}, [], US)
    assert a["pending"] == [] and a["equity"][-1]["total"] == 1000.0
