from datetime import date, timedelta

import pytest

from src.paper import account as pa

US = pa.PaperConfig(cost_bps=10.0, lot_size=None)
CN = pa.PaperConfig(cost_bps=15.0, lot_size=100)
D = [date(2026, 11, 9) + timedelta(days=i) for i in range(30)]


def bar(o, c=None, atr=1.0):
    return {"open": o, "close": c if c is not None else o, "atr": atr}


def rate(*pairs):
    return [{"ticker": t, "overall_rating": r} for t, r in pairs]


def _buy(a, t="AAA", price=10.0, rating=12.0, cfg=US, atr=1.0):
    a = pa.run_day(a, D[0], {}, rate((t, rating)), cfg)
    return pa.run_day(a, D[1], {t: bar(price, price, atr)}, rate((t, rating)), cfg)


def test_new_account():
    a = pa.new_account("us", "USD", 2000.0)
    assert a["cash"] == 2000.0 and a["positions"] == {} and a["pending"] == []


def test_signal_creates_pending_then_fills_next_open_with_costs():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {"AAA": bar(50)}, rate(("AAA", 12.0)), US)
    assert a["positions"] == {} and [p["ticker"] for p in a["pending"]] == ["AAA"]
    a = pa.run_day(a, D[1], {"AAA": bar(40, 44)}, rate(("AAA", 12.0)), US)
    pos = a["positions"]["AAA"]
    # rating 12 -> conviction 1.0 -> one equal slot: 1000 / 5 = 200 incl. 10 bps fee
    assert pos["shares"] == pytest.approx(200 / (40 * 1.001), rel=1e-6)
    assert a["cash"] == pytest.approx(800.0, abs=1e-6)
    assert a["equity"][-1]["total"] == pytest.approx(800 + pos["shares"] * 44)


def test_size_scales_with_rating_within_bounds():
    for rating, mult in ((24.0, 1.5), (6.0 * 2, 1.0), (8.0, 8 / 12)):
        a = pa.new_account("us", "USD", 1000.0)
        a = pa.run_day(a, D[0], {}, rate(("AAA", rating)), US)
        a = pa.run_day(a, D[1], {"AAA": bar(10)}, rate(("AAA", rating)), US)
        assert a["cash"] == pytest.approx(1000 - 200 * mult, abs=1e-6)


def test_top5_only_and_rating_threshold():
    a = pa.new_account("us", "USD", 1000.0)
    ratings = rate(*[(f"T{i}", 20.0 - i) for i in range(8)], ("LOW", 7.9))
    a = pa.run_day(a, D[0], {}, ratings, US)
    assert [p["ticker"] for p in a["pending"]] == ["T0", "T1", "T2", "T3", "T4"]


def test_high_risk_blocks_buy():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 15.0), ("BBB", 12.0)), US, high_risk={"AAA"})
    assert [p["ticker"] for p in a["pending"]] == ["BBB"]


def test_sell_model_exit():
    a = _buy(pa.new_account("us", "USD", 1000.0))
    a = pa.run_day(a, D[2], {"AAA": bar(10)}, rate(("AAA", 12.0)), US, high_risk={"AAA"})
    assert [(p["ticker"], p["side"]) for p in a["pending"]] == [("AAA", "sell")]
    assert "sell model" in a["pending"][0]["reason"]


def test_stop_loss_two_atr_below_entry():
    a = _buy(pa.new_account("us", "USD", 1000.0), price=10.0, atr=0.5)
    a = pa.run_day(a, D[2], {"AAA": bar(9.5, 9.1)}, rate(("AAA", 12.0)), US)
    assert a["pending"] == []                       # 9.1 > 10 - 2*0.5
    a = pa.run_day(a, D[3], {"AAA": bar(9.1, 8.9)}, rate(("AAA", 12.0)), US)
    assert a["pending"][0]["side"] == "sell" and "stop-loss" in a["pending"][0]["reason"]


def test_stop_loss_fallback_without_atr():
    a = _buy(pa.new_account("us", "USD", 1000.0), price=10.0, atr=None)
    a = pa.run_day(a, D[2], {"AAA": bar(9.3, 9.15, None)}, rate(("AAA", 12.0)), US)
    assert "stop-loss" in a["pending"][0]["reason"]   # -8.5% < -8%


def test_trailing_stop_after_gain():
    a = _buy(pa.new_account("us", "USD", 1000.0), price=10.0, atr=0.5)
    a = pa.run_day(a, D[2], {"AAA": bar(11, 12.0, 0.5)}, rate(("AAA", 12.0)), US)   # peak 12
    assert a["pending"] == []
    a = pa.run_day(a, D[3], {"AAA": bar(11.5, 10.9, 0.5)}, rate(("AAA", 12.0)), US)  # < 12 - 1.0
    assert "trailing stop" in a["pending"][0]["reason"]


def test_hold_past_5_days_only_while_still_rated_buy():
    a = _buy(pa.new_account("us", "USD", 1000.0))
    for i in range(2, 7):
        a = pa.run_day(a, D[i], {"AAA": bar(10)}, rate(("AAA", 12.0)), US)
    assert a["positions"]["AAA"]["held_days"] == 6 and a["pending"] == []   # still Buy: keep
    a = pa.run_day(a, D[7], {"AAA": bar(10)}, rate(("AAA", 3.0)), US)
    assert "no longer rated Buy" in a["pending"][0]["reason"]


def test_rating_drop_before_day_5_does_not_sell():
    a = _buy(pa.new_account("us", "USD", 1000.0))
    a = pa.run_day(a, D[2], {"AAA": bar(10)}, rate(("AAA", 0.0)), US)
    assert a["pending"] == []


def test_max_hold_20_days():
    a = _buy(pa.new_account("us", "USD", 1000.0))
    for i in range(2, 21):
        a = pa.run_day(a, D[i], {"AAA": bar(10)}, rate(("AAA", 12.0)), US)
    assert a["positions"]["AAA"]["held_days"] == 20
    assert "max hold" in a["pending"][0]["reason"]


def test_sell_fill_records_pnl_and_cooldown_after_stop():
    a = _buy(pa.new_account("us", "USD", 1000.0), price=10.0, atr=0.5)
    a = pa.run_day(a, D[2], {"AAA": bar(9, 8.5)}, rate(("AAA", 12.0)), US)
    a = pa.run_day(a, D[3], {"AAA": bar(8.4)}, rate(("AAA", 12.0)), US)
    assert "AAA" not in a["positions"] and a["trades"][-1]["pnl"] < 0
    assert all(p["ticker"] != "AAA" for p in a["pending"])        # cooldown: no re-buy
    for i in range(4, 8):
        a = pa.run_day(a, D[i], {"AAA": bar(8.4)}, rate(("AAA", 12.0)), US)
    assert any(p["ticker"] == "AAA" for p in a["pending"])        # after 5 trading days


def test_china_lots_of_100_and_unaffordable_skipped():
    a = pa.new_account("china", "CNY", 1000.0)
    a = pa.run_day(a, D[0], {"CHEAP": bar(8), "PRICEY": bar(30)},
                   rate(("PRICEY", 15.0), ("CHEAP", 12.0)), CN)
    assert [p["ticker"] for p in a["pending"]] == ["CHEAP"]   # one lot of PRICEY = 3000 > cash
    a = pa.run_day(a, D[1], {"CHEAP": bar(8.5)}, rate(("CHEAP", 12.0)), CN)
    assert a["positions"]["CHEAP"]["shares"] == 100           # a full lot, above the 200 slot
    assert a["cash"] == pytest.approx(1000 - 850 * 1.0015)


def test_china_lot_rounding_on_bigger_account():
    a = pa.new_account("china", "CNY", 10000.0)
    a = pa.run_day(a, D[0], {"X": bar(9)}, rate(("X", 12.0)), CN)
    a = pa.run_day(a, D[1], {"X": bar(9)}, rate(("X", 12.0)), CN)
    assert a["positions"]["X"]["shares"] == 200     # slot 2000 / (9*1.0015*100) -> 2 lots


def test_no_bar_keeps_order_pending():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    a = pa.run_day(a, D[1], {}, rate(("AAA", 12.0)), US)
    assert a["positions"] == {} and [p["ticker"] for p in a["pending"]] == ["AAA"]


def test_same_day_rerun_is_idempotent():
    a = pa.new_account("us", "USD", 1000.0)
    a = pa.run_day(a, D[0], {}, rate(("AAA", 12.0)), US)
    b = pa.run_day(a, D[0], {}, rate(("BBB", 12.0)), US)
    assert [p["ticker"] for p in b["pending"]] == ["AAA"] and len(b["equity"]) == 1


def test_no_signals_holds_cash():
    a = pa.run_day(pa.new_account("us", "USD", 1000.0), D[0], {}, [], US)
    assert a["pending"] == [] and a["equity"][-1]["total"] == 1000.0
