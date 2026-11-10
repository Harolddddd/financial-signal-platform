"""Intraday stop checks on held paper positions (every 5 min while the market is open)."""
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.paper import account as pa

US = pa.PaperConfig(cost_bps=10.0, lot_size=None)
CN = pa.PaperConfig(cost_bps=15.0, lot_size=100)


def _acct(entry=100.0, atr=5.0, peak=None, last=None, shares=10):
    a = pa.new_account("us", "USD", 1000.0)
    a["cash"] = 0.0
    a["positions"]["AAA"] = {"shares": shares, "entry_date": "2026-11-09", "entry_price": entry,
                             "cost_basis": entry * shares, "held_days": 1, "last_price": last or entry,
                             "peak": peak or entry, "entry_atr": atr, "atr": atr}
    a["equity"] = [{"date": "2026-11-09", "equity": entry * shares}]
    return a


def test_stop_hit_intraday_sells_at_that_price_with_costs_and_cooldown():
    a = _acct()  # stop = 100 - 2*5 = 90
    sold = pa.intraday_stop(a, "2026-11-10", {"AAA": 89.0}, US)
    assert [s["ticker"] for s in sold] == ["AAA"] and "AAA" not in a["positions"]
    assert a["cash"] == pytest.approx(10 * 89.0 * (1 - 0.001))
    t = a["trades"][-1]
    assert t["side"] == "sell" and t["reason"].startswith("intraday stop-loss")
    assert a["cooldown"]["AAA"] == len(a["equity"]) + US.cooldown_days


def test_price_above_stop_keeps_position():
    a = _acct()
    assert pa.intraday_stop(a, "2026-11-10", {"AAA": 91.0}, US) == []
    assert "AAA" in a["positions"]


def test_trailing_stop_intraday():
    a = _acct(peak=120.0)  # up 4 ATR: trail at 120 - 2*5 = 110
    sold = pa.intraday_stop(a, "2026-11-10", {"AAA": 109.0}, US)
    assert sold and a["trades"][-1]["reason"].startswith("intraday trailing stop")
    assert "AAA" not in a["cooldown"]  # like the daily rule: only a stop-loss starts a cooldown


def test_pending_sell_for_sold_stock_is_dropped():
    a = _acct()
    a["pending"].append({"ticker": "AAA", "side": "sell", "signal_date": "2026-11-09", "reason": "max hold"})
    pa.intraday_stop(a, "2026-11-10", {"AAA": 80.0}, US)
    assert a["pending"] == []


def test_china_limit_down_cannot_be_sold():
    a = _acct(entry=10.0, atr=0.3, last=10.0, shares=100)  # stop 9.4; limit-down = 9.0
    assert pa.intraday_stop(a, "2026-11-10", {"AAA": 9.0}, CN) == []
    assert "AAA" in a["positions"]
    sold = pa.intraday_stop(a, "2026-11-10", {"AAA": 9.2}, CN)
    assert sold and a["cash"] == pytest.approx(100 * 9.2 * (1 - 0.0015))


def test_daily_rule_uses_the_same_stop_levels():
    p = _acct()["positions"]["AAA"]
    assert pa.stop_hit(p, 90.0, US).startswith("stop-loss")
    assert pa.stop_hit(p, 90.5, US) is None


# --- script ---------------------------------------------------------------------------

def test_sessions():
    from scripts import intraday_guard as ig
    ny, sh = ZoneInfo("America/New_York"), ZoneInfo("Asia/Shanghai")
    assert ig.in_session("us", datetime(2026, 11, 10, 10, 0, tzinfo=ny))
    assert not ig.in_session("us", datetime(2026, 11, 10, 16, 5, tzinfo=ny))
    assert not ig.in_session("us", datetime(2026, 11, 14, 11, 0, tzinfo=ny))  # Saturday
    assert ig.in_session("china", datetime(2026, 11, 10, 10, 0, tzinfo=sh))
    assert not ig.in_session("china", datetime(2026, 11, 10, 12, 0, tzinfo=sh))  # lunch break
    assert ig.in_session("china", datetime(2026, 11, 10, 14, 59, tzinfo=sh))


def test_guard_sells_on_fresh_quote_and_ignores_stale(tmp_path, monkeypatch):
    from scripts import intraday_guard as ig
    acct = _acct()
    acct["positions"]["BBB"] = dict(acct["positions"]["AAA"])
    (tmp_path / "paper_account.json").write_text(json.dumps(acct))
    now = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)  # 10:00 ET
    quotes = {"AAA": (80.0, now - timedelta(minutes=2)), "BBB": (80.0, now - timedelta(hours=3))}
    out = ig.check_market("us", tmp_path, now=now, quote=lambda t: quotes[t])
    assert out["sold"] == ["AAA"] and out["stale"] == ["BBB"]
    saved = json.loads((tmp_path / "paper_account.json").read_text())
    assert "AAA" not in saved["positions"] and "BBB" in saved["positions"]


def test_guard_does_nothing_outside_session_or_without_positions(tmp_path):
    from scripts import intraday_guard as ig
    (tmp_path / "paper_account.json").write_text(json.dumps(_acct()))
    called = []
    sat = datetime(2026, 11, 14, 16, 0, tzinfo=timezone.utc)
    assert ig.check_market("us", tmp_path, now=sat, quote=lambda t: called.append(t))["skipped"] == "closed"
    empty = pa.new_account("us", "USD", 1000.0)
    (tmp_path / "paper_account.json").write_text(json.dumps(empty))
    now = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)
    assert ig.check_market("us", tmp_path, now=now, quote=lambda t: called.append(t))["skipped"] == "no positions"
    assert called == []
