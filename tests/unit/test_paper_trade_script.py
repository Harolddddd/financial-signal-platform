import json
from datetime import datetime, timezone

import polars as pl

from scripts import paper_trade as pt


def _raw(dirp, ticker, rows):
    pl.DataFrame({
        "time": [datetime(2026, 11, d, tzinfo=timezone.utc) for d, _, _ in rows],
        "open": [o for _, o, _ in rows], "high": [max(o, c) + 1 for _, o, c in rows],
        "low": [min(o, c) - 1 for _, o, c in rows], "close": [c for _, _, c in rows],
    }).write_parquet(dirp / f"{ticker}.parquet")


def _no_risk(market, today, tickers):
    return set(), {"checked": len(tickers)}


def test_daily_run_uses_today_ratings_and_bars(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    _raw(raw, "SPY", [(9, 500, 501), (10, 502, 505)])
    _raw(raw, "AAA", [(9, 10, 10), (10, 11, 12)])
    ratings = [{"ticker": "AAA", "overall_rating": 15.0, "date": "2026-11-09 00:00:00+00:00", "entry_price": 10}]
    monkeypatch.setattr(pt, "get_combined_ratings", lambda market: (ratings, {}))
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY", risk_fn=_no_risk)
    # latest bar is 11-10, but the only rating is dated 11-09 -> stale, ignored
    assert out["last_day"] == "2026-11-10" and out["pending"] == []
    assert out["equity"][-1]["benchmark"] == 505
    _raw(raw, "SPY", [(9, 500, 501), (10, 502, 505), (11, 506, 507)])
    _raw(raw, "AAA", [(9, 10, 10), (10, 11, 12), (11, 13, 13)])
    ratings[0]["date"] = "2026-11-11 00:00:00+00:00"
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY", risk_fn=_no_risk)
    assert [p["ticker"] for p in out["pending"]] == ["AAA"]
    saved = json.loads((tmp_path / "paper_account.json").read_text())
    assert saved["start_cash"] == 2000.0 and saved["currency"] == "USD"
    assert saved["sell_model"] == {"checked": 1}


def test_sell_model_flag_blocks_buy_and_failure_is_tolerated(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    _raw(raw, "SPY", [(9, 500, 501)])
    _raw(raw, "AAA", [(9, 10, 10)])
    ratings = [{"ticker": "AAA", "overall_rating": 15.0, "date": "2026-11-09", "entry_price": 10}]
    monkeypatch.setattr(pt, "get_combined_ratings", lambda market: (ratings, {}))
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY",
                  risk_fn=lambda m, d, t: ({"AAA"}, {}))
    assert out["pending"] == []

    def boom(m, d, t):
        raise RuntimeError("no features")
    (tmp_path / "paper_account.json").unlink()
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY", risk_fn=boom)
    assert [p["ticker"] for p in out["pending"]] == ["AAA"] and "error" in out["sell_model"]


def test_china_account_in_cny(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    _raw(raw, "510300.SS", [(9, 4.0, 4.1)])
    monkeypatch.setattr(pt, "get_combined_ratings", lambda market: ([], {}))
    out = pt.main("china", raw_dir=raw, cache_dir=tmp_path, risk_fn=_no_risk)
    assert out["currency"] == "CNY" and out["cash"] == 10000.0


def test_atr_from_raw(tmp_path):
    raw = tmp_path
    _raw(raw, "AAA", [(d, 10, 10) for d in range(1, 17)])   # every day high-low = 2
    b = pt._bar(raw, "AAA", datetime(2026, 11, 16).date())
    assert b["atr"] == 2.0 and b["open"] == 10
