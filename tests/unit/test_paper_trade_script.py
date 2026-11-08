import json
from datetime import datetime, timezone

import polars as pl

from scripts import paper_trade as pt


def _raw(dirp, ticker, rows):
    pl.DataFrame({
        "time": [datetime(2026, 11, d, tzinfo=timezone.utc) for d, _, _ in rows],
        "open": [o for _, o, _ in rows], "close": [c for _, _, c in rows],
    }).write_parquet(dirp / f"{ticker}.parquet")


def test_daily_run_uses_today_ratings_and_bars(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    _raw(raw, "SPY", [(9, 500, 501), (10, 502, 505)])
    _raw(raw, "AAA", [(9, 10, 10), (10, 11, 12)])
    ratings = [
        {"ticker": "AAA", "overall_rating": 15.0, "date": "2026-11-09 00:00:00+00:00", "entry_price": 10},
    ]
    monkeypatch.setattr(pt, "get_combined_ratings", lambda market: (ratings, {}))
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY", today_override=None)
    # latest bar is 11-10, but the only rating is dated 11-09 -> stale, ignored
    assert out["last_day"] == "2026-11-10" and out["pending"] == []
    assert out["equity"][-1]["benchmark"] == 505
    ratings[0]["date"] = "2026-11-10 00:00:00+00:00"
    _raw(raw, "SPY", [(9, 500, 501), (10, 502, 505), (11, 506, 507)])
    _raw(raw, "AAA", [(9, 10, 10), (10, 11, 12), (11, 13, 13)])
    ratings[0]["date"] = "2026-11-11 00:00:00+00:00"
    out = pt.main("us", raw_dir=raw, cache_dir=tmp_path, benchmark="SPY")
    assert [p["ticker"] for p in out["pending"]] == ["AAA"]
    saved = json.loads((tmp_path / "paper_account.json").read_text())
    assert saved["start_cash"] == 1000.0 and saved["currency"] == "USD"


def test_china_account_in_cny(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    _raw(raw, "510300.SS", [(9, 4.0, 4.1)])
    monkeypatch.setattr(pt, "get_combined_ratings", lambda market: ([], {}))
    out = pt.main("china", raw_dir=raw, cache_dir=tmp_path)
    assert out["currency"] == "CNY" and out["cash"] == 1000.0
