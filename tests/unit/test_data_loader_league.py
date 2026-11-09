import json
from datetime import date

import pytest

from dashboard import data_loader as dl
from src.league import league as lg

TODAY = date(2026, 12, 1)


def _h(d, s, trades=50):
    return {"date": d, "net_sharpe": s, "n_trades": trades, "precision": 0.55, "error": None}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "get_cache_dir", lambda market: tmp_path)
    return tmp_path


def _save(cache, methods):
    st = lg.new_state("us", seed=[])
    for n, (status, hist, extra) in methods.items():
        m = lg.ensure_method(st, n)
        m.update(status=status, history=hist, **extra)
    lg.save_state(cache, st)


def test_sections_staleness_and_trend(cache):
    _save(cache, {
        "a": ("live", [_h("2026-11-15", 1.0), _h("2026-11-25", 1.3)], {}),       # 6 days: fresh
        "b": ("live", [_h("2026-11-20", 0.5)], {}),                              # 11 days: stale (>8)
        "c": ("bench", [_h("2026-11-01", 0.9), _h("2026-11-17", 0.6)], {"duplicate_of": "a"}),  # 14 d fresh
        "d": ("bench", [_h("2026-11-10", 0.2)], {}),                             # 21 d stale (>15)
        "e": ("retired", [_h("2026-11-10", 0.1)], {"retired_because": "similar to a since 2026-10-20"}),
    })
    t = dl.get_league_table("us", today=TODAY)
    assert [r["method"] for r in t["live"]] == ["a", "b"]  # sorted by net_sharpe
    row = {r["method"]: r for sec in t.values() for r in sec}
    assert (row["a"]["stale"], row["b"]["stale"], row["c"]["stale"], row["d"]["stale"]) == (False, True, False, True)
    assert (row["a"]["trend"], row["c"]["trend"], row["b"]["trend"]) == ("↑", "↓", "")
    assert row["c"]["duplicate_of"] == "a" and row["a"]["tested_on"] == "2026-11-25"
    assert t["retired"][0]["retired_because"].startswith("similar to a")


def test_fallback_without_league_uses_backtest_files(cache, monkeypatch):
    monkeypatch.setattr(dl, "list_strategies", lambda market=None: ["x", "y"])
    (cache / "backtest_x.json").write_text(json.dumps({
        "generated_at": "2026-08-04T14:20:00+00:00", "mean_sharpe": 0.7, "weighted_precision_buy": 0.4,
        "grade": {"metrics": {"n_trades": 123}},
    }))
    t = dl.get_league_table("us", today=TODAY)
    assert t["live"] == [] and t["retired"] == []
    row = {r["method"]: r for r in t["bench"]}
    assert row["x"]["tested_on"] == "2026-08-04" and row["x"]["stale"] and row["x"]["n_trades"] == 123
    assert row["y"]["tested_on"] == "" and row["y"]["stale"]


def test_beijing_time():
    from dashboard.system_status import beijing_time
    assert beijing_time("2026-11-08T02:47:12Z") == "11-08 10:47 CST"


def test_live_tested_on_uses_weekly_backtest_date(cache):
    _save(cache, {"a": ("live", [_h("2026-11-15", 1.0)], {})})
    (cache / "backtest_a.json").write_text(json.dumps({"generated_at": "2026-11-29T10:00:00+00:00"}))
    r = dl.get_league_table("us", today=TODAY)["live"][0]
    assert r["tested_on"] == "2026-11-29" and r["stale"] is False


def test_ticker_names_and_generated_time(tmp_path, monkeypatch):
    import polars as pl
    from types import SimpleNamespace
    monkeypatch.setattr(dl, "get_market", lambda m: SimpleNamespace(data_root=tmp_path))
    assert dl.get_ticker_names("china") == {}
    (tmp_path / "index_compositions").mkdir()
    pl.DataFrame({"ticker": ["600000.SS"], "name": ["浦发银行"]}).write_parquet(
        tmp_path / "index_compositions" / "names.parquet")
    assert dl.get_ticker_names("china") == {"600000.SS": "浦发银行"}
    assert dl.get_ticker_names("us") == {}
    assert dl.format_generated("2026-11-08T02:27:41.966+00:00", "us") == "11-07 21:27 EST"
    assert dl.format_generated("2026-11-08T02:27:41+00:00", "china") == "11-07 21:27 EST (11-08 10:27 Beijing)"
    assert dl.format_generated("unknown", "us") == "unknown"


def test_rows_show_where_and_how_long_and_pass_status(cache, monkeypatch):
    _save(cache, {"a": ("live", [_h("2026-11-25", 1.0)], {}), "b": ("bench", [], {})})
    st = lg.load_state(cache, "us", seed=[])
    st["passes"].append({"date": "2026-11-25", "completed": False})
    lg.save_state(cache, st)
    (cache / "bench").mkdir()
    (cache / "bench" / "bench_a.json").write_text(json.dumps({"ran_on": "B", "runtime_s": 600}))
    monkeypatch.setattr(dl, "list_strategies", lambda market=None: ["a", "b", "c"])  # c: never tested
    t = dl.get_league_table("us", today=TODAY)
    a, b = t["live"][0], t["bench"][0]
    assert (a["ran_on"], a["runtime_min"]) == ("B", 10.0)
    assert (b["ran_on"], b["runtime_min"]) == (None, None)
    assert dl.get_league_pass("us") == {"date": "2026-11-25", "completed": False, "untested": ["b", "c"]}
