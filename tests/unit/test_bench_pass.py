import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest

from scripts import bench_pass as bp
from src.league import league as lg
from src.strategies.base import PredictionResult, Strategy

TODAY = date(2026, 11, 15)


class _Buy(Strategy):
    data_source = "ohlcv"

    def __init__(self, conf=0.9, every=2):
        self.conf, self.every = conf, every

    def predict(self, df):
        n = len(df)
        sig = ["Buy" if i % self.every == 0 else "Hold" for i in range(n)]
        return PredictionResult(confidence=pd.Series([self.conf] * n), signal=pd.Series(sig))


class _Boom(Strategy):
    data_source = "ohlcv"

    def predict(self, df):
        raise RuntimeError("boom")


class _Fits(Strategy):
    """Stateful, raises in fit -> walk-forward produces no folds -> error."""
    data_source = "ohlcv"

    def fit(self, df):
        raise RuntimeError("fit boom")

    def predict(self, df):
        raise RuntimeError("unused")


STRATS = {"buy2": lambda: _Buy(0.9, 2), "buy3": lambda: _Buy(0.8, 3), "boom": lambda: _Fits()}


def _df(_dir=None, start=None, **_):
    rng = np.random.default_rng(1)
    days = [datetime(2022, 1, 3, tzinfo=timezone.utc) + timedelta(days=i) for i in range(1780)]
    rows = []
    for tk in ("AAA", "BBB", "CCC"):
        for t in days:
            r = float(rng.normal(0.003, 0.02))
            rows.append((tk, t, 100.0, "Buy" if r > 0 else "Hold", r))
    df = pl.DataFrame(rows, schema=["ticker", "time", "close", "label", "forward_return_5d"], orient="row")
    return df.filter(pl.col("time") >= start) if start is not None else df


def _run(tmp_path, **kw):
    return bp.main("us", feature_dir=tmp_path, cache_dir=tmp_path, today=TODAY,
                   load=_df, loader=lambda n: STRATS[n](), strategies=list(STRATS), **kw)


def test_scores_and_writes_files(tmp_path):
    out = _run(tmp_path, force=True)
    assert set(out["tested"]) == {"buy2", "buy3"} and out["errors"] == ["boom"]
    js = json.loads((tmp_path / "bench" / "bench_buy2.json").read_text())
    assert js["tested_on"] == TODAY.isoformat() and js["window_days"] == 1095
    assert js["score"]["n_trades"] > 20
    assert (tmp_path / "bench" / "preds_buy2.parquet").exists()
    st = json.loads((tmp_path / "league.json").read_text())
    assert st["methods"]["boom"]["history"][-1]["error"]


def test_skips_when_recent(tmp_path):
    _run(tmp_path, force=True)
    out = bp.main("us", feature_dir=tmp_path, cache_dir=tmp_path, today=TODAY + timedelta(days=7),
                  load=_df, loader=lambda n: STRATS[n](), strategies=list(STRATS))
    assert out == {"skipped": "recent"}


def test_runs_after_gap(tmp_path):
    _run(tmp_path, force=True)
    out = bp.main("us", feature_dir=tmp_path, cache_dir=tmp_path, today=TODAY + timedelta(days=14),
                  load=_df, loader=lambda n: STRATS[n](), strategies=list(STRATS))
    assert "skipped" not in out


def test_deadline_stops_and_untested_go_first_next_time(tmp_path):
    out = _run(tmp_path, force=True, deadline=datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert out["stopped"] and out["tested"] == []
    st = json.loads((tmp_path / "league.json").read_text())
    assert st["passes"] == []  # nothing tested -> no pass recorded
    # order puts never-tested methods first
    st = lg.load_state(tmp_path, "us", seed=[])
    st["methods"]["buy3"] = {**lg._blank_method("bench"), "history": [{"date": "2026-10-01"}]}
    assert bp.test_order(["buy3", "buy2"], st)[0] == "buy2"


def test_seed_from_pass(tmp_path):
    out = _run(tmp_path, force=True, seed_from_pass=True)
    assert set(out["live"]) <= {"buy2", "buy3"} and out["live"]


def test_precompute_uses_league_live_set(tmp_path, monkeypatch):
    import scripts.precompute_dashboard as pd_
    st = lg.new_state("us", seed=["random_forest"])
    lg.save_state(tmp_path, st)
    monkeypatch.setattr(pd_, "_market_paths", lambda m: (tmp_path, tmp_path))
    seen = {}
    monkeypatch.setattr(pd_, "step_signals", lambda names, **k: seen.setdefault("names", names))
    monkeypatch.setattr(pd_, "step_data_summary", lambda *a, **k: None)
    pd_.main("us", signals_only=True)
    assert seen["names"] == ["random_forest"]


def test_nothing_tested_records_no_pass(tmp_path):
    _run(tmp_path, force=True, deadline=datetime(2000, 1, 1, tzinfo=timezone.utc))
    st = json.loads((tmp_path / "league.json").read_text())
    assert st["passes"] == []


def test_skips_method_that_cannot_finish_before_deadline(tmp_path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "bench_buy2.json").write_text(json.dumps({"runtime_s": 10**7}))
    out = _run(tmp_path, force=True, deadline=datetime.now(timezone.utc) + timedelta(hours=1))
    assert "buy2" not in out["tested"] and "buy3" in out["tested"]
    assert "buy2" in out["skipped_for_time"]


def test_runtime_recorded(tmp_path):
    _run(tmp_path, force=True)
    js = json.loads((tmp_path / "bench" / "bench_buy3.json").read_text())
    assert js["runtime_s"] >= 0
