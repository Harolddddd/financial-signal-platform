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


# --- split with the compute server (B) -------------------------------------------------
from scripts import bench_worker as bw  # noqa: E402
from src.league import remote_bench as rb  # noqa: E402

RSTRATS = {"buy2": lambda: _Buy(0.9, 2), "buy3": lambda: _Buy(0.8, 3),
           "buy4": lambda: _Buy(0.7, 4), "buy5": lambda: _Buy(0.6, 5)}
RCFG = {"enabled": True, "remote_speed": 1.0, "max_share": 0.8, "exclude": [], "stale_minutes": 30}


def _remote_run(tmp_path, **kw):
    return bp.main("china", feature_dir=tmp_path, cache_dir=tmp_path / "cache", today=TODAY, force=True,
                   load=_df, loader=lambda n: RSTRATS[n](), strategies=list(RSTRATS), remote=RCFG,
                   outgoing=tmp_path / "out", inbox_root=tmp_path / "inbox", sleep=lambda s: None, **kw)


def _publish(tmp_path):
    runtimes = {n: 100.0 for n in RSTRATS}
    return bp.publish("china", TODAY, datetime(2099, 1, 1, tzinfo=timezone.utc), strategies=list(RSTRATS),
                      cfg=RCFG, cache_dir=tmp_path / "cache", feature_dir=tmp_path, outgoing=tmp_path / "out",
                      runtimes=runtimes, code_tar=None, load=_df)


def _worker(tmp_path, req, uploads=None):
    out = tmp_path / "inbox" / req["id"]
    return bw.main(tmp_path / "out" / rb.REQUEST_FILE, feature_dir=tmp_path, out_dir=out,
                   upload=lambda d: (uploads.append(json.loads((d / rb.MANIFEST_FILE).read_text()))
                                     if uploads is not None else None),
                   load=_df, loader=lambda n: RSTRATS[n](), heartbeat_s=None)


def test_split_runs_part_on_b_and_rescores_it_on_a(tmp_path):
    req = _publish(tmp_path)
    assert req["methods"] and set(req["methods"]) < set(RSTRATS)  # B got part, not all
    _worker(tmp_path, req)
    out = _remote_run(tmp_path)
    assert set(out["tested"]) == set(RSTRATS) and not out["stopped"]
    assert set(out["remote"]) == set(req["methods"])
    name = req["methods"][0]
    js = json.loads((tmp_path / "cache" / "bench" / f"bench_{name}.json").read_text())
    assert js["ran_on"] == "B" and js["score"]["n_trades"] > 20
    st = json.loads((tmp_path / "cache" / "league.json").read_text())
    assert st["passes"][-1]["completed"] is True
    assert not (tmp_path / "out" / rb.REQUEST_FILE).exists()  # retracted


def test_b_results_scored_by_a_not_by_b(tmp_path):
    req = _publish(tmp_path)
    _worker(tmp_path, req)
    name = req["methods"][0]
    inbox = tmp_path / "inbox" / req["id"]
    jf = inbox / f"bench_{name}.json"
    raw = json.loads(jf.read_text()); raw["score"]["net_sharpe"] = 99.0
    jf.write_text(json.dumps(raw))
    rb.write_manifest(inbox, req, list(json.loads((inbox / rb.MANIFEST_FILE).read_text())["methods"]), complete=True)
    _remote_run(tmp_path)
    js = json.loads((tmp_path / "cache" / "bench" / f"bench_{name}.json").read_text())
    assert js["score"]["net_sharpe"] != 99.0


def test_silent_b_methods_are_taken_back(tmp_path):
    req = _publish(tmp_path)
    late = iter([datetime.now(timezone.utc) + timedelta(hours=h) for h in range(1, 100)])
    out = _remote_run(tmp_path, clock=lambda: next(late))
    assert set(out["tested"]) == set(RSTRATS) and out["remote"] == []
    assert set(out["taken_back"]) == set(req["methods"])


def test_worker_uploads_every_quarter(tmp_path):
    req = _publish(tmp_path)
    req = {**req, "methods": list(RSTRATS)}
    (tmp_path / "out" / rb.REQUEST_FILE).write_text(json.dumps(req))
    uploads = []
    _worker(tmp_path, req, uploads)
    assert [len(u["methods"]) for u in uploads] == [0, 1, 2, 3, 4, 4]  # 0 = "started"
    assert [u["complete"] for u in uploads] == [False] * 5 + [True]


def test_publish_ships_one_trimmed_feature_file(tmp_path):
    _publish(tmp_path)
    files = list((tmp_path / "out" / "features").iterdir())
    assert [f.name for f in files] == ["features.parquet"]
    t = pl.read_parquet(files[0])["time"]
    assert t.min() >= datetime(2026, 11, 15, tzinfo=timezone.utc) - timedelta(days=1095 + 400 + 5)
    assert not (tmp_path / ".build-china").exists()


def test_b_gets_start_grace_before_takeover(tmp_path):
    _publish(tmp_path)
    # No sign of B: A waits through the 90 min start grace (clock moves 30 min per wait), then takes over.
    t0 = datetime.now(timezone.utc)
    slept = []
    out = bp.main("china", feature_dir=tmp_path, cache_dir=tmp_path / "cache", today=TODAY, force=True,
                  load=_df, loader=lambda n: RSTRATS[n](), strategies=list(RSTRATS), remote=RCFG,
                  outgoing=tmp_path / "out", inbox_root=tmp_path / "inbox", sleep=slept.append,
                  clock=lambda: t0 + timedelta(minutes=45 + 30 * len(slept)))
    assert len(slept) == 2 and out["taken_back"]  # waited at 45 and 75 min, took over at 105


def _plan(tmp_path, monkeypatch, cfgs, due=("us", "china")):
    monkeypatch.setattr(bp, "_remote_cfg", lambda m: cfgs.get(m))
    monkeypatch.setattr(bp, "_last_runtime", lambda d, n: 60.0)
    monkeypatch.setattr(bp, "list_strategies", lambda m: ["a", "b"])
    monkeypatch.setattr(bp, "_due", lambda m, today, cache_dir, force: m in due)
    posted = []

    def fake(market, today, deadline, strategies, cache_dir=None, head_start_s=0.0, remote_head_start_s=0.0, **k):
        # Regression: this path once called lg.load_state without its seed.
        bp._state(market, cache_dir)
        posted.append((market, head_start_s, remote_head_start_s))
        return {"id": market, "est": {"local_s": head_start_s + 60, "remote_s": remote_head_start_s + 60}}

    out = bp.publish_plan(["us", "china"], TODAY, cache_dir_for=lambda m: tmp_path / m, publish_fn=fake)
    return out, posted


def test_publish_plan_chains_both_markets(tmp_path, monkeypatch):
    cfg = dict(RCFG, download_minutes=10)
    out, posted = _plan(tmp_path, monkeypatch, {"us": cfg, "china": cfg})
    assert [r["id"] for r in out] == ["us", "china"]
    assert posted[0] == ("us", 0.0, 1200.0)          # B first downloads both markets (2 x 10 min)
    assert posted[1] == ("china", 1260.0, 1260.0)   # A waits for B's US results, then both do China


def test_publish_plan_a_runs_us_alone_when_us_not_split(tmp_path, monkeypatch):
    out, posted = _plan(tmp_path, monkeypatch, {"china": dict(RCFG, download_minutes=10)})
    assert posted == [("china", 120.0, 600.0)]      # A: US 2 x 60 s first; B: download


def test_publish_plan_skips_markets_not_due(tmp_path, monkeypatch):
    cfg = dict(RCFG, download_minutes=10)
    out, posted = _plan(tmp_path, monkeypatch, {"us": cfg, "china": cfg}, due=("china",))
    assert posted == [("china", 0.0, 600.0)]


def test_start_grace_covers_b_working_on_the_other_market(tmp_path):
    bp.publish("china", TODAY, datetime(2099, 1, 1, tzinfo=timezone.utc), strategies=list(RSTRATS),
               cfg=RCFG, cache_dir=tmp_path / "cache", feature_dir=tmp_path, outgoing=tmp_path / "out",
               runtimes={n: 10_000.0 for n in RSTRATS}, code_tar=None, load=_df, remote_head_start_s=3600)
    t0 = datetime.now(timezone.utc)
    slept = []
    out = bp.main("china", feature_dir=tmp_path, cache_dir=tmp_path / "cache", today=TODAY, force=True,
                  load=_df, loader=lambda n: RSTRATS[n](), strategies=list(RSTRATS), remote=RCFG,
                  outgoing=tmp_path / "out", inbox_root=tmp_path / "inbox", sleep=slept.append,
                  clock=lambda: t0 + timedelta(minutes=45 + 30 * len(slept)))
    # grace = 90 min + 1.5 x B's 60 min on US = 180 min: waits at 45..165, takes over at 195
    assert len(slept) == 5 and out["taken_back"]


def _parallel_worker(tmp_path, mem_ok):
    import threading
    req = {**_publish(tmp_path), "methods": list(RSTRATS)}
    (tmp_path / "out" / rb.REQUEST_FILE).write_text(json.dumps(req))
    live, peak, lock = [0], [0], threading.Lock()
    real = bp.run_method

    def tracked(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            import time as _t
            _t.sleep(0.2)
            return real(*a, **k)
        finally:
            with lock:
                live[0] -= 1

    uploads = []
    import unittest.mock as um
    with um.patch.object(bp, "run_method", tracked):
        out = bw.main(tmp_path / "out" / rb.REQUEST_FILE, feature_dir=tmp_path,
                      out_dir=tmp_path / "inbox" / req["id"],
                      upload=lambda d: uploads.append(json.loads((d / rb.MANIFEST_FILE).read_text())),
                      load=_df, loader=lambda n: RSTRATS[n](), heartbeat_s=None, parallel=2, mem_ok=mem_ok)
    return out, peak[0], uploads


def test_worker_runs_two_methods_at_once_when_memory_allows(tmp_path):
    out, peak, uploads = _parallel_worker(tmp_path, mem_ok=lambda: True)
    assert sorted(out["done"]) == sorted(RSTRATS) and peak == 2
    assert [len(u["methods"]) for u in uploads] == [0, 1, 2, 3, 4, 4]
    assert uploads[-1]["complete"] is True


def test_worker_waits_for_memory_before_a_second_method(tmp_path):
    out, peak, _ = _parallel_worker(tmp_path, mem_ok=lambda: False)
    assert sorted(out["done"]) == sorted(RSTRATS) and peak == 1


def test_rsync_upload_retries_a_failed_transfer(tmp_path, monkeypatch):
    import subprocess
    from scripts import bench_worker as bw

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(12, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw.time, "sleep", lambda s: None)
    bw.rsync_upload("dest/")(tmp_path)
    assert len(calls) == 3  # data (failed), data (retried), manifest


def test_rsync_upload_gives_up_after_retries(tmp_path, monkeypatch):
    import subprocess

    import pytest
    from scripts import bench_worker as bw

    def always_fail(cmd, **kw):
        raise subprocess.CalledProcessError(12, cmd)

    monkeypatch.setattr(bw.subprocess, "run", always_fail)
    monkeypatch.setattr(bw.time, "sleep", lambda s: None)
    with pytest.raises(subprocess.CalledProcessError):
        bw.rsync_upload("dest/")(tmp_path)
