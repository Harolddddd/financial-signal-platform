import json
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from src.league import remote_bench as rb

NOW = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)
REQ = {"id": "china-2026-11-15-1", "market": "china", "today": "2026-11-15",
       "deadline": "2026-11-16T09:30:00+00:00", "methods": ["a", "b"], "created": NOW.isoformat()}


def _write_result(d, name, preds=None, strategy_name=None):
    jf, pf = rb.result_files(name)
    (d / jf).write_text(json.dumps({"strategy_name": strategy_name or name, "runtime_s": 12.5,
                                    "mean_sharpe": float("nan"), "n_folds": 51, "score": {"net_sharpe": 99}}))
    (preds if preds is not None else pl.DataFrame(
        {"ticker": ["X"], "time": [date(2026, 1, 2)], "buy_score": [0.7]}, schema=rb.PRED_SCHEMA)
     ).write_parquet(d / pf)


def test_split_balances_and_respects_exclude_and_cap():
    rt = {"slow": 600, "mid": 300, "s1": 100, "s2": 100, "stacked_live": 400}
    local, remote = rb.split_methods(list(rt), rt, exclude=["stacked_live"])
    assert "stacked_live" in local and "stacked_live" not in remote
    assert sorted(local + remote) == sorted(rt)
    t_local = sum(rt[n] for n in local)
    t_remote = sum(rt[n] for n in remote)
    assert abs(t_local - t_remote) <= 300  # balanced within one mid-sized method
    # A busy with the other market for a long time -> B takes up to the cap (80%).
    local, remote = rb.split_methods(list(rt), rt, local_head_start_s=10_000, exclude=["stacked_live"])
    assert sum(rt[n] for n in remote) <= 0.8 * sum(rt.values())
    assert remote  # B got work


def test_split_unknown_runtime_uses_median():
    local, remote = rb.split_methods(["a", "b", "c"], {"a": 100, "b": 0})
    assert sorted(local + remote) == ["a", "b", "c"]


def test_manifest_round_trip_imports_and_never_trusts_b_score(tmp_path):
    _write_result(tmp_path, "a")
    m = rb.write_manifest(tmp_path, REQ, ["a", "b"], complete=False, now=NOW)
    assert list(m["methods"]) == ["a"]  # b has no files yet -> not listed
    got, rejected = rb.import_results(tmp_path, REQ, rb.read_manifest(tmp_path, REQ), skip=set())
    assert rejected == [] and list(got) == ["a"]
    assert got["a"]["meta"]["runtime_s"] == 12.5 and got["a"]["meta"]["mean_sharpe"] is None
    assert "score" not in got["a"]["meta"]
    assert got["a"]["preds"].height == 1
    assert rb.import_results(tmp_path, REQ, m, skip={"a"})[0] == {}


@pytest.mark.parametrize("tamper,reason", [
    ("hash", "hash mismatch"), ("unassigned", "not assigned"), ("schema", "schema"), ("name", "another method"),
])
def test_import_rejects_tampered_results(tmp_path, tamper, reason):
    name = "zzz" if tamper == "unassigned" else "a"
    if tamper == "schema":
        _write_result(tmp_path, name, preds=pl.DataFrame({"ticker": ["X"], "buy_score": ["1"]}))
    elif tamper == "name":
        _write_result(tmp_path, name, strategy_name="b")
    else:
        _write_result(tmp_path, name)
    m = rb.write_manifest(tmp_path, REQ, [name], complete=True, now=NOW)
    if tamper == "hash":
        (tmp_path / rb.result_files(name)[1]).write_bytes(b"x" * m["methods"][name][rb.result_files(name)[1]]["bytes"])
    got, rejected = rb.import_results(tmp_path, REQ, m, skip=set())
    assert got == {} and len(rejected) == 1 and reason in rejected[0]


def test_manifest_for_another_request_is_ignored(tmp_path):
    rb.write_manifest(tmp_path, {**REQ, "id": "old"}, [], complete=True, now=NOW)
    assert rb.read_manifest(tmp_path, REQ) is None


def test_remote_alive_uses_heartbeat_age():
    m = {"updated": NOW.isoformat()}
    assert rb.remote_alive(m, NOW + timedelta(minutes=20), stale_s=1800)
    assert not rb.remote_alive(m, NOW + timedelta(minutes=40), stale_s=1800)
    assert not rb.remote_alive(None, NOW, stale_s=1800)


def test_publish_writes_request_last_and_retract_clears(tmp_path):
    feats = tmp_path / "features"; feats.mkdir()
    (feats / "AAA.parquet").write_bytes(b"pq")
    out = tmp_path / "outgoing"
    rb.publish_request(out, REQ, feats)
    assert (out / "features" / "AAA.parquet").read_bytes() == b"pq"
    assert rb.load_request(out / rb.REQUEST_FILE)["id"] == REQ["id"]
    rb.retract_request(out)
    assert not (out / rb.REQUEST_FILE).exists() and not (out / "features").exists()
    with pytest.raises(ValueError):
        rb.publish_request(out, {"id": "x"}, feats)
