from dashboard import system_status as ss


def test_parse_show_splits_on_first_equals():
    out = ss.parse_show("ActiveState=active\nExecMainStartTimestamp=Sun 2026-11-08 02:28:32 UTC\nX=a=b\n")
    assert out == {
        "ActiveState": "active",
        "ExecMainStartTimestamp": "Sun 2026-11-08 02:28:32 UTC",
        "X": "a=b",
    }


def test_human_bytes():
    assert ss.human_bytes(str(3 * 2**30)) == "3 GB"
    assert ss.human_bytes("[not set]") == "-"
    assert ss.human_bytes("512") == "512 B"
    assert ss.human_bytes(str(5 * 2**20)) == "5 MB"


def test_parse_backfill_progress_line():
    text = (
        "state=running\ndays=150\nstarted=2026-11-08T02:28:32Z\nupdated=2026-11-08T03:00:00Z\n"
        "last=2026-11-08 03:00:00,000 INFO news progress 120/511 PEP new_total=40213\nmem=1500/3719 MB\n"
    )
    info = ss.parse_backfill(text)
    assert info["state"] == "running"
    assert (info["done"], info["total"], info["ticker"], info["new_total"]) == (120, 511, "PEP", 40213)
    assert abs(info["pct"] - 120 / 511) < 1e-9


def test_parse_backfill_done_without_progress_line():
    info = ss.parse_backfill("state=done\nlast=INFO News backfill: ok=511 failed=[] new=9\n")
    assert info["pct"] == 1.0


def test_backfill_status_missing_file(tmp_path):
    assert ss.backfill_status(tmp_path / "nope.txt") is None


def test_summarize_run_steps_and_durations():
    from datetime import datetime
    lines = [
        "=== 2026-11-08T01:30:59Z scripts/refresh_data.py --market us ===",
        "2026-11-08 01:32:18,954 ERROR $SMAR: possibly delisted; no timezone found",
        "=== 2026-11-08T01:59:21Z scripts/incremental_train.py --market us ===",
        "2026-11-08 01:59:25,000 WARNING something",
        "=== 2026-11-08T01:59:30Z scripts/precompute_dashboard.py --market us --signals-only ===",
        "=== 2026-11-08T02:17:56Z daily pipeline done ===",
    ]
    out = ss.summarize_run(lines, running=False)
    assert [s["step"] for s in out["steps"]] == [
        "Refresh data (prices, news, macro, features)", "Incremental training", "Generate signals",
    ]
    assert [s["duration"] for s in out["steps"]] == ["28m 22s", "0m 09s", "18m 26s"]
    assert all(s["status"] == "done" for s in out["steps"])
    assert out["steps"][0]["started (ET)"] == "11-07 20:30 EST"
    assert (out["errors"], out["warnings"], out["finished"]) == (1, 1, True)


def test_summarize_run_marks_last_step_running_or_failed():
    from datetime import datetime
    lines = ["=== 2026-11-08T02:29:41Z scripts/train_models.py --market us ==="]
    now = datetime(2026, 11, 8, 3, 29, 41)
    run = ss.summarize_run(lines, running=True, now=now)["steps"][0]
    assert (run["step"], run["status"], run["duration"]) == ("Full retrain", "running", "1h 00m")
    assert ss.summarize_run(lines, running=False, now=now)["steps"][0]["status"] == "failed"


def test_last_invocation_from_journal_takes_newest():
    lines = "\n".join([
        '{"_SYSTEMD_INVOCATION_ID": "aaa"}',
        'not json',
        '{"INVOCATION_ID": "bbb"}',
        '{"MESSAGE": "x"}',
    ])
    assert ss.last_invocation_from_journal(lines) == "bbb"
    assert ss.last_invocation_from_journal("") == ""


def test_to_et_handles_dst_and_bad_input():
    assert ss.to_et("2026-11-08T02:47:12Z") == "11-07 21:47 EST"
    assert ss.to_et("2026-07-01T16:00:00Z") == "07-01 12:00 EDT"
    assert ss.to_et("?") == "?"


def test_china_and_bench_steps_named():
    lines = [
        "=== 2026-11-10T10:30:00Z applying china upload 2026-11-10T08:40:00Z ===",
        "=== 2026-11-10T10:31:00Z scripts/refresh_data.py --market china ===",
        "=== 2026-11-10T10:50:00Z scripts/bench_pass.py --market us --deadline x ===",
        "=== 2026-11-10T11:00:00Z cn-daily pipeline done ===",
    ]
    steps = [s["step"] for s in ss.summarize_run(lines, running=False)["steps"]]
    assert steps == ["Apply China news from Server B",
                     "Refresh data (prices, news, macro, features)",
                     "Bench re-test (us)"]


def test_china_jobs_listed():
    for unit in ("fsp-cn-daily", "fsp-cn-weekly", "fsp-bench"):
        assert unit in ss.SERVICES
        assert f"{unit}.timer" in ss.TIMERS


def test_extra_units_lists_unscheduled_running_jobs():
    out = (
        "fsp-backup.service      loaded inactive dead    FSP nightly data backup\n"
        "fsp-bench-first.service loaded active   running /bin/bash -c \"bench\"\n"
        "fsp-dashboard.service   loaded active   running FSP Streamlit dashboard\n"
        "fsp-rescore-us.service  loaded active   running /bin/bash -c \"rescore\"\n"
    )
    assert ss.parse_extra_units(out) == ["fsp-bench-first", "fsp-rescore-us"]


def test_parse_progress_bench_on_a_and_b_and_news():
    a = ["2026-11-09 12:28:38,286 INFO bench china 1/12: pooled_lgbm_news_cn",
         "2026-11-09 12:37:55,695 INFO bench china 2/12: regime_consensus", "fold 3/51"]
    assert ss.parse_progress(a) == {"done": 1, "total": 12, "item": "regime_consensus", "pct": 1 / 12}
    b = ["INFO bench china (B) 5/23: knn_classifier"]
    assert ss.parse_progress(b)["item"] == "knn_classifier" and ss.parse_progress(b)["done"] == 4
    n = ["INFO news progress 120/500 600519.SS new_total=40"]
    assert ss.parse_progress(n) == {"done": 120, "total": 500, "item": "600519.SS", "pct": 0.24}
    assert ss.parse_progress(["nothing here"]) is None


def test_collect_snapshot_shape(monkeypatch):
    monkeypatch.setattr(ss, "unit_info", lambda u: {"ActiveState": "activating", "SubState": "start", "Result": "success"})
    monkeypatch.setattr(ss, "timer_info", lambda t: {"ActiveState": "active", "NextElapseUSecRealtime": "Tue"})
    monkeypatch.setattr(ss, "extra_units", lambda known=None: [])
    monkeypatch.setattr(ss, "last_run_lines", lambda u: [
        "=== 2026-11-09T17:57:52Z bench request china-x ===", "INFO bench china (B) 2/23: lda"])
    monkeypatch.setattr(ss, "recent_log", lambda u, n=30: "log")
    snap = ss.collect("B", {"fsp-bench-worker": "Bench worker"}, ["fsp-bench-worker.timer"],
                      now=ss.datetime(2026, 11, 9, 18, 0, tzinfo=ss.timezone.utc))
    assert snap["host"] == "B" and snap["updated"] == "2026-11-09T18:00:00Z"
    assert {"cpu_pct", "mem_used", "mem_total", "disk_used", "disk_total"} <= set(snap["resources"])
    u = snap["units"]["fsp-bench-worker"]
    assert u["label"] == "Bench worker" and u["running"] is True
    assert u["summary"]["steps"][0]["status"] == "running" and u["progress"]["done"] == 1
    assert snap["timers"]["fsp-bench-worker.timer"]["NextElapseUSecRealtime"] == "Tue"


def test_snapshot_staleness(tmp_path):
    import json
    p = tmp_path / "server_b.json"
    assert ss.read_snapshot(p) is None
    p.write_text(json.dumps({"host": "B", "updated": "2026-11-09T18:00:00Z"}))
    snap = ss.read_snapshot(p)
    assert not ss.is_stale(snap, ss.datetime(2026, 11, 9, 18, 10, tzinfo=ss.timezone.utc))
    assert ss.is_stale(snap, ss.datetime(2026, 11, 9, 18, 16, tzinfo=ss.timezone.utc))


def test_extra_units_respects_each_servers_own_list():
    out = "fsp-bench-worker.service loaded activating start FSP bench worker\n"
    assert ss.parse_extra_units(out) == ["fsp-bench-worker"]  # unknown to A
    assert ss.parse_extra_units(out, known=ss.SERVICES_B) == []


def test_systemd_time_to_et_and_plain_state():
    assert ss.systemd_to_et("Mon 2026-11-09 17:50:06 UTC") == "11-09 12:50 EST"
    assert ss.systemd_to_et("Mon 2026-11-09 12:50:06 EST") == "11-09 12:50 EST"
    assert ss.systemd_to_et("") == "-" and ss.systemd_to_et("n/a") == "n/a"
    assert ss.plain_state({"ActiveState": "activating"}) == "running"
    assert ss.plain_state({"ActiveState": "inactive", "Result": "success"}) == "idle"
    assert ss.plain_state({"ActiveState": "failed"}) == "failed"


def test_collect_names_one_offs_and_skips_self(monkeypatch):
    monkeypatch.setattr(ss, "unit_info", lambda u: {"ActiveState": "activating"})
    monkeypatch.setattr(ss, "timer_info", lambda t: {})
    monkeypatch.setattr(ss, "extra_units", lambda known=None: ["fsp-split-trial"])
    monkeypatch.setattr(ss, "last_run_lines", lambda u: [])
    monkeypatch.setattr(ss, "recent_log", lambda u, n=30: "")
    snap = ss.collect("B", {"fsp-status-b": "Status"}, [])
    assert snap["units"]["fsp-split-trial"]["label"] == "One-off: split-trial"
    assert snap["units"]["fsp-status-b"]["running"] is False
