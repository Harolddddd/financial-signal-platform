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
