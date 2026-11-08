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
