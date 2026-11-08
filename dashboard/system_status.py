"""Read-only status of the server's systemd jobs and the compute server's backfill.

Works on the Linux production server; elsewhere `available()` is False and the
page shows a notice instead.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

SERVICES = {
    "fsp-daily": "Daily pipeline (refresh, incremental train, signals)",
    "fsp-weekly": "Weekly full retrain + backtests",
    "fsp-backup": "Nightly backup",
    "fsp-dashboard": "Dashboard",
}
TIMERS = ["fsp-daily.timer", "fsp-weekly.timer", "fsp-backup.timer"]
BACKFILL_STATUS = Path("/home/fspsync/incoming/status/backfill.txt")
_SHOW_PROPS = [
    "ActiveState", "SubState", "Result", "ExecMainStartTimestamp",
    "ExecMainExitTimestamp", "MemoryCurrent", "MemoryPeak",
]
_PROGRESS_RE = re.compile(r"news progress (\d+)/(\d+) (\S+) new_total=(\d+)")


def available() -> bool:
    return shutil.which("systemctl") is not None


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_show(text: str) -> dict[str, str]:
    """`systemctl show -p A -p B` output -> {A: value, B: value}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def human_bytes(value: str) -> str:
    if not value.isdigit():
        return "-"
    n = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def unit_info(unit: str) -> dict[str, str]:
    args = ["systemctl", "show", f"{unit}.service"] + [f"-p{p}" for p in _SHOW_PROPS]
    return parse_show(_run(args))


def timer_info(timer: str) -> dict[str, str]:
    props = ["ActiveState", "NextElapseUSecRealtime", "LastTriggerUSec"]
    return parse_show(_run(["systemctl", "show", timer] + [f"-p{p}" for p in props]))


def recent_log(unit: str, lines: int = 15) -> str:
    return _run(["journalctl", "-u", f"{unit}.service", "-n", str(lines), "--no-pager", "-o", "short-iso"])


def parse_backfill(text: str) -> dict:
    """Parse the key=value status file B uploads; adds done/total/pct from the progress line."""
    info = parse_show(text)
    m = _PROGRESS_RE.search(info.get("last", ""))
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        info.update(done=done, total=total, ticker=m.group(3),
                    new_total=int(m.group(4)), pct=done / total if total else 0.0)
    elif info.get("state") == "done":
        info.update(pct=1.0)
    return info


def backfill_status(path: Path = BACKFILL_STATUS) -> dict | None:
    try:
        return parse_backfill(path.read_text())
    except OSError:
        return None


_STEP_RE = re.compile(r"^=== (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.+?) ===$")
_STEP_NAMES = [
    ("refresh_data.py", "Refresh data (prices, news, macro, features)"),
    ("incremental_train.py", "Incremental training"),
    ("--signals-only", "Generate signals"),
    ("train_models.py", "Full retrain"),
    ("precompute_dashboard.py", "Backtests + signals"),
    ("applying news upload", "Apply news from Server B"),
]


def _step_name(raw: str) -> str:
    for needle, name in _STEP_NAMES:
        if needle in raw:
            return name
    return raw


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def summarize_run(lines: list[str], running: bool, now: datetime | None = None) -> dict:
    """Turn one run's log lines into steps + error/warning counts.

    Steps come from the `=== <utc> <what> ===` markers run_pipeline.sh prints;
    a marker ending in "done" closes the run.
    """
    marks = []
    for line in lines:
        m = _STEP_RE.match(line.strip())
        if m:
            marks.append((datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ"), m.group(2)))
    steps = []
    finished = bool(marks) and marks[-1][1].endswith("done")
    body = marks[:-1] if finished else marks
    for i, (ts, raw) in enumerate(body):
        end = marks[i + 1][0] if i + 1 < len(marks) else None
        if end is None:
            status = "running" if running else "failed"
            end_for_dur = now or datetime.utcnow()
        else:
            status, end_for_dur = "done", end
        steps.append({
            "step": _step_name(raw),
            "started (UTC)": ts.strftime("%m-%d %H:%M"),
            "duration": _fmt_duration((end_for_dur - ts).total_seconds()),
            "status": status,
        })
    errors = sum(1 for l in lines if " ERROR " in l or "Traceback" in l)
    warnings = sum(1 for l in lines if " WARNING " in l)
    return {"steps": steps, "errors": errors, "warnings": warnings, "finished": finished}


def last_run_lines(unit: str) -> list[str]:
    """Message lines of the unit's most recent run only."""
    inv = parse_show(_run(["systemctl", "show", f"{unit}.service", "-pInvocationID"])).get("InvocationID", "")
    if not inv:
        # systemd forgets a finished unit's invocation after a daemon-reload; the journal keeps it.
        inv = last_invocation_from_journal(_run([
            "journalctl", "-u", f"{unit}.service", "-n", "200", "--no-pager", "-o", "json",
            "--output-fields=_SYSTEMD_INVOCATION_ID,INVOCATION_ID",
        ]))
    if not inv:
        return []
    out = _run(["journalctl", f"_SYSTEMD_INVOCATION_ID={inv}", "--no-pager", "-o", "cat"])
    return out.splitlines()


def last_invocation_from_journal(json_lines: str) -> str:
    """Newest invocation id in `journalctl -o json` output (one JSON object per line)."""
    inv = ""
    for line in json_lines.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        inv = rec.get("_SYSTEMD_INVOCATION_ID") or rec.get("INVOCATION_ID") or inv
    return inv
