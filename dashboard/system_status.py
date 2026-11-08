"""Read-only status of the server's systemd jobs and the compute server's backfill.

Works on the Linux production server; elsewhere `available()` is False and the
page shows a notice instead.
"""
from __future__ import annotations

import re
import shutil
import subprocess
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
