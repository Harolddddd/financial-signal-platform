"""Read-only status of the server's systemd jobs and the compute server's backfill.

Works on the Linux production server; elsewhere `available()` is False and the
page shows a notice instead.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

SERVICES = {
    "fsp-daily": "Daily pipeline (refresh, incremental train, signals)",
    "fsp-weekly": "Weekly full retrain + backtests",
    "fsp-cn-daily": "China daily pipeline (refresh, incremental train, signals)",
    "fsp-cn-weekly": "China weekly full retrain + backtests",
    "fsp-bench": "Bench re-test, both markets (every 2nd Sunday)",
    "fsp-backup": "Nightly backup",
    "fsp-intraday": "Intraday stop check on held paper stocks (every 5 min, market hours)",
    "fsp-dashboard": "Dashboard",
}
TIMERS = [
    "fsp-daily.timer", "fsp-weekly.timer", "fsp-cn-daily.timer", "fsp-cn-weekly.timer",
    "fsp-bench.timer", "fsp-backup.timer", "fsp-intraday.timer",
]
BACKFILL_STATUS = Path("/home/fspsync/incoming/status/backfill.txt")
CN_NEWS_STATUS = Path("/home/fspsync/incoming/status/cn_news.txt")
SERVER_B_STATUS = Path("/home/fspsync/incoming/status/server_b.json")

# Compute server (B): pushes a snapshot of these every 5 min (scripts/server_status.py).
SERVICES_B = {
    "fsp-cn-news": "China news + macro collect, then sync to A (daily 16:30 Beijing)",
    "fsp-bench-worker": "Bench worker: B's share of a bench pass from A",
    "fsp-backfill": "US news backfill (one-off)",
    "fsp-status-b": "Status report to dashboard (every 5 min)",
}
TIMERS_B = ["fsp-cn-news.timer", "fsp-bench-worker.timer", "fsp-status-b.timer"]
# Always-on or self: never listed under "running now".
QUIET_UNITS = {"fsp-dashboard", "fsp-status-b"}
SNAPSHOT_STALE_MIN = 15
_SHOW_PROPS = [
    "ActiveState", "SubState", "Result", "ExecMainStartTimestamp",
    "ExecMainExitTimestamp", "MemoryCurrent", "MemoryPeak",
]
ET = ZoneInfo("America/New_York")
CST = ZoneInfo("Asia/Shanghai")
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


def parse_extra_units(list_units: str, known: dict | None = None) -> list[str]:
    """Running fsp-* services that are not in SERVICES (one-off jobs started by hand)."""
    out = []
    for line in list_units.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].endswith(".service") and parts[2] in ("active", "activating"):
            name = parts[0][: -len(".service")]
            if name not in (SERVICES if known is None else known):
                out.append(name)
    return out


def extra_units(known: dict | None = None) -> list[str]:
    return parse_extra_units(_run([
        "systemctl", "list-units", "fsp-*", "--type=service", "--all", "--no-legend", "--plain",
    ]), known)


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
    ("applying china upload", "Apply China news from Server B"),
    ("paper_trade.py", "Paper account trades"),
]


def _step_name(raw: str) -> str:
    if "bench_pass.py" in raw:
        m = re.search(r"--market (\S+)", raw)
        return f"Bench re-test ({m.group(1) if m else '?'})"
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
            "started (ET)": ts.replace(tzinfo=timezone.utc).astimezone(ET).strftime("%m-%d %H:%M %Z"),
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


_BENCH_PROGRESS_RE = re.compile(r"bench \S+ (?:\(B\) )?(\d+)/(\d+): (\S+)")


def parse_progress(lines: list[str]) -> dict | None:
    """Latest 'bench <market> [(B)] i/n: method' or 'news progress i/n ticker' line -> done/total/item/pct."""
    for line in reversed(lines):
        m = _BENCH_PROGRESS_RE.search(line)
        if m:  # 'i/n: x' means x is the i-th, still running
            done, total = int(m.group(1)) - 1, int(m.group(2))
            return {"done": done, "total": total, "item": m.group(3), "pct": done / total if total else 0.0}
        m = _PROGRESS_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            return {"done": done, "total": total, "item": m.group(3), "pct": done / total if total else 0.0}
    return None


def _utc_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect(host: str, services: dict[str, str], timers: list[str], now: datetime | None = None) -> dict:
    """One server's status: resources, jobs (with last-run steps and progress) and timers."""
    import psutil

    now = now or datetime.now(timezone.utc)
    mem, disk = psutil.virtual_memory(), psutil.disk_usage("/")
    units = {}
    for unit, label in {**services, **{u: f"One-off: {u.removeprefix('fsp-')}" for u in extra_units(services)}}.items():
        info = unit_info(unit)
        running = info.get("ActiveState") in ("activating", "active") and unit not in QUIET_UNITS
        lines = last_run_lines(unit) if unit not in QUIET_UNITS else []
        units[unit] = {
            "label": label, **info, "running": running,
            "summary": summarize_run(lines, running=running, now=now.replace(tzinfo=None)),
            "progress": parse_progress(lines) if running else None,
            "log": recent_log(unit, 30) if unit not in QUIET_UNITS else "",
        }
    return {
        "host": host,
        "updated": _utc_iso(now),
        "resources": {
            "cpu_pct": psutil.cpu_percent(interval=0.5), "cpus": psutil.cpu_count(),
            "mem_used": mem.total - mem.available, "mem_total": mem.total,
            "disk_used": disk.used, "disk_total": disk.total,
        },
        "units": units,
        "timers": {t: timer_info(t) for t in timers},
    }


def read_snapshot(path: Path = SERVER_B_STATUS) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def is_stale(snapshot: dict, now: datetime | None = None, minutes: int = SNAPSHOT_STALE_MIN) -> bool:
    try:
        ts = datetime.strptime(snapshot["updated"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return True
    return ((now or datetime.now(timezone.utc)) - ts).total_seconds() > minutes * 60


_TZ_ABBR = {"UTC": timezone.utc, "EST": ET, "EDT": ET, "CST": CST}


def systemd_to_et(value: str) -> str:
    """'Mon 2026-11-09 17:50:06 UTC' (any server's local zone) -> '11-09 12:50 EST'."""
    if not value:
        return "-"
    parts = value.split()
    if len(parts) != 4 or parts[3] not in _TZ_ABBR:
        return value
    try:
        ts = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=_TZ_ABBR[parts[3]])
    except ValueError:
        return value
    return ts.astimezone(ET).strftime("%m-%d %H:%M %Z")


def plain_state(info: dict) -> str:
    s = info.get("ActiveState", "")
    return {"activating": "running", "active": "running", "deactivating": "stopping",
            "failed": "failed", "inactive": "idle"}.get(s, s or "?")


def to_et(utc_iso: str) -> str:
    """'2026-11-08T02:47:12Z' -> '11-07 21:47 EST'; anything unparsable is returned unchanged."""
    try:
        ts = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return utc_iso
    return ts.astimezone(ET).strftime("%m-%d %H:%M %Z")


def beijing_time(utc_iso: str) -> str:
    """'2026-11-08T02:47:12Z' -> '11-08 10:47 CST'; anything unparsable is returned unchanged."""
    try:
        ts = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return utc_iso
    return ts.astimezone(CST).strftime("%m-%d %H:%M CST")
