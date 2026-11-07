"""Run a long pipeline step (training, signals, refresh) in its own console
window, outside the VS Code extension host, so an extension-host crash
(`JavaScript heap out of memory`) cannot take the job down with it.

Output is mirrored to logs/jobs/<name>.log and the job state is kept in
logs/jobs/<name>.status.json, so progress can be checked from any shell
(or by Claude) while the window runs.

Usage:
    python scripts/run_job.py train_us -- scripts/train_models.py --market us
    python scripts/run_job.py signals_us -- scripts/precompute_dashboard.py --market us --signals-only
    python scripts/run_job.py --status              # every job, one line each
    python scripts/run_job.py --status train_us     # one job + tail of its log
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_JOBS_DIR = _ROOT / "logs" / "jobs"

_CREATE_NEW_CONSOLE = 0x00000010
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _paths(name: str) -> tuple[Path, Path]:
    return _JOBS_DIR / f"{name}.log", _JOBS_DIR / f"{name}.status.json"


def _read_status(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(path: Path, status: dict) -> None:
    # Write-then-rename so a reader never sees a half-written file.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def launch(name: str, cmd: list[str]) -> None:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path, status_path = _paths(name)
    if status_path.exists():
        prev = _read_status(status_path)
        if prev["state"] in ("starting", "running") and _pid_alive(prev["worker_pid"]):
            sys.exit(f"job {name!r} is already running (pid {prev['worker_pid']})")

    argv = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--", *cmd]
    try:
        # Breakaway leaves the job object Claude Code / VS Code may have put
        # this shell in, so tearing down that tree does not kill the job.
        proc = subprocess.Popen(argv, cwd=_ROOT, creationflags=_CREATE_NEW_CONSOLE | _CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        # The parent's job object forbids breakaway — still gets its own console.
        proc = subprocess.Popen(argv, cwd=_ROOT, creationflags=_CREATE_NEW_CONSOLE)

    _write_status(status_path, {
        "name": name, "cmd": cmd, "state": "starting",
        "worker_pid": proc.pid, "child_pid": None,
        "started_at": _now(), "finished_at": None, "exit_code": None,
        "log": str(log_path),
    })
    print(f"started {name} (pid {proc.pid}) in a new window")
    print(f"  log:    {log_path}")
    print(f"  status: python scripts/run_job.py --status {name}")


def work(name: str, cmd: list[str]) -> None:
    log_path, status_path = _paths(name)
    os.system(f"title {name}")

    status = {
        "name": name, "cmd": cmd, "state": "running",
        "worker_pid": os.getpid(), "child_pid": None,
        "started_at": _now(), "finished_at": None, "exit_code": None,
        "log": str(log_path),
    }
    # Pipeline scripts import `config`, `src`, `scripts` from the repo root.
    pythonpath = os.pathsep.join(p for p in (str(_ROOT), os.environ.get("PYTHONPATH")) if p)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": pythonpath}

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"# {_now()} started: python {' '.join(cmd)}\n")
        child = subprocess.Popen(
            [sys.executable, "-u", *cmd], cwd=_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        status["child_pid"] = child.pid
        _write_status(status_path, status)
        try:
            for raw in child.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                log.write(line + "\n")
                print(line, flush=True)
            code = child.wait()
            state = "done" if code == 0 else "failed"
        except KeyboardInterrupt:
            code = child.wait()
            state = "interrupted"
        log.write(f"# {_now()} {state} exit={code}\n")

    status.update(state=state, exit_code=code, finished_at=_now())
    _write_status(status_path, status)
    print(f"\n=== {name}: {state} (exit {code}) ===")
    if state != "done":
        input("Press Enter to close this window...")


def show_status(name: str | None, tail: int) -> None:
    if name:
        files = [_paths(name)[1]]
    else:
        files = sorted(_JOBS_DIR.glob("*.status.json")) if _JOBS_DIR.exists() else []
    if not files:
        print("no jobs")
        return

    for path in files:
        if not path.exists():
            print(f"no job {name!r}")
            continue
        s = _read_status(path)
        state = s["state"]
        if state in ("starting", "running") and not _pid_alive(s["worker_pid"]):
            state = "died (window closed or process killed)"
        print(f"{s['name']}: {state}  exit={s['exit_code']}  started={s['started_at']}  finished={s['finished_at']}")
        print(f"  cmd: python {' '.join(s['cmd'])}")
        print(f"  log: {s['log']}")
        if name and Path(s["log"]).exists():
            lines = Path(s["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"  --- last {tail} log lines ---")
            for line in lines[-tail:]:
                print(f"  {line}")


if __name__ == "__main__":
    # Logs contain non-cp1252 text (e.g. "→"); don't crash printing them.
    sys.stdout.reconfigure(errors="replace")
    argv = sys.argv[1:]
    cmd: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(description="Run a pipeline step in its own console window.")
    parser.add_argument("name", nargs="?", help="job name, e.g. train_us")
    parser.add_argument("--status", action="store_true", help="show job state instead of starting one")
    parser.add_argument("--tail", type=int, default=15, help="log lines shown by --status NAME")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.status:
        show_status(args.name, args.tail)
    elif not args.name or not cmd:
        parser.error("usage: run_job.py NAME -- <script.py> [args...]")
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("NAME may only contain letters, digits, '_', '.', '-'")
    elif args.worker:
        work(args.name, cmd)
    else:
        launch(args.name, cmd)
