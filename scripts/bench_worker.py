"""Compute server (B): run the bench methods A assigned, uploading results every quarter.

A posts request.json (+ a feature snapshot) in /home/fspsync/outgoing/<market>/;
deploy/bench_worker.sh pulls it read-only and starts this script. After each
quarter of the assigned methods (and every `heartbeat_s` as a sign of life) the
finished results plus a sha256 manifest go to A's write-only inbox, so a crash on
B loses at most the quarter in progress, and A takes the rest back.

Usage: python scripts/bench_worker.py --request R.json --features DIR --out DIR --upload-to DEST
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from pathlib import Path

from scripts import bench_pass as bp
from src.features.duckdb_client import load_training_data
from src.league import remote_bench as rb
from src.strategies.registry import load_strategy

logger = logging.getLogger(__name__)


def checkpoints(n: int) -> set[int]:
    """Method counts after which to upload: each quarter of n (fewer if n < 4)."""
    return {math.ceil(k * n / 4) for k in range(1, 5)} - {0}


def _cgroup_headroom_mb() -> float | None:
    """MB left under this process's cgroup memory cap (systemd MemoryMax), or None."""
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        cg = Path("/sys/fs/cgroup") / rel.lstrip("/")
        cap = (cg / "memory.max").read_text().strip()
        if cap == "max":
            return None
        return (int(cap) - int((cg / "memory.current").read_text())) / 2**20
    except (OSError, IndexError, ValueError):
        return None


def memory_ok(reserve_mb: float | None = None) -> bool:
    """Room to start one more method: >= reserve MB under the cgroup cap (else under system RAM)."""
    reserve_mb = reserve_mb if reserve_mb is not None else float(os.environ.get("BENCH_MEM_RESERVE_MB", "800"))
    free = _cgroup_headroom_mb()
    if free is None:
        import psutil
        free = psutil.virtual_memory().available / 2**20
    return free >= reserve_mb


def main(request_path: Path, feature_dir: Path, out_dir: Path, upload, load=load_training_data,
         loader=load_strategy, heartbeat_s: float | None = 600.0, clock=None,
         parallel: int | None = None, mem_ok=memory_ok) -> dict:
    """Run the assigned methods, up to `parallel` at once (threads sharing the data).

    A further method starts only while `mem_ok()` says there is room; otherwise
    it waits for a running one to finish.
    """
    parallel = max(1, parallel if parallel is not None else int(os.environ.get("BENCH_PARALLEL", "1")))
    clock = clock or (lambda: datetime.now(timezone.utc))
    req = rb.load_request(request_path)
    if req is None:
        raise SystemExit(f"no valid request at {request_path}")
    market, today = req["market"], date.fromisoformat(req["today"])
    deadline = datetime.fromisoformat(req["deadline"])
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(req["methods"])
    done: list[str] = []
    skipped: list[str] = []
    lock = threading.Lock()
    stop = threading.Event()

    def report(complete: bool = False) -> None:
        with lock:
            rb.write_manifest(out_dir, req, done, complete=complete, skipped=skipped)
            try:
                upload(out_dir)
            except Exception as exc:  # noqa: BLE001 - next checkpoint retries the upload
                logger.warning("upload failed (retried at the next checkpoint): %s", exc)

    def beat() -> None:
        while not stop.wait(heartbeat_s):
            report()

    report()  # A sees B has started
    hb = None
    if heartbeat_s:
        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
    try:
        data = bp.prepare(market, today, feature_dir, load)
        marks, fired = checkpoints(len(names)), set()
        pending = list(names)
        running: dict = {}
        started = 0

        def run_one(name: str) -> None:
            bp.run_method(name, data, market, today, out_dir, loader, ran_on="B")

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            while pending or running:
                while pending and len(running) < parallel and (not running or mem_ok()):
                    if clock() >= deadline:
                        logger.warning("deadline reached, %d methods left", len(pending))
                        with lock:
                            skipped.extend(pending)
                        pending.clear()
                        break
                    name = pending.pop(0)
                    started += 1
                    logger.info("bench %s (B) %d/%d: %s", market, started, len(names), name)
                    running[ex.submit(run_one, name)] = name
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    name = running.pop(f)
                    try:
                        f.result()
                        with lock:
                            done.append(name)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("%s failed: %s", name, exc)
                        with lock:
                            skipped.append(name)
                    n = len(done) + len(skipped)
                    for m in sorted(marks - fired):
                        if n >= m:
                            fired.add(m)
                            report()
                            break
    finally:
        stop.set()
        if hb is not None:
            hb.join(timeout=5)
        report(complete=True)
    return {"done": done, "skipped": skipped}


def rsync_upload(dest: str):
    """Data files first, manifest last, so A never sees a manifest for files still in flight."""
    def up(d: Path) -> None:
        base = ["rsync", "-a", "--timeout=120"]
        subprocess.run(base + ["--exclude", rb.MANIFEST_FILE, "--exclude", "*.tmp", f"{d}/", dest],
                       check=True, timeout=1800)
        subprocess.run(base + [str(d / rb.MANIFEST_FILE), dest], check=True, timeout=300)
    return up


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--upload-to", required=True, help="rsync destination, e.g. fspsync@A:bench/china/<id>/")
    a = ap.parse_args()
    main(a.request, a.features, a.out, rsync_upload(a.upload_to))
