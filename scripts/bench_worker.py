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
import subprocess
import threading
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


def main(request_path: Path, feature_dir: Path, out_dir: Path, upload, load=load_training_data,
         loader=load_strategy, heartbeat_s: float | None = 600.0, clock=None) -> dict:
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
        marks = checkpoints(len(names))
        for i, name in enumerate(names, start=1):
            if clock() >= deadline:
                skipped.extend(names[i - 1:])
                logger.warning("deadline reached, %d methods left", len(names) - i + 1)
                break
            logger.info("bench %s (B) %d/%d: %s", market, i, len(names), name)
            try:
                bp.run_method(name, data, market, today, out_dir, loader, ran_on="B")
                done.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s failed: %s", name, exc)
                skipped.append(name)
            if i in marks:
                report()
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
