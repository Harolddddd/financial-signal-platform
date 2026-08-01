"""Run each strategy's live-signal computation in its own subprocess so
memory is fully released to the OS between strategies, instead of one long
loop where a leak (wherever it is) can accumulate across 31 strategies.

A background thread polls available memory every 1s while each subprocess
runs; if it drops below _MIN_FREE_GB, the subprocess is killed immediately
(not just logged) to prevent another full-system crash, and the run stops
so the culprit strategy can be inspected.

Speed: the live-features/training data (~5-8s to build) is computed ONCE
here and cached to parquet for every subprocess to read, instead of each of
31 subprocesses rebuilding it from scratch. Also resumable — strategies
that already have a data/cache/signals_partial/{name}.json are skipped, so
a killed/interrupted run can just be re-invoked (pass --force to redo all).

After all strategies succeed, merges data/cache/signals_partial/*.json into
the final data/cache/signals.json (same schema step_signals() would write).
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from config.markets import get_market
from dashboard.config import PARQUET_DIR
from scripts.build_features import build_live_features
from src.features.duckdb_client import load_training_data
from src.strategies.registry import list_strategies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CACHE_DIR = get_market("us").data_root / "cache"
_PARTIAL_DIR = _CACHE_DIR / "signals_partial"
_LIVE_CACHE = _CACHE_DIR / "_tmp_live_features.parquet"
_TRAIN_CACHE = _CACHE_DIR / "_tmp_training_data.parquet"
_SIGNALS_PATH = _CACHE_DIR / "signals.json"
_MIN_FREE_GB = 3.0
_POLL_SECONDS = 1.0
_EXCLUDE = {"knn_classifier"}


class _MemoryWatchdog:
    """Kills `proc` if available memory drops below _MIN_FREE_GB while armed."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.triggered = False
        self.min_seen_gb = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            self.min_seen_gb = min(self.min_seen_gb, available_gb)
            if available_gb < _MIN_FREE_GB and self.proc is not None and self.proc.poll() is None:
                logger.error(
                    "  MEMORY WATCHDOG: only %.2fGB free — killing subprocess now", available_gb,
                )
                self.triggered = True
                self.proc.kill()
            time.sleep(_POLL_SECONDS)

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    force = "--force" in sys.argv
    _PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    names = [n for n in list_strategies() if n not in _EXCLUDE]

    if force:
        for f in _PARTIAL_DIR.glob("*.json"):
            f.unlink()

    already_done = {f.stem for f in _PARTIAL_DIR.glob("*.json")}
    todo = [n for n in names if n not in already_done]
    logger.info("=== Isolated signal run: %d strategies, %d already done, %d to run, watchdog floor=%.1fGB ===",
                len(names), len(already_done), len(todo), _MIN_FREE_GB)

    if todo:
        logger.info("Building shared live-features/training-data cache once...")
        t0 = time.time()
        build_live_features().write_parquet(_LIVE_CACHE)
        load_training_data(PARQUET_DIR).write_parquet(_TRAIN_CACHE)
        logger.info("  cache built in %.0fs", time.time() - t0)

    watchdog = _MemoryWatchdog()
    completed: list[str] = list(already_done)
    try:
        for name in todo:
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < _MIN_FREE_GB:
                logger.error("Aborting before %s — only %.2fGB free", name, available_gb)
                break

            out_path = _PARTIAL_DIR / f"{name}.json"
            logger.info("--- %s (free=%.2fGB) ---", name, available_gb)
            t0 = time.time()
            watchdog.triggered = False
            watchdog.min_seen_gb = float("inf")
            proc = subprocess.Popen(
                [sys.executable, "-u", "scripts/signal_one_strategy.py", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            watchdog.proc = proc
            proc.wait()
            watchdog.proc = None
            elapsed = time.time() - t0

            if watchdog.triggered:
                logger.error(
                    "%s KILLED by watchdog after %.0fs (min free seen: %.2fGB) — stopping run",
                    name, elapsed, watchdog.min_seen_gb,
                )
                break
            if proc.returncode != 0:
                logger.error("%s exited with code %d after %.0fs — stopping run",
                              name, proc.returncode, elapsed)
                break
            if not out_path.exists():
                logger.error("%s completed but wrote no output — stopping run", name)
                break

            logger.info("%s OK — %.0fs, min free during run: %.2fGB",
                        name, elapsed, watchdog.min_seen_gb)
            completed.append(name)
    finally:
        watchdog.stop()
        _LIVE_CACHE.unlink(missing_ok=True)
        _TRAIN_CACHE.unlink(missing_ok=True)

    logger.info("=== %d/%d strategies completed cleanly ===", len(completed), len(names))
    if len(completed) < len(names):
        missing = [n for n in names if n not in completed]
        logger.warning("Not merging signals.json — missing: %s. Re-run this script to resume.", missing)
        return

    _merge_and_write()


def _merge_and_write() -> None:
    all_signals: list[dict] = []
    for f in sorted(_PARTIAL_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        all_signals.extend(data["signals"])

    buy_count = sum(1 for s in all_signals if s["signal"] == "Buy")
    logger.info("  total signals: %d  buy: %d", len(all_signals), buy_count)
    _SIGNALS_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": all_signals,
    }, indent=2, default=str))
    logger.info("  wrote %s", _SIGNALS_PATH)


if __name__ == "__main__":
    main()
