"""Bench re-test: every non-retired method of a market on the last 3 years, then league update.

Runs from the weekly Sun 13:00 ET timer; exits at once unless the last completed
pass is >= bench_min_gap_days old (so it really runs every 2nd week), or --force.

With `remote_bench.<market>.enabled` in config/league.yaml (and the outgoing dir
present, i.e. on the server), part of that market's methods goes to the compute
server B: `--publish-only` posts the request before the other market runs, B works
on it in parallel (scripts/bench_worker.py), and the normal run for the market
imports B's verified results (src/league/remote_bench.py).

Usage: python scripts/bench_pass.py --market us [--deadline ISO] [--force] [--seed-from-pass]
       python scripts/bench_pass.py --market china --publish-only --head-start-market us
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from dashboard.ui_config import FEATURE_COLS, OHLCV_COLS
from src.backtesting.limit_filter import blocked_entries
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.league import league as lg
from src.league import remote_bench as rb
from src.league.scoring import score_predictions
from src.strategies.registry import list_strategies, load_strategy

warnings.filterwarnings("ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be used")
logger = logging.getLogger(__name__)

_TRAIN_WINDOW_DAYS = 400
_PRED_SCHEMA = {"ticker": pl.Utf8, "time": pl.Date, "buy_score": pl.Float64}
_OUTGOING_ROOT = Path(os.environ.get("FSP_BENCH_OUTGOING", "/home/fspsync/outgoing"))
_INBOX_ROOT = Path(os.environ.get("FSP_BENCH_INBOX", "/home/fspsync/incoming/bench"))
_REQUEST_MAX_AGE = timedelta(hours=20)
_REPO = Path(__file__).resolve().parents[1]


class Terminated(Exception):
    """Raised from the SIGTERM handler so a systemd stop still saves finished results."""


def _on_sigterm(signum, frame):
    raise Terminated()


def _last_runtime(bench_dir: Path, name: str) -> float:
    try:
        return float(json.loads((bench_dir / f"bench_{_safe(name)}.json").read_text(encoding="utf-8"))["runtime_s"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0.0


def test_order(names: list[str], state: dict) -> list[str]:
    """Never-tested (or stopped-before) methods first, then the oldest test date."""
    def last(n: str) -> str:
        m = state["methods"].get(n)
        return m["history"][-1]["date"] if m and m["history"] else ""
    return sorted(names, key=last)


def _safe(name: str) -> str:
    return name.replace("/", "_")


def prepare(market: str, today: date, feature_dir: Path, load=load_training_data) -> dict:
    """The data every method of one pass is tested on (shared with the B worker)."""
    window = lg.load_config()["bench_window_days"]
    t0 = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    test_start = t0 - timedelta(days=window)
    df = load(feature_dir, start=test_start - timedelta(days=_TRAIN_WINDOW_DAYS))
    outcomes = df.select("ticker", pl.col("time").dt.date().alias("time"), "forward_return_5d")
    return {"df": df, "outcomes": outcomes, "blocked": blocked_entries(market, df),
            "test_start": test_start, "window": window}


def _write_result(bench_dir: Path, name: str, today: date, window: int, score: dict, p: pl.DataFrame,
                  metrics: dict, runtime_s: float | None, ran_on: str) -> None:
    p.write_parquet(bench_dir / f"preds_{_safe(name)}.parquet")
    (bench_dir / f"bench_{_safe(name)}.json").write_text(json.dumps({
        "strategy_name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tested_on": today.isoformat(),
        "window_days": window,
        "score": score,
        **metrics,
        "runtime_s": runtime_s,
        "ran_on": ran_on,
    }, indent=2), encoding="utf-8")


def run_method(name: str, data: dict, market: str, today: date, bench_dir: Path,
               loader=load_strategy, ran_on: str = "A") -> tuple[dict, pl.DataFrame]:
    """Walk-forward test one method; writes its bench json + preds. Raises on failure."""
    mcfg = get_market(market)
    t_start = time.monotonic()
    log: list = []
    wf = walk_forward_backtest_strategy(
        data["df"], loader(name), OHLCV_COLS, FEATURE_COLS,
        train_window_days=_TRAIN_WINDOW_DAYS, test_window_days=21, step_days=21,
        start=data["test_start"], pred_log=log, blocked_entries=data["blocked"],
    )
    p = pl.DataFrame(log, schema=_PRED_SCHEMA, orient="row")
    score = score_predictions(p, data["outcomes"], cost_bps=mcfg.cost_bps)
    metrics = {"mean_sharpe": wf.mean_sharpe, "weighted_precision_buy": wf.weighted_precision_buy,
               "worst_drawdown": wf.worst_drawdown, "n_folds": len(wf.folds)}
    _write_result(bench_dir, name, today, data["window"], score, p, metrics,
                  round(time.monotonic() - t_start, 1), ran_on)
    logger.info("  %s net_sharpe=%.2f trades=%d", name, score["net_sharpe"], score["n_trades"])
    return score, p


def _code_tarball(dest: Path) -> Path | None:
    """The exact code A runs, for B: tracked files only (no market data, no .env)."""
    try:
        subprocess.run(["git", "-C", str(_REPO), "archive", "--format=tar.gz", "-o", str(dest),
                        "HEAD", "--", ".", ":(exclude)markets"], check=True, capture_output=True, timeout=120)
        return dest
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("bench: code tarball failed: %s", exc)
        return None


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _remote_cfg(market: str) -> dict | None:
    rc = (lg.load_config().get("remote_bench") or {}).get(market)
    return rc if rc and rc.get("enabled") and _OUTGOING_ROOT.is_dir() else None


def _candidates(state: dict, strategies: list[str] | None, market: str) -> list[str]:
    retired = {n for n, m in state["methods"].items() if m["status"] == "retired"}
    return test_order([n for n in (strategies or list_strategies(market)) if n not in retired], state)


def _state(market: str, cache_dir: Path) -> dict:
    return lg.load_state(cache_dir, market, seed=list(get_market(market).default_strategies or ()))


def publish_if_due(market: str, today: date, deadline: datetime | None = None,
                   strategies: list[str] | None = None, force: bool = False,
                   head_start_market: str | None = None, cache_dir_for=None, publish_fn=None) -> dict | None:
    """--publish-only: post B's share when this market's pass is due. Returns the request or None."""
    cache_dir_for = cache_dir_for or (lambda m: get_market(m).data_root / "cache")
    publish_fn = publish_fn or publish
    if _remote_cfg(market) is None:
        logger.info("bench %s: remote bench off, nothing to post", market)
        return None
    last = lg.last_completed_pass(_state(market, cache_dir_for(market)))
    if not force and last is not None and (today - last).days < lg.load_config()["bench_min_gap_days"]:
        logger.info("bench %s: not due, nothing to post", market)
        return None
    head = 0.0
    if head_start_market:
        hcache = cache_dir_for(head_start_market)
        hst = _state(head_start_market, hcache)
        head = sum(_last_runtime(hcache / "bench", n) for n in _candidates(hst, None, head_start_market))
    return publish_fn(market, today, deadline or datetime.now(timezone.utc) + timedelta(hours=15),
                      strategies, cache_dir=cache_dir_for(market), head_start_s=head)


def publish(market: str, today: date, deadline: datetime, strategies: list[str] | None = None,
            cfg: dict | None = None, cache_dir: Path | None = None, feature_dir: Path | None = None,
            outgoing: Path | None = None, runtimes: dict | None = None, head_start_s: float = 0.0,
            code_tar="auto", load=load_training_data) -> dict:
    """Post B's share of this market's pass (request + feature snapshot + code)."""
    mcfg = get_market(market)
    cfg = cfg or _remote_cfg(market) or {}
    cache_dir = cache_dir or mcfg.data_root / "cache"
    feature_dir = feature_dir or mcfg.data_root / "features"
    outgoing = outgoing or _OUTGOING_ROOT / market
    state = lg.load_state(cache_dir, market, seed=list(mcfg.default_strategies or ()))
    names = _candidates(state, strategies, market)
    if runtimes is None:
        runtimes = {n: _last_runtime(cache_dir / "bench", n) for n in names}
    local, remote = rb.split_methods(
        names, runtimes, local_head_start_s=head_start_s, remote_speed=float(cfg.get("remote_speed", 1.0)),
        exclude=cfg.get("exclude", []), max_share=float(cfg.get("max_share", 0.8)))
    now = datetime.now(timezone.utc)
    req = {"id": f"{market}-{today.isoformat()}-{int(now.timestamp())}", "market": market,
           "today": today.isoformat(), "deadline": deadline.isoformat(), "methods": remote,
           "local": local, "created": now.isoformat(), "commit": _commit()}
    # Only the rows this pass reads, in one zstd file: the link to B is slow
    # (~30 KB/s), and the full history back to 1991 is ~5x bigger.
    build = outgoing.parent / f".build-{market}"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    t0 = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    window = lg.load_config()["bench_window_days"]
    (load(feature_dir, start=t0 - timedelta(days=window + _TRAIN_WINDOW_DAYS + 5))
     .write_parquet(build / "features.parquet", compression="zstd", compression_level=19))
    tar = None
    if code_tar == "auto":
        outgoing.mkdir(parents=True, exist_ok=True)
        tar = _code_tarball(outgoing / "code.build.tar.gz")
    elif code_tar is not None:
        tar = Path(code_tar)
    rb.publish_request(outgoing, req, build, tar)
    shutil.rmtree(build, ignore_errors=True)
    if code_tar == "auto" and tar is not None:
        tar.unlink(missing_ok=True)
    logger.info("bench %s: posted %d methods for B (%s), %d stay on A", market, len(remote), req["id"], len(local))
    return req


def main(
    market: str,
    deadline: datetime | None = None,
    strategies: list[str] | None = None,
    force: bool = False,
    seed_from_pass: bool = False,
    today: date | None = None,
    feature_dir: Path | None = None,
    cache_dir: Path | None = None,
    load=load_training_data,
    loader=load_strategy,
    remote="auto",
    outgoing: Path | None = None,
    inbox_root: Path | None = None,
    poll_s: float = 60.0,
    sleep=time.sleep,
    clock=None,
) -> dict:
    cfg = lg.load_config()
    mcfg = get_market(market)
    feature_dir = feature_dir or mcfg.data_root / "features"
    cache_dir = cache_dir or mcfg.data_root / "cache"
    clock = clock or (lambda: datetime.now(timezone.utc))
    remote = _remote_cfg(market) if remote == "auto" else remote
    outgoing = outgoing or _OUTGOING_ROOT / market
    req = None
    if remote:
        req = rb.load_request(outgoing / rb.REQUEST_FILE)
        if req is not None and (req["market"] != market
                                or clock() - datetime.fromisoformat(req["created"]) > _REQUEST_MAX_AGE):
            req = None
        if req is not None:  # same pass as the posted request (it may have crossed midnight UTC)
            today = date.fromisoformat(req["today"])
    today = today or datetime.now(timezone.utc).date()

    state = lg.load_state(cache_dir, market, seed=list(mcfg.default_strategies or ()))
    last = lg.last_completed_pass(state)
    if not force and last is not None and (today - last).days < cfg["bench_min_gap_days"]:
        logger.info("bench %s: last pass %s, skipping", market, last)
        return {"skipped": "recent"}

    names = _candidates(state, strategies, market)
    if remote and req is None:
        req = publish(market, today, deadline or clock() + timedelta(hours=15), strategies, remote,
                      cache_dir, feature_dir, outgoing, load=load)
    remote_names = [n for n in (req["methods"] if req else []) if n in names]

    data = prepare(market, today, feature_dir, load)
    bench_dir = cache_dir / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    preds: dict[str, pl.DataFrame] = {}
    stopped = False
    skipped_for_time: list[str] = []

    def run_local(todo: list[str]) -> None:
        nonlocal stopped
        for i, name in enumerate(todo, start=1):
            now = clock()
            if deadline is not None and now >= deadline:
                logger.warning("bench %s: deadline reached, %d methods left", market, len(todo) - i + 1)
                stopped = True
                break
            need = _last_runtime(bench_dir, name) * 1.2
            if deadline is not None and now + timedelta(seconds=need) > deadline:
                logger.warning("bench %s: %s needs ~%.0f min, not enough time left; next pass", market, name, need / 60)
                skipped_for_time.append(name)
                stopped = True
                continue
            logger.info("bench %s %d/%d: %s", market, i, len(todo), name)
            try:
                results[name], preds[name] = run_method(name, data, market, today, bench_dir, loader)
            except Exception as exc:
                logger.error("bench %s: %s failed: %s", market, name, exc)
                results[name] = {"error": str(exc)[:300]}

    got_remote: list[str] = []
    taken_back: list[str] = []
    try:
        run_local([n for n in names if n not in remote_names])
        if remote_names and not stopped:
            got_remote, taken_back = _collect_remote(
                market, req, remote_names, data, today, bench_dir, results, preds,
                inbox_root or _INBOX_ROOT / market, float(remote.get("stale_minutes", 30)) * 60,
                deadline, poll_s, sleep, clock, run_local, float(remote.get("start_minutes", 90)) * 60)
        if set(remote_names) - set(got_remote) - set(taken_back):
            stopped = True  # B did not deliver everything: the pass is incomplete
    except Terminated:
        logger.warning("bench %s: stopped by SIGTERM, saving %d finished results", market, len(results))
        stopped = True
    finally:
        if req is not None:
            rb.retract_request(outgoing)

    if results:  # a pass that tested nothing must not count (or reset streaks)
        lg.update_league(state, results, preds, today, cfg, completed=not stopped)
    if seed_from_pass and results:
        lg.seed_live_from_latest(state, cfg, n=cfg["live_max"] - 1)
    retired_dir = cache_dir / "retired"
    for n, m in state["methods"].items():
        src = bench_dir / f"bench_{_safe(n)}.json"
        if m["status"] == "retired" and src.exists():
            retired_dir.mkdir(exist_ok=True)
            shutil.move(str(src), retired_dir / src.name)
    lg.save_state(cache_dir, state)

    live = [n for n, m in state["methods"].items() if m["status"] == "live"]
    logger.info("bench %s done: live=%s", market, live)
    return {
        "tested": [n for n in results if "error" not in results[n]],
        "errors": [n for n in results if "error" in results[n]],
        "stopped": stopped,
        "skipped_for_time": skipped_for_time,
        "live": live,
        "remote": got_remote,
        "taken_back": taken_back,
    }


def _collect_remote(market, req, remote_names, data, today, bench_dir, results, preds, inbox_root,
                    stale_s, deadline, poll_s, sleep, clock, run_local,
                    start_s: float = 5400.0) -> tuple[list[str], list[str]]:
    """Wait for B, importing each verified quarter as it lands; take the rest back if B goes quiet."""
    mcfg = get_market(market)
    inbox = inbox_root / req["id"]
    created = datetime.fromisoformat(req["created"])
    want, got, seen_rejects = set(remote_names), [], set()
    while True:
        m = rb.read_manifest(inbox, req)
        if m is not None:
            new, rejected = rb.import_results(inbox, req, m, skip=set(got))
            for r in rejected:
                if r not in seen_rejects:
                    logger.error("bench %s: rejected result from B: %s", market, r)
                    seen_rejects.add(r)
            for name, item in new.items():
                if name not in want:
                    continue
                p, meta = item["preds"], item["meta"]
                score = score_predictions(p, data["outcomes"], cost_bps=mcfg.cost_bps)  # A's own score
                runtime = meta.pop("runtime_s")
                _write_result(bench_dir, name, today, data["window"], score, p, meta, runtime, "B")
                results[name], preds[name] = score, p
                got.append(name)
                logger.info("  %s (from B, %.0f min) net_sharpe=%.2f trades=%d", name,
                            (runtime or 0) / 60, score["net_sharpe"], score["n_trades"])
        left = [n for n in remote_names if n not in got]
        if not left:
            return got, []
        if m is not None and m.get("complete"):
            logger.warning("bench %s: B finished without %s", market, left)
            return got, []
        now = clock()
        if deadline is not None and now >= deadline:
            logger.warning("bench %s: deadline reached waiting for B; %s untested", market, left)
            return got, []
        if m is not None:
            silent = not rb.remote_alive(m, now, stale_s)
        else:
            silent = (now - created).total_seconds() > max(stale_s, start_s)  # B still downloading?
        if silent:
            logger.warning("bench %s: B silent for %.0f min; running %s here", market, stale_s / 60, left)
            run_local(left)
            return got, [n for n in left if n in results]
        sleep(poll_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=sorted(MARKETS))
    ap.add_argument("--deadline", type=datetime.fromisoformat)
    ap.add_argument("--strategies", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed-from-pass", action="store_true")
    ap.add_argument("--publish-only", action="store_true", help="post B's share and exit")
    ap.add_argument("--head-start-market", help="A runs this market first; B's share grows to match")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    if a.publish_only:
        publish_if_due(a.market, datetime.now(timezone.utc).date(), deadline=a.deadline,
                       strategies=a.strategies, force=a.force, head_start_market=a.head_start_market)
    else:
        main(a.market, deadline=a.deadline, strategies=a.strategies, force=a.force, seed_from_pass=a.seed_from_pass)
